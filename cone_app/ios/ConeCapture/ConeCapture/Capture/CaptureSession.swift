import Foundation
import ARKit
import SwiftUI

// =====================================================================================================
// HARD RULE — ARKit is used ONLY for (a) the camera image stream (`frame.capturedImage`) and
// (b) per-frame camera intrinsics (`frame.camera.intrinsics`, `frame.camera.imageResolution`).
// This app NEVER reads or sends `frame.camera.transform`, `eulerAngles`, `viewMatrix`, projection-derived
// pose, anchors, or any other tracking pose. ARWorldTrackingConfiguration only runs because ARSession
// requires a configuration; every optional feature is disabled below.
// =====================================================================================================

/// Receives ARFrames on a private serial queue (NOT the main thread).
final class FrameSink: NSObject, ARSessionDelegate {
    struct Snapshot {
        var resolution: CGSize
        var fps: Double
        var intrinsics: CameraIntrinsics
        var recordElapsed: Double
        var encoded: Int
        var dropped: Int
    }

    var onSnapshot: (@Sendable (Snapshot) -> Void)?
    var onError: (@Sendable (String) -> Void)?

    private let lock = NSLock()
    private var recorder: VideoRecorder?
    private var lastTimestamp: TimeInterval = 0
    private var smoothedFPS: Double = 0
    private var lastHUD: TimeInterval = 0

    func attach(_ recorder: VideoRecorder?) {
        lock.lock(); self.recorder = recorder; lock.unlock()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let ts = frame.timestamp
        if lastTimestamp > 0, ts > lastTimestamp {
            let inst = 1.0 / (ts - lastTimestamp)
            smoothedFPS = smoothedFPS == 0 ? inst : smoothedFPS * 0.9 + inst * 0.1
        }
        lastTimestamp = ts

        lock.lock(); let rec = recorder; lock.unlock()
        rec?.append(frame: frame)

        if ts - lastHUD >= 0.25 {
            lastHUD = ts
            let st = rec?.stats ?? (0, 0, 0)
            let snap = Snapshot(
                resolution: frame.camera.imageResolution,
                fps: smoothedFPS,
                intrinsics: IntrinsicsConverter.intrinsics(fromColumnMajor: frame.camera.intrinsics),
                recordElapsed: st.0, encoded: st.1, dropped: st.2)
            onSnapshot?(snap)
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        onError?("AR session failed: \(error.localizedDescription)")
    }

    func sessionWasInterrupted(_ session: ARSession) {
        onError?("Camera session interrupted.")
    }
}

@MainActor
final class CaptureSession: ObservableObject {
    enum Mode: String, CaseIterable, Identifiable {
        case photo = "Photo"
        case video = "Video"
        var id: String { rawValue }
    }

    @Published var mode: Mode = .video {
        didSet { if oldValue != mode, !isRecording { applyConfiguration() } }
    }
    @Published private(set) var hud: FrameSink.Snapshot?
    @Published private(set) var isRecording = false
    @Published private(set) var isBusy = false          // photo capture or video finalising
    @Published private(set) var focusLabel = "Locked"
    @Published private(set) var activeFormatLabel = ""
    @Published var errorMessage: String?
    @Published var toast: String?

    let arSession = ARSession()
    private let sink = FrameSink()
    private let sessionQueue = DispatchQueue(label: "cone.capture.session", qos: .userInitiated)

    private weak var settings: AppSettings?
    private weak var store: RecordingStore?
    private weak var uploader: Uploader?

    private struct PendingVideo {
        var id: String
        var folder: URL
        var createdAt: Date
        var recorder: VideoRecorder
        var name: String?
    }
    private var pendingVideo: PendingVideo?

    init() {
        sink.onSnapshot = { [weak self] snap in
            Task { @MainActor in self?.hud = snap }
        }
        sink.onError = { [weak self] msg in
            Task { @MainActor in self?.errorMessage = msg }
        }
        arSession.delegateQueue = sessionQueue
        arSession.delegate = sink
    }

    func configure(settings: AppSettings, store: RecordingStore, uploader: Uploader) {
        self.settings = settings
        self.store = store
        self.uploader = uploader
    }

    // MARK: Session configuration

    /// Which configuration: ARWorldTrackingConfiguration, with everything optional disabled.
    /// AROrientationTrackingConfiguration was considered (it does less work), but ARKit only documents the
    /// 4K / high-res-still "recommended" formats on ARWorldTrackingConfiguration, and its supportedVideoFormats
    /// list is the one guaranteed to contain the 3840x2160 format on iPhone 14 Pro.
    // VERIFY: if AROrientationTrackingConfiguration.supportedVideoFormats also lists 4K on your device, it can
    // be swapped in here (same property names) to reduce CPU use. We never read the pose either way.
    func applyConfiguration() {
        guard let settings else { return }
        let cfg = ARWorldTrackingConfiguration()
        cfg.planeDetection = []
        cfg.sceneReconstruction = []
        cfg.environmentTexturing = .none
        cfg.wantsHDREnvironmentTextures = false
        cfg.frameSemantics = []
        cfg.isLightEstimationEnabled = false
        cfg.isAutoFocusEnabled = settings.autofocus
        cfg.isCollaborationEnabled = false
        cfg.automaticImageScaleEstimationEnabled = false
        cfg.maximumNumberOfTrackedImages = 0
        cfg.userFaceTrackingEnabled = false
        cfg.videoHDRAllowed = false

        var format = settings.selectedFormat
        if mode == .photo, settings.photoMaxResolution, let still = VideoFormats.recommendedHighResStill {
            format = still
        }
        if let format { cfg.videoFormat = format }

        focusLabel = settings.autofocus ? "Auto" : "Locked"
        activeFormatLabel = format.map(VideoFormats.label) ?? "default"
        arSession.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    func start() { applyConfiguration() }
    func pause() { arSession.pause() }

    // MARK: Video

    func toggleRecording() {
        isRecording ? stopRecording() : startRecording()
    }

    private func startRecording() {
        guard mode == .video, !isRecording, !isBusy, let store, let settings else { return }
        do {
            let (id, folder) = try store.makeFolder()
            let recorder = VideoRecorder(outputURL: folder.appendingPathComponent("video.mp4"))
            pendingVideo = PendingVideo(id: id, folder: folder, createdAt: Date(),
                                        recorder: recorder, name: settings.trimmedName)
            sink.attach(recorder)
            isRecording = true
        } catch {
            errorMessage = "Cannot start recording: \(error.localizedDescription)"
        }
    }

    func stopRecordingIfNeeded() {
        if isRecording { stopRecording() }
    }

    func stopRecording() {
        guard isRecording, let p = pendingVideo else { return }
        isRecording = false
        isBusy = true
        sink.attach(nil)
        pendingVideo = nil
        p.recorder.finish { [weak self] result in
            Task { @MainActor in self?.finishVideo(result, pending: p) }
        }
    }

    private func finishVideo(_ result: Result<VideoRecorder.Output, Error>, pending p: PendingVideo) {
        isBusy = false
        switch result {
        case .success(let out):
            let manifest = CaptureManifest(kind: .video, device: .current, createdAt: p.createdAt,
                                           imageSize: out.imageSize, frames: out.frames,
                                           name: p.name, notes: nil)
            do {
                try commit(id: p.id, folder: p.folder, mediaName: "video.mp4", manifest: manifest)
                toast = "Saved \(out.frames.count) frames (\(out.droppedFrames) dropped)"
            } catch {
                errorMessage = "Could not save capture.json: \(error.localizedDescription)"
            }
        case .failure(let error):
            try? FileManager.default.removeItem(at: p.folder)
            errorMessage = error.localizedDescription
        }
    }

    // MARK: Photo

    func takePhoto() {
        guard mode == .photo, !isBusy, let store, let settings else { return }
        isBusy = true
        let name = settings.trimmedName
        let session = arSession
        Task {
            defer { isBusy = false }
            do {
                let photo = try await PhotoCapture.capture(session: session)
                let (id, folder) = try store.makeFolder()
                try photo.jpegData.write(to: folder.appendingPathComponent("photo.jpg"), options: .atomic)
                let manifest = CaptureManifest(
                    kind: .photo, device: .current, createdAt: Date(), imageSize: photo.imageSize,
                    frames: [.init(index: 0, t: 0.0, K: photo.intrinsics.rowMajorK)],
                    name: name, notes: nil)
                try commit(id: id, folder: folder, mediaName: "photo.jpg", manifest: manifest)
                toast = "Photo saved \(photo.imageSize.width)\u{00D7}\(photo.imageSize.height)"
            } catch {
                errorMessage = "Photo failed: \(error.localizedDescription)"
            }
        }
    }

    // MARK: Commit

    private func commit(id: String, folder: URL, mediaName: String, manifest: CaptureManifest) throws {
        guard let store, let settings else { return }
        try manifest.jsonData().write(to: folder.appendingPathComponent("capture.json"), options: .atomic)
        let attrs = try? FileManager.default.attributesOfItem(atPath: folder.appendingPathComponent(mediaName).path)
        let bytes = (attrs?[.size] as? NSNumber)?.int64Value ?? 0
        store.add(Recording(id: id, kind: manifest.kind, createdAt: manifest.createdAt, name: manifest.name,
                            mediaFileName: mediaName, mediaBytes: bytes, status: .pending,
                            jobID: nil, errorMessage: nil, attempts: 0, mediaDeleted: false))
        if settings.autoUpload { uploader?.upload(id) }
    }
}
