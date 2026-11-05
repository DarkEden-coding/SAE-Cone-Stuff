import Foundation
import ARKit
import CoreGraphics
import ConeSLAM

/// What the capture screen draws for one processed live frame (view coordinates, points).
struct LiveDisplay {
    struct Cone: Identifiable {
        var id: Int
        var classID: Int
        var box: CGRect
        var keypoints: [CGPoint?]
        var landmarkID: Int?
        var rangeM: Double?
    }
    var status: TrackingStatus = .waiting
    var cones: [Cone] = []
    var minimap = MinimapSnapshot()
    var fps = 0.0
    var detMs = 0.0
    var slamMs = 0.0
    var nLandmarks = 0
    var nInliers = 0
    var skipped = 0
}

/// Pure conversions between the camera's native buffer, the gravity-upright inference image and CoreMotion.
enum LiveGeometry {
    /// World up in the camera frame of the rotated image, from CoreMotion gravity (device frame, pointing
    /// down). The native buffer's camera axes are (-device y, -device x, -device z) for the rear camera.
    static func upInCamera(gravity g: SIMD3<Double>, rotation: MediaRotation) -> SIMD3<Double>? {
        let n = simd_length(g)
        guard n > 0.5 else { return nil }
        let v = SIMD3(g.y, g.x, g.z) / n
        switch rotation {
        case .none: return v
        case .clockwise90: return SIMD3(-v.y, v.x, v.z)
        case .clockwise180: return SIMD3(-v.x, -v.y, v.z)
        case .clockwise270: return SIMD3(v.y, -v.x, v.z)
        }
    }

    /// Normalised (0...1) rotated-image coordinates back to normalised native-buffer coordinates.
    static func nativeNormalized(_ p: CGPoint, rotation: MediaRotation) -> CGPoint {
        switch rotation {
        case .none: return p
        case .clockwise90: return CGPoint(x: p.y, y: 1 - p.x)
        case .clockwise180: return CGPoint(x: 1 - p.x, y: 1 - p.y)
        case .clockwise270: return CGPoint(x: 1 - p.y, y: p.x)
        }
    }

    /// Inference-image intrinsics: native K (ARKit reference resolution -> buffer), rotated, then scaled.
    static func intrinsics(native k: CameraIntrinsics, bufferSize: PixelSize, rotation: MediaRotation,
                           imageSize: PixelSize) -> CameraIntrinsics {
        let rotated = rotation.intrinsics(k, sourceSize: bufferSize)
        let s = Double(imageSize.width) / Double(rotation.outputSize(for: bufferSize).width)
        return CameraIntrinsics(fx: rotated.fx * s, fy: rotated.fy * s, cx: rotated.cx * s, cy: rotated.cy * s)
    }
}

/// On-device live mode: CoreML detection -> keypoint correction -> cone-only SLAM, streamed to the server
/// and logged next to recordings. ARKit supplies only images, intrinsics and the 2D display transform for
/// drawing overlays; CoreMotion gravity is the SLAM's tilt prior. Never reads any ARKit pose.
final class LivePipeline: @unchecked Sendable {   // state is lock- or queue-confined
    struct Config {
        var rotation: MediaRotation
        var conf: Double
        var keypointCorrection: Bool
        var stream: Bool
        var baseURL: String
        var token: String?
        var name: String?
    }

    var onDisplay: ((LiveDisplay) -> Void)?
    let sessionID = UUID().uuidString.lowercased()
    let streamer: LiveStreamer?

    private let config: Config
    var rotation: MediaRotation { config.rotation }
    private let gravity: GravityHistory
    private let detector: ConeDetector
    private let correction: KeypointCorrection?
    private let slam = LiveSLAM()
    private let queue = DispatchQueue(label: "cone.live.process", qos: .userInitiated)
    private let startedAt = Date()

    // Cross-thread state (lock).
    private let lock = NSLock()
    private var busy = false
    private var stopped = false
    private var viewport = CGSize.zero
    private var orientation = UIInterfaceOrientation.portrait
    private var skipped = 0
    private var pendingReset = false

    // Processing-queue state.
    private var t0: TimeInterval?
    private var frameCount = 0
    private var helloData: Data?
    private var lastMapT = -1.0
    private var latestMap: Data?
    private var smoothedFPS = 0.0
    private var lastFrameT: Double?
    private var trail: [SIMD2<Double>] = []
    private var recording: (id: String, log: LiveLogWriter?, recorder: VideoRecorder)?

    init(config: Config, gravity: GravityHistory) throws {
        self.config = config
        self.gravity = gravity
        let portrait = config.rotation == .clockwise90 || config.rotation == .clockwise270
        detector = try ConeDetector(portrait: portrait)
        if config.keypointCorrection, let url = Bundle.main.url(forResource: "kp_correction", withExtension: "json") {
            // Table bins are model-input pixels; the inference image is the model input minus letterbox.
            correction = try? KeypointCorrection(json: Data(contentsOf: url))
        } else {
            correction = nil
        }
        streamer = config.stream && !config.baseURL.isEmpty ? LiveStreamer(baseURL: config.baseURL, token: config.token) : nil
    }

    // MARK: Main thread

    func setViewport(_ size: CGSize, orientation: UIInterfaceOrientation) {
        lock.lock(); viewport = size; self.orientation = orientation; lock.unlock()
    }

    func resetMap() {
        lock.lock(); pendingReset = true; lock.unlock()
    }

    // MARK: AR session queue

    /// Non-blocking: drops the frame while the previous one is still being processed.
    func handle(_ frame: ARFrame) {
        lock.lock()
        if busy || stopped {
            if busy { skipped += 1 }
            lock.unlock()
            return
        }
        busy = true
        let size = viewport, orientation = orientation
        lock.unlock()
        let buffer = frame.capturedImage
        let timestamp = frame.timestamp
        let intrinsics = IntrinsicsConverter.intrinsics(fromColumnMajor: frame.camera.intrinsics)
        let reference = frame.camera.imageResolution
        let display = size.width > 0 ? frame.displayTransform(for: orientation, viewportSize: size) : .identity
        queue.async { [self] in
            autoreleasepool {
                process(buffer, timestamp: timestamp, intrinsics: intrinsics, reference: reference,
                        display: display, viewport: size)
            }
            lock.lock(); busy = false; lock.unlock()
        }
    }

    // MARK: Recording (main thread calls; work hops to the processing queue)

    func startRecording(id: String, folder: URL, name: String?, recorder: VideoRecorder) {
        queue.async { [self] in
            let log = try? LiveLogWriter(url: folder.appendingPathComponent("live.jsonl"))
            recording = (id, log, recorder)
            if let helloData { log?.append(helloData) }
            if let latestMap { log?.append(latestMap) }
            emit(LiveCoding.encode(LiveRecordingEvent(t: now(), event: "start", recordingID: id, name: name)), kind: .control)
        }
    }

    /// Completion runs on the processing queue once the log is closed.
    func stopRecording(completion: (() -> Void)? = nil) {
        queue.async { [self] in
            if let r = recording {
                if let latestMap { r.log?.append(latestMap) }
                emit(LiveCoding.encode(LiveRecordingEvent(t: now(), event: "stop", recordingID: r.id, name: nil)), kind: .control)
                r.log?.close()
            }
            recording = nil
            completion?()
        }
    }

    func stop() {
        lock.lock(); stopped = true; lock.unlock()
        queue.async { [self] in
            if recording != nil { recording?.log?.close(); recording = nil }
            streamer?.stop(bye: LiveCoding.encode(LiveBye(t: now())))
        }
    }

    // MARK: Processing queue

    private func now() -> Double { Date().timeIntervalSince(startedAt) }

    private func emit(_ data: Data, kind: LiveStreamer.Kind) {
        streamer?.send(data, kind: kind)
        recording?.log?.append(data)
    }

    private func process(_ buffer: CVPixelBuffer, timestamp: TimeInterval, intrinsics: CameraIntrinsics,
                         reference: CGSize, display: CGAffineTransform, viewport: CGSize) {
        lock.lock()
        let reset = pendingReset
        pendingReset = false
        let skippedFrames = skipped
        lock.unlock()
        if reset { slam.reset(); trail = [] }
        if t0 == nil { t0 = timestamp - now() }
        let t = timestamp - t0!
        let rotation = config.rotation
        let bufferSize = PixelSize(width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer))
        let imageSize = detector.imageSize
        let nativeK = IntrinsicsConverter.scaled(intrinsics, from: reference, to: bufferSize)
        let k = LiveGeometry.intrinsics(native: nativeK, bufferSize: bufferSize, rotation: rotation, imageSize: imageSize)

        let detStart = Date()
        var detections: [Detection]
        do {
            detections = try detector.detect(buffer, rotation: rotation, conf: config.conf)
        } catch {
            detections = []
        }
        let detMs = Date().timeIntervalSince(detStart) * 1000
        var rangeScales: [Double] = []
        if let correction {
            detections = detections.map { correction.apply($0) }
            rangeScales = detections.map { correction.rangeScale(boxHeight: $0.box.w - $0.box.y) }
        }

        let up = gravity.gravity(at: timestamp).flatMap { LiveGeometry.upInCamera(gravity: $0, rotation: rotation) }
        let slamStart = Date()
        let K = Intrinsics(fx: k.fx, fy: k.fy, cx: k.cx, cy: k.cy)
        let result = slam.process(SLAMFrame(t: t, K: K, imageSize: Vec2(Double(imageSize.width), Double(imageSize.height)),
                                            up: up, detections: detections, rangeScales: rangeScales.isEmpty ? nil : rangeScales))
        let slamMs = Date().timeIntervalSince(slamStart) * 1000
        if let last = lastFrameT, t > last {
            let fps = 1 / (t - last)
            smoothedFPS = smoothedFPS == 0 ? fps : 0.85 * smoothedFPS + 0.15 * fps
        }
        lastFrameT = t

        if helloData == nil {
            let hello = LiveHello(sessionID: sessionID, t: t, startedAt: CaptureManifest.iso8601String(startedAt),
                                  name: config.name, device: liveDevice(), imageSize: imageSize.asArray, K: k.rowMajorK,
                                  model: ["name": .string("conepose-yolo26n"),
                                          "imgsz": .array([.number(Double(detector.inputSize.height)), .number(Double(detector.inputSize.width))])],
                                  options: ["conf": .number(config.conf), "kp_conf": .number(slam.options.kpConf),
                                            "kp_correction": .bool(correction != nil), "flat_ground": .bool(slam.options.flatGround),
                                            "gravity_prior": .bool(slam.options.gravitySigma != nil)])
            let data = LiveCoding.encode(hello)
            helloData = data
            streamer?.start(hello: data)
            recording?.log?.append(data)
        }

        var rec: LiveRecordingRef?
        if let r = recording, let first = r.recorder.firstFrameTimestamp {
            rec = LiveRecordingRef(recordingID: r.id, t: timestamp - first)
        }
        let liveDetections = detections.enumerated().map { i, d in
            LiveDetection(classID: d.classID, conf: d.conf, box: [d.box.x, d.box.y, d.box.z, d.box.w],
                          keypoints: d.keypoints.map { [$0.x, $0.y, $0.z] },
                          pCam: result.measurements[i].map { [$0.pCam.x, $0.pCam.y, $0.pCam.z] },
                          landmarkID: result.landmarkOf[i])
        }
        let message = LiveFrame(i: frameCount, t: t, status: result.status.rawValue, T: result.pose?.matrix, K: k.rowMajorK,
                                imageSize: imageSize.asArray, detections: liveDetections, nInliers: result.nInliers, rec: rec,
                                stats: LiveStats(detMs: detMs, slamMs: slamMs, fps: smoothedFPS))
        frameCount += 1
        emit(LiveCoding.encode(message), kind: .frame)

        let landmarks = slam.landmarks
        if result.mapAdjusted || t - lastMapT >= 0.5 {
            lastMapT = t
            let map = LiveMap(t: t, landmarks: landmarks.map {
                LiveLandmark(id: $0.id, classID: $0.classID, p: [$0.p.x, $0.p.y, $0.p.z],
                             sigma: [$0.sigma.x, $0.sigma.y, $0.sigma.z], nObs: $0.nObs, confirmed: $0.confirmed)
            }, keyframes: slam.keyframes.map { LiveKeyframe(i: $0.index, t: $0.t, T: $0.pose.matrix) })
            let data = LiveCoding.encode(map)
            latestMap = data
            emit(data, kind: .map)
        }

        if let pose = result.pose {
            trail.append(SIMD2(pose.t.x, pose.t.y))
            if trail.count > 600 { trail.removeFirst(trail.count - 600) }
        }
        var snapshot = MinimapSnapshot(status: result.status.rawValue)
        snapshot.cones = landmarks.map { .init(id: $0.id, classID: $0.classID, x: $0.p.x, y: $0.p.y, confirmed: $0.confirmed) }
        snapshot.camera = result.pose.map { .init(x: $0.t.x, y: $0.t.y, yaw: ConeSLAM.yaw(of: $0)) }
        snapshot.trail = trail

        let toView = { (p: SIMD2<Double>) -> CGPoint in
            let normalized = CGPoint(x: p.x / Double(imageSize.width), y: p.y / Double(imageSize.height))
            let n = LiveGeometry.nativeNormalized(normalized, rotation: rotation).applying(display)
            return CGPoint(x: n.x * viewport.width, y: n.y * viewport.height)
        }
        let cones = detections.enumerated().map { i, d -> LiveDisplay.Cone in
            let a = toView(SIMD2(d.box.x, d.box.y)), b = toView(SIMD2(d.box.z, d.box.w))
            let points = d.keypoints.prefix(ConeModel.nKeypoints(d.classID)).map { kp -> CGPoint? in
                kp.z >= slam.options.kpConf ? toView(SIMD2(kp.x, kp.y)) : nil
            }
            return .init(id: i, classID: d.classID,
                         box: CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(b.x - a.x), height: abs(b.y - a.y)),
                         keypoints: Array(points), landmarkID: result.landmarkOf[i],
                         rangeM: result.measurements[i].map { simd_length($0.pCam) })
        }
        let displayState = LiveDisplay(status: result.status, cones: cones, minimap: snapshot, fps: smoothedFPS, detMs: detMs,
                                       slamMs: slamMs, nLandmarks: landmarks.filter(\.confirmed).count,
                                       nInliers: result.nInliers, skipped: skippedFrames)
        if let onDisplay {
            DispatchQueue.main.async { onDisplay(displayState) }
        }
    }

    private func liveDevice() -> LiveDevice {
        let d = CaptureManifest.Device.current
        return LiveDevice(model: d.model, system: d.system, appVersion: d.appVersion)
    }
}
