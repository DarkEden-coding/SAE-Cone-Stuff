import Foundation
import ARKit
import AVFoundation
import CoreVideo
import CoreImage
import simd

/// H.264 MP4 writer with a gravity-selected quarter turn baked into every encoded frame.
///
/// Threading:
///  - `append(frame:)` is called on the ARSession delegate queue. It only does cheap bookkeeping and
///    hands the pixel buffer to `writerQueue` (backpressure: at most `maxInFlight` buffers queued, else drop).
///  - All AVAssetWriter work and the `frames` array live exclusively on `writerQueue`.
///  - A frame's intrinsics are appended to `frames` ONLY when the pixel buffer was actually appended to the
///    encoder, so `frames[i]` corresponds 1:1 to encoded frame i.
///
/// HARD RULE: from the ARFrame we read only `capturedImage`, `timestamp`, `camera.intrinsics` and
/// `camera.imageResolution`. Never camera.transform / eulerAngles / viewMatrix / projection / anchors.
final class VideoRecorder {
    struct Output {
        var frames: [CaptureManifest.Frame]
        var imageSize: PixelSize
        var droppedFrames: Int
    }

    enum RecorderError: LocalizedError {
        case noFrames
        case writer(String)
        var errorDescription: String? {
            switch self {
            case .noFrames: return "No frames were recorded."
            case .writer(let m): return "Video writer failed: \(m)"
            }
        }
    }

    private static let timescale: CMTimeScale = 90_000
    private static let maxInFlight = 3

    private let outputURL: URL
    private let rotation: MediaRotation
    private lazy var imageContext = CIContext(options: [.cacheIntermediates: false])
    private let writerQueue = DispatchQueue(label: "cone.capture.writer", qos: .userInitiated)

    // Cross-thread counters (protected by `lock`).
    private let lock = NSLock()
    private var accepting = true
    private var inFlight = 0
    private var droppedCount = 0
    private var encodedCount = 0
    private var elapsedSeconds: Double = 0

    // writerQueue-only state.
    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var firstTimestamp: TimeInterval?
    private var lastPTS: CMTime?
    private var frames: [CaptureManifest.Frame] = []
    private var imageSize = PixelSize(width: 0, height: 0)
    private var sourceSize = PixelSize(width: 0, height: 0)
    private var failure: Error?

    init(outputURL: URL, rotation: MediaRotation) {
        self.outputURL = outputURL
        self.rotation = rotation
    }

    /// Thread-safe live stats for the HUD.
    var stats: (elapsed: Double, encoded: Int, dropped: Int) {
        lock.lock(); defer { lock.unlock() }
        return (elapsedSeconds, encodedCount, droppedCount)
    }

    // MARK: Session-queue entry point

    func append(frame: ARFrame) {
        append(pixelBuffer: frame.capturedImage, timestamp: frame.timestamp,
               intrinsics: frame.camera.intrinsics, referenceResolution: frame.camera.imageResolution)
    }

    func append(pixelBuffer: CVPixelBuffer, timestamp: TimeInterval,
                intrinsics: simd_float3x3, referenceResolution reference: CGSize) {
        lock.lock()
        guard accepting else { lock.unlock(); return }
        if inFlight >= Self.maxInFlight {
            droppedCount += 1
            lock.unlock()
            return
        }
        inFlight += 1
        // Enqueue while holding the lock so finish() can never be overtaken by a late append.
        writerQueue.async { [self] in
            autoreleasepool {
                process(pixelBuffer, timestamp: timestamp, intrinsics: intrinsics, reference: reference)
            }
            lock.lock(); inFlight -= 1; lock.unlock()
        }
        lock.unlock()
    }

    // MARK: Writer queue

    private func process(_ pb: CVPixelBuffer, timestamp: TimeInterval, intrinsics: simd_float3x3, reference: CGSize) {
        if failure != nil { noteDrop(); return }
        if writer == nil {
            do { try setUpWriter(pb, firstTimestamp: timestamp) } catch { failure = error; noteDrop(); return }
        }
        guard let writer, let input, let adaptor, writer.status == .writing,
              let t0 = firstTimestamp else { noteDrop(); return }

        let pts = CMTime(seconds: timestamp - t0, preferredTimescale: Self.timescale)
        if let last = lastPTS, pts <= last { noteDrop(); return }       // keep PTS strictly increasing
        guard input.isReadyForMoreMediaData else { noteDrop(); return }  // encoder backpressure

        let encodedBuffer: CVPixelBuffer
        do { encodedBuffer = try orientedBuffer(pb, adaptor: adaptor) }
        catch { failure = error; noteDrop(); return }
        if adaptor.append(encodedBuffer, withPresentationTime: pts) {
            let nativeK = IntrinsicsConverter.storedIntrinsics(columnMajor: intrinsics,
                                                         referenceResolution: reference,
                                                         storedSize: sourceSize)
            let k = rotation.intrinsics(nativeK, sourceSize: sourceSize)
            let t = Double(pts.value) / Double(pts.timescale)       // exactly the PTS written to the MP4
            frames.append(.init(index: frames.count, t: t, K: k.rowMajorK))
            lastPTS = pts
            lock.lock(); encodedCount = frames.count; elapsedSeconds = t; lock.unlock()
        } else {
            failure = writer.error ?? RecorderError.writer("append failed")
            noteDrop()
        }
    }

    private func noteDrop() {
        lock.lock(); droppedCount += 1; lock.unlock()
    }

    private func orientedBuffer(_ pb: CVPixelBuffer,
                                adaptor: AVAssetWriterInputPixelBufferAdaptor) throws -> CVPixelBuffer {
        if rotation == .none { return pb }
        guard let pool = adaptor.pixelBufferPool else {
            throw RecorderError.writer("rotated frame buffer pool unavailable")
        }
        var output: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &output)
        guard status == kCVReturnSuccess, let output else {
            throw RecorderError.writer("cannot allocate rotated frame (\(status))")
        }
        let image = CIImage(cvPixelBuffer: pb).oriented(rotation.imageOrientation)
        imageContext.render(image, to: output,
                            bounds: CGRect(x: 0, y: 0, width: imageSize.width, height: imageSize.height),
                            colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return output
    }

    private func setUpWriter(_ pb: CVPixelBuffer, firstTimestamp ts: TimeInterval) throws {
        try? FileManager.default.removeItem(at: outputURL)
        sourceSize = PixelSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        imageSize = rotation.outputSize(for: sourceSize)
        let w = imageSize.width
        let h = imageSize.height

        // ~50 Mbps at 3840x2160, scaled by pixel count for other formats.
        let bitrate = max(8_000_000, Int(50_000_000.0 * Double(w * h) / Double(3840 * 2160)))
        let settings: [String: Any] = [
            AVVideoCodecKey: AVVideoCodecType.h264,          // H.264 only: the server cannot decode HEVC
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoProfileLevelKey: AVVideoProfileLevelH264HighAutoLevel,
                AVVideoMaxKeyFrameIntervalKey: 30,
                AVVideoAllowFrameReorderingKey: false,       // no B-frames -> decode order == PTS order
            ] as [String: Any],
        ]
        let writer = try AVAssetWriter(outputURL: outputURL, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        input.mediaTimeScale = Self.timescale // Preserve the same PTS precision used by capture.json.
        input.transform = .identity // The decoder sees the same rotated pixels described by capture.json.
        let attributes: [String: Any]? = rotation == .none ? nil : [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: attributes)
        guard writer.canAdd(input) else { throw RecorderError.writer("cannot add video input") }
        writer.add(input)
        guard writer.startWriting() else {
            throw RecorderError.writer(writer.error?.localizedDescription ?? "startWriting failed")
        }
        writer.startSession(atSourceTime: .zero)
        self.writer = writer
        self.input = input
        self.adaptor = adaptor
        self.firstTimestamp = ts
    }

    // MARK: Finish

    /// Stops accepting frames, drains the queue, finalises the MP4. Completion runs on an arbitrary queue.
    func finish(completion: @escaping (Result<Output, Error>) -> Void) {
        lock.lock()
        accepting = false
        writerQueue.async { [self] in
            let dropped = stats.dropped
            guard let writer, let input else {
                completion(.failure(failure ?? RecorderError.noFrames))
                return
            }
            if let failure {
                writer.cancelWriting()
                completion(.failure(failure))
                return
            }
            guard writer.status == .writing, !frames.isEmpty else {
                writer.cancelWriting()
                completion(.failure(failure ?? writer.error ?? RecorderError.noFrames))
                return
            }
            let output = Output(frames: frames, imageSize: imageSize, droppedFrames: dropped)
            input.markAsFinished()
            writer.finishWriting {
                if writer.status == .completed {
                    completion(.success(output))
                } else {
                    completion(.failure(RecorderError.writer(writer.error?.localizedDescription ?? "finishWriting failed")))
                }
            }
        }
        lock.unlock()
    }
}
