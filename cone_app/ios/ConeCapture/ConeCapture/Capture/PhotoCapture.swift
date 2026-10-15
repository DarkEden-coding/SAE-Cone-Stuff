import Foundation
import ARKit
import CoreImage
import ImageIO

struct CapturedPhoto {
    var jpegData: Data
    var imageSize: PixelSize
    var intrinsics: CameraIntrinsics   // already in stored-image pixel coordinates
}

/// Still capture via `ARSession.captureHighResolutionFrame` (iOS 16+).
///
/// HARD RULE: only `capturedImage`, `camera.intrinsics` and `camera.imageResolution` are read from the frame.
enum PhotoCapture {
    enum PhotoError: LocalizedError {
        case noFrame
        case encodeFailed
        var errorDescription: String? {
            switch self {
            case .noFrame: return "No camera frame available yet."
            case .encodeFailed: return "JPEG encoding failed."
            }
        }
    }

    static func capture(session: ARSession, quality: CGFloat = 0.95) async throws -> CapturedPhoto {
        let frame = try await highResFrame(session)
        return try await Task.detached(priority: .userInitiated) {
            try encode(frame: frame, quality: quality)
        }.value
    }

    private static func highResFrame(_ session: ARSession) async throws -> ARFrame {
        do {
            // VERIFY: completion-handler form `captureHighResolutionFrame(completion:)`; an async variant also exists.
            return try await withCheckedThrowingContinuation { (cont: CheckedContinuation<ARFrame, Error>) in
                session.captureHighResolutionFrame { frame, error in
                    if let frame { cont.resume(returning: frame) }
                    else { cont.resume(throwing: error ?? PhotoError.noFrame) }
                }
            }
        } catch {
            // Fall back to the latest streamed frame (lower resolution) rather than failing the shot.
            if let f = session.currentFrame { return f }
            throw error
        }
    }

    private static func encode(frame: ARFrame, quality: CGFloat) throws -> CapturedPhoto {
        let pb = frame.capturedImage
        let size = PixelSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        let k = IntrinsicsConverter.storedIntrinsics(columnMajor: frame.camera.intrinsics,
                                                     referenceResolution: frame.camera.imageResolution,
                                                     storedSize: size)
        // YCbCr pixel buffer -> CIImage -> JPEG. No orientation applied: sensor-native landscape.
        let ci = CIImage(cvPixelBuffer: pb)
        let ctx = CIContext()
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { throw PhotoError.encodeFailed }
        let key = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        // VERIFY: jpegRepresentation writes no EXIF orientation tag (input CIImage has identity orientation).
        guard let data = ctx.jpegRepresentation(of: ci, colorSpace: cs, options: [key: quality]) else {
            throw PhotoError.encodeFailed
        }
        return CapturedPhoto(jpegData: data, imageSize: size, intrinsics: k)
    }
}
