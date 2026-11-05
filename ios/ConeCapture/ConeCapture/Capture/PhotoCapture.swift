import Foundation
import ARKit
import CoreImage
import ImageIO
import simd

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

    static func capture(session: ARSession, rotation: MediaRotation, quality: CGFloat = 0.95) async throws -> CapturedPhoto {
        let frame = try await highResFrame(session)
        return try await Task.detached(priority: .userInitiated) {
            try encode(pixelBuffer: frame.capturedImage, intrinsics: frame.camera.intrinsics,
                       referenceResolution: frame.camera.imageResolution, rotation: rotation, quality: quality)
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

    static func encode(pixelBuffer pb: CVPixelBuffer, intrinsics: simd_float3x3,
                       referenceResolution: CGSize, rotation: MediaRotation,
                       quality: CGFloat = 0.95) throws -> CapturedPhoto {
        let sourceSize = PixelSize(width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb))
        let nativeK = IntrinsicsConverter.storedIntrinsics(columnMajor: intrinsics,
                                                          referenceResolution: referenceResolution,
                                                          storedSize: sourceSize)
        let size = rotation.outputSize(for: sourceSize)
        let k = rotation.intrinsics(nativeK, sourceSize: sourceSize)
        // Bake quarter turns into JPEG pixels; consumers need no EXIF auto-rotation.
        let ci = CIImage(cvPixelBuffer: pb).oriented(rotation.imageOrientation)
        let ctx = CIContext()
        guard let cs = CGColorSpace(name: CGColorSpace.sRGB) else { throw PhotoError.encodeFailed }
        let key = CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String)
        guard let data = ctx.jpegRepresentation(of: ci, colorSpace: cs, options: [key: quality]) else {
            throw PhotoError.encodeFailed
        }
        return CapturedPhoto(jpegData: data, imageSize: size, intrinsics: k)
    }
}
