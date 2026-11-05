import Foundation
import ImageIO

/// Clockwise quarter turns baked into the media, fixed when a capture starts.
enum MediaRotation: Int, CaseIterable {
    case none = 0
    case clockwise90 = 90
    case clockwise180 = 180
    case clockwise270 = 270

    /// CoreMotion uses portrait device axes; the rear camera's native pixels are landscape.
    /// Preserve the last useful orientation when the camera points almost straight up/down.
    static func fromGravity(x: Double, y: Double, fallback: MediaRotation) -> MediaRotation {
        guard hypot(x, y) >= 0.15 else { return fallback }
        if abs(x) > abs(y) { return x < 0 ? .none : .clockwise180 }
        return y < 0 ? .clockwise90 : .clockwise270
    }

    var imageOrientation: CGImagePropertyOrientation {
        switch self {
        case .none: return .up
        case .clockwise90: return .right
        case .clockwise180: return .down
        case .clockwise270: return .left
        }
    }

    func outputSize(for size: PixelSize) -> PixelSize {
        switch self {
        case .none, .clockwise180: return size
        case .clockwise90, .clockwise270: return .init(width: size.height, height: size.width)
        }
    }

    /// Rotate both pixel axes and the camera basis to retain a standard positive-focal-length K.
    /// Pixel centers use integer coordinates: the last column is width - 1.
    func intrinsics(_ k: CameraIntrinsics, sourceSize size: PixelSize) -> CameraIntrinsics {
        let w = Double(size.width - 1)
        let h = Double(size.height - 1)
        switch self {
        case .none: return k
        case .clockwise90: return .init(fx: k.fy, fy: k.fx, cx: h - k.cy, cy: k.cx)
        case .clockwise180: return .init(fx: k.fx, fy: k.fy, cx: w - k.cx, cy: h - k.cy)
        case .clockwise270: return .init(fx: k.fy, fy: k.fx, cx: k.cy, cy: w - k.cx)
        }
    }
}
