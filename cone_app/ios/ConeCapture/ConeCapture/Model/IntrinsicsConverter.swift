import Foundation
import CoreGraphics
import simd

/// Pixel dimensions of a stored image / video frame.
struct PixelSize: Equatable, Codable {
    var width: Int
    var height: Int

    var asArray: [Int] { [width, height] }
}

/// Pinhole intrinsics in pixel units.
struct CameraIntrinsics: Equatable {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double

    /// Row-major 3x3 K exactly as written in capture.json.
    var rowMajorK: [[Double]] {
        [[fx, 0, cx],
         [0, fy, cy],
         [0, 0, 1]]
    }
}

/// Pure conversion helpers (no ARKit dependency, unit-testable).
enum IntrinsicsConverter {

    /// ARKit `camera.intrinsics` is a column-major `simd_float3x3`:
    /// `m[col][row]` -> fx = m[0][0], fy = m[1][1], cx = m[2][0], cy = m[2][1].
    static func intrinsics(fromColumnMajor m: simd_float3x3) -> CameraIntrinsics {
        CameraIntrinsics(fx: Double(m[0][0]),
                         fy: Double(m[1][1]),
                         cx: Double(m[2][0]),
                         cy: Double(m[2][1]))
    }

    /// Scale intrinsics from one pixel grid to another (e.g. camera.imageResolution -> stored size).
    /// Plain proportional scaling of fx, cx (by width ratio) and fy, cy (by height ratio).
    // VERIFY: ignores the half-pixel centre-convention offset; negligible for the ratios seen here (1.0).
    static func scaled(_ k: CameraIntrinsics, from: CGSize, to: PixelSize) -> CameraIntrinsics {
        guard from.width > 0, from.height > 0 else { return k }
        let sx = Double(to.width) / Double(from.width)
        let sy = Double(to.height) / Double(from.height)
        if sx == 1.0 && sy == 1.0 { return k }
        return CameraIntrinsics(fx: k.fx * sx, fy: k.fy * sy, cx: k.cx * sx, cy: k.cy * sy)
    }

    /// Convenience: ARKit matrix + the resolution it refers to + the size actually stored -> K for the manifest.
    static func storedIntrinsics(columnMajor m: simd_float3x3,
                                 referenceResolution: CGSize,
                                 storedSize: PixelSize) -> CameraIntrinsics {
        scaled(intrinsics(fromColumnMajor: m), from: referenceResolution, to: storedSize)
    }
}
