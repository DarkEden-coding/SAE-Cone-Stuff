import XCTest
import simd
@testable import ConeCapture

final class IntrinsicsConverterTests: XCTestCase {
    /// ARKit layout: columns are (fx,0,0), (0,fy,0), (cx,cy,1).
    private func arkitMatrix(fx: Float, fy: Float, cx: Float, cy: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3<Float>(fx, 0, 0), SIMD3<Float>(0, fy, 0), SIMD3<Float>(cx, cy, 1)))
    }

    func testColumnMajorToIntrinsics() {
        let k = IntrinsicsConverter.intrinsics(fromColumnMajor: arkitMatrix(fx: 3000, fy: 3001, cx: 1920, cy: 1080))
        XCTAssertEqual(k, CameraIntrinsics(fx: 3000, fy: 3001, cx: 1920, cy: 1080))
    }

    func testRowMajorKLayout() {
        let k = CameraIntrinsics(fx: 1, fy: 2, cx: 3, cy: 4)
        XCTAssertEqual(k.rowMajorK, [[1, 0, 3], [0, 2, 4], [0, 0, 1]])
    }

    func testNoScalingWhenSizesMatch() {
        let k = CameraIntrinsics(fx: 3000, fy: 3000, cx: 1920, cy: 1080)
        let out = IntrinsicsConverter.scaled(k, from: CGSize(width: 3840, height: 2160),
                                             to: PixelSize(width: 3840, height: 2160))
        XCTAssertEqual(out, k)
    }

    func testScalingToDifferentStoredSize() {
        let k = CameraIntrinsics(fx: 3000, fy: 3000, cx: 1920, cy: 1080)
        let out = IntrinsicsConverter.scaled(k, from: CGSize(width: 1920, height: 1080),
                                             to: PixelSize(width: 3840, height: 2160))
        XCTAssertEqual(out, CameraIntrinsics(fx: 6000, fy: 6000, cx: 3840, cy: 2160))
    }

    func testAnisotropicScaling() {
        let k = CameraIntrinsics(fx: 1000, fy: 1000, cx: 500, cy: 400)
        let out = IntrinsicsConverter.scaled(k, from: CGSize(width: 1000, height: 800),
                                             to: PixelSize(width: 2000, height: 800))
        XCTAssertEqual(out, CameraIntrinsics(fx: 2000, fy: 1000, cx: 1000, cy: 400))
    }

    func testStoredIntrinsicsEndToEnd() {
        let m = arkitMatrix(fx: 1500, fy: 1500, cx: 960, cy: 540)
        let out = IntrinsicsConverter.storedIntrinsics(columnMajor: m,
                                                       referenceResolution: CGSize(width: 1920, height: 1080),
                                                       storedSize: PixelSize(width: 3840, height: 2160))
        XCTAssertEqual(out.fx, 3000, accuracy: 1e-9)
        XCTAssertEqual(out.cx, 1920, accuracy: 1e-9)
        XCTAssertEqual(out.cy, 1080, accuracy: 1e-9)
    }
}
