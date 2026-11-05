import XCTest
@testable import ConeCapture

final class MediaRotationTests: XCTestCase {
    func testGravityChoosesUprightQuarterTurnsWithoutInterfaceOrientation() {
        XCTAssertEqual(MediaRotation.fromGravity(x: -1, y: 0, fallback: .clockwise90), .none)
        XCTAssertEqual(MediaRotation.fromGravity(x: 0, y: -1, fallback: .none), .clockwise90)
        XCTAssertEqual(MediaRotation.fromGravity(x: 1, y: 0, fallback: .none), .clockwise180)
        XCTAssertEqual(MediaRotation.fromGravity(x: 0, y: 1, fallback: .none), .clockwise270)
        XCTAssertEqual(MediaRotation.fromGravity(x: -0.8, y: -0.6, fallback: .clockwise90), .none)
        XCTAssertEqual(MediaRotation.fromGravity(x: -0.6, y: -0.8, fallback: .none), .clockwise90)
    }

    func testLookingStraightDownKeepsLastUsefulRotation() {
        for rotation in MediaRotation.allCases {
            XCTAssertEqual(MediaRotation.fromGravity(x: 0.01, y: -0.01, fallback: rotation), rotation)
        }
    }

    func testIntrinsicsAndImageDimensionsAfterEveryQuarterTurn() {
        let size = PixelSize(width: 160, height: 96)
        let native = CameraIntrinsics(fx: 120, fy: 140, cx: 51, cy: 37)
        let expected: [(MediaRotation, PixelSize, CameraIntrinsics)] = [
            (.none, size, native),
            (.clockwise90, .init(width: 96, height: 160), .init(fx: 140, fy: 120, cx: 58, cy: 51)),
            (.clockwise180, size, .init(fx: 120, fy: 140, cx: 108, cy: 58)),
            (.clockwise270, .init(width: 96, height: 160), .init(fx: 140, fy: 120, cx: 37, cy: 108)),
        ]
        for (rotation, outputSize, k) in expected {
            XCTAssertEqual(rotation.outputSize(for: size), outputSize)
            XCTAssertEqual(rotation.intrinsics(native, sourceSize: size), k)
        }
    }
}
