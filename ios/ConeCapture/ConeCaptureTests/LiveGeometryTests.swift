import XCTest
import simd
@testable import ConeCapture

final class LiveGeometryTests: XCTestCase {
    /// The rotation MediaRotation picks for a held phone must put world up at image up (-y).
    func testUpIsImageUpForEveryHold() {
        let holds: [SIMD3<Double>] = [[0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0]]
        for g in holds {
            let rotation = MediaRotation.fromGravity(x: g.x, y: g.y, fallback: .none)
            let up = LiveGeometry.upInCamera(gravity: g, rotation: rotation)!
            XCTAssertLessThan(simd_length(up - SIMD3(0, -1, 0)), 1e-12, "gravity \(g) -> \(rotation)")
        }
    }

    func testLookingDownTiltsUpTowardsCamera() {
        let a = 0.4   // portrait, rear camera pitched down by a
        let up = LiveGeometry.upInCamera(gravity: SIMD3(0, -cos(a), -sin(a)), rotation: .clockwise90)!
        XCTAssertEqual(up.z, -sin(a), accuracy: 1e-12)    // optical axis points below the horizon
        XCTAssertEqual(up.y, -cos(a), accuracy: 1e-12)
    }

    /// Projecting with the rotated K, then mapping back, must land where the native K projects.
    func testRotatedPixelsMapBackToNative() {
        let size = PixelSize(width: 3840, height: 2160)
        let k = CameraIntrinsics(fx: 2900, fy: 2900, cx: 1915, cy: 1085)
        for rotation in MediaRotation.allCases {
            let imageSize = rotation.outputSize(for: size).width > rotation.outputSize(for: size).height
                ? PixelSize(width: 1280, height: 720) : PixelSize(width: 720, height: 1280)
            let rk = LiveGeometry.intrinsics(native: k, bufferSize: size, rotation: rotation, imageSize: imageSize)
            for p in [SIMD3<Double>(0.5, -0.2, 4), SIMD3(-1, 0.4, 9), SIMD3(0.1, 0.1, 2)] {
                let native = CGPoint(x: (k.fx * p.x / p.z + k.cx) / Double(size.width),
                                     y: (k.fy * p.y / p.z + k.cy) / Double(size.height))
                // Same axis permutation as upInCamera.
                let q: SIMD3<Double>
                switch rotation {
                case .none: q = p
                case .clockwise90: q = SIMD3(-p.y, p.x, p.z)
                case .clockwise180: q = SIMD3(-p.x, -p.y, p.z)
                case .clockwise270: q = SIMD3(p.y, -p.x, p.z)
                }
                let rotated = CGPoint(x: (rk.fx * q.x / q.z + rk.cx) / Double(imageSize.width),
                                      y: (rk.fy * q.y / q.z + rk.cy) / Double(imageSize.height))
                let back = LiveGeometry.nativeNormalized(rotated, rotation: rotation)
                XCTAssertEqual(back.x, native.x, accuracy: 1e-3, "\(rotation)")
                XCTAssertEqual(back.y, native.y, accuracy: 1e-3, "\(rotation)")
            }
        }
    }

    func testGravityHistoryInterpolates() {
        let h = GravityHistory()
        h.append(t: 1.00, g: SIMD3(0, -1, 0))
        h.append(t: 1.01, g: SIMD3(0, -1, -0.2))
        XCTAssertEqual(h.gravity(at: 1.005)!.z, -0.1, accuracy: 1e-9)
        XCTAssertEqual(h.gravity(at: 1.05)!.z, -0.2, accuracy: 1e-9)
        XCTAssertNil(h.gravity(at: 2))
    }
}

final class LiveWireFormatTests: XCTestCase {
    func testFrameWritesSpecKeysAndExplicitNulls() throws {
        let frame = LiveFrame(i: 3, t: 0.5, status: "waiting", T: nil, K: [[1, 0, 2], [0, 1, 3], [0, 0, 1]],
                              imageSize: [1280, 720],
                              detections: [LiveDetection(classID: 4, conf: 0.9, box: [1, 2, 3, 4],
                                                         keypoints: [[1, 2, 0.9]], pCam: nil, landmarkID: nil)],
                              nInliers: 0, rec: nil, stats: LiveStats(detMs: 10, slamMs: 1, fps: 12))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: LiveCoding.encode(frame)) as? [String: Any])
        XCTAssertEqual(Set(json.keys), ["type", "i", "t", "status", "T_world_cam", "K", "image_size", "detections",
                                        "n_inliers", "rec", "stats"])
        XCTAssertTrue(json["T_world_cam"] is NSNull)
        XCTAssertTrue(json["rec"] is NSNull)
        let det = try XCTUnwrap((json["detections"] as? [[String: Any]])?.first)
        XCTAssertEqual(Set(det.keys), ["class_id", "conf", "box_xyxy", "keypoints", "p_cam", "landmark_id"])
        XCTAssertTrue(det["p_cam"] is NSNull)
        XCTAssertEqual((json["stats"] as? [String: Double])?["det_ms"], 10)
    }
}
