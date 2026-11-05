import XCTest
@testable import ConeCapture

final class CaptureManifestTests: XCTestCase {
    private func makeManifest(kind: CaptureManifest.Kind = .video, name: String? = nil) -> CaptureManifest {
        let k = CameraIntrinsics(fx: 3000, fy: 3000, cx: 1920, cy: 1080).rowMajorK
        return CaptureManifest(
            kind: kind,
            device: .init(model: "iPhone15,2", system: "iOS 26.0", appVersion: "1.0"),
            createdAt: Date(timeIntervalSince1970: 1_791_000_000),   // fixed instant
            imageSize: PixelSize(width: 3840, height: 2160),
            frames: [.init(index: 0, t: 0.0, K: k), .init(index: 1, t: 0.0333, K: k)],
            name: name, notes: nil)
    }

    private func json(_ m: CaptureManifest) throws -> [String: Any] {
        let data = try m.jsonData()
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testTopLevelKeysExactlyMatchSpec() throws {
        let obj = try json(makeManifest())
        XCTAssertEqual(Set(obj.keys),
                       ["schema", "kind", "device", "created_at", "image_size", "frames", "distortion"])
        XCTAssertEqual(obj["schema"] as? String, "cone-capture/1")
        XCTAssertEqual(obj["kind"] as? String, "video")
        XCTAssertTrue(obj["distortion"] is NSNull, "distortion must be an explicit JSON null")
        XCTAssertEqual(obj["image_size"] as? [Int], [3840, 2160])
    }

    func testOptionalNameIncludedWhenSet() throws {
        let obj = try json(makeManifest(name: "cone run 1"))
        XCTAssertEqual(obj["name"] as? String, "cone run 1")
        XCTAssertNil(obj["notes"])
    }

    func testDeviceKeys() throws {
        let obj = try json(makeManifest())
        let dev = try XCTUnwrap(obj["device"] as? [String: Any])
        XCTAssertEqual(Set(dev.keys), ["model", "system", "app_version"])
    }

    func testFrameShape() throws {
        let obj = try json(makeManifest(kind: .photo))
        XCTAssertEqual(obj["kind"] as? String, "photo")
        let frames = try XCTUnwrap(obj["frames"] as? [[String: Any]])
        XCTAssertEqual(frames.count, 2)
        XCTAssertEqual(Set(frames[0].keys), ["index", "t", "K"])
        XCTAssertEqual(frames[1]["index"] as? Int, 1)
        let K = try XCTUnwrap(frames[0]["K"] as? [[Double]])
        XCTAssertEqual(K, [[3000, 0, 1920], [0, 3000, 1080], [0, 0, 1]])
    }

    func testCreatedAtFormat() throws {
        let s = CaptureManifest.iso8601String(Date(timeIntervalSince1970: 0))
        XCTAssertEqual(s, "1970-01-01T00:00:00Z")
        let obj = try json(makeManifest())
        let created = try XCTUnwrap(obj["created_at"] as? String)
        XCTAssertTrue(created.hasSuffix("Z"))
    }
}
