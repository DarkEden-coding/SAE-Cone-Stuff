import XCTest
import UIKit
import CoreVideo
@testable import ConeCapture

/// Runs the bundled CoreML model (CPU in the simulator) through the app's own preprocessing on a real
/// cone image. Reference values: ultralytics NMS on the same image letterboxed by OpenCV
/// (ConeSLAM/Tests/ConeSLAMTests/Fixtures/decode_expected.json).
final class ConeDetectorTests: XCTestCase {
    func testDetectsTheYellowConeLikeUltralytics() throws {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: "cone_crop_720x1280", withExtension: "jpg"))
        let image = try XCTUnwrap(UIImage(contentsOfFile: url.path)?.cgImage)
        let buffer = try XCTUnwrap(Self.pixelBuffer(image))
        let detector = try ConeDetector(portrait: true)
        XCTAssertEqual(detector.imageSize, PixelSize(width: 720, height: 1280))
        let detections = try detector.detect(buffer, rotation: .none, conf: 0.2)
        let yellow = try XCTUnwrap(detections.first { $0.classID == 4 }, "\(detections)")
        XCTAssertEqual(yellow.conf, 0.735, accuracy: 0.1)
        let expected = [275.5, 441.75, 323.0, 506.25]
        for i in 0..<4 { XCTAssertEqual(yellow.box[i], expected[i], accuracy: 3) }
        XCTAssertEqual(yellow.keypoints[0].x, 292.0, accuracy: 3)
        XCTAssertEqual(yellow.keypoints[0].y, 462.0, accuracy: 3)
    }

    private static func pixelBuffer(_ image: CGImage) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        let attributes: [String: Any] = [kCVPixelBufferIOSurfacePropertiesKey as String: [:],
                                         kCVPixelBufferCGImageCompatibilityKey as String: true]
        CVPixelBufferCreate(kCFAllocatorDefault, image.width, image.height, kCVPixelFormatType_32BGRA,
                            attributes as CFDictionary, &buffer)
        guard let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: image.width, height: image.height,
                                      bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return buffer
    }
}
