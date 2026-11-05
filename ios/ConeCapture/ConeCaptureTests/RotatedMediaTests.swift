import XCTest
import AVFoundation
import CoreImage
import ImageIO
import simd
@testable import ConeCapture

/// Exercise the actual JPEG and H.264 encoders; no camera permission or separate UI runner needed.
final class RotatedMediaTests: XCTestCase {
    private let sourceSize = PixelSize(width: 160, height: 96)
    // Deliberately a different reference resolution: scale first, then rotate.
    private let referenceSize = CGSize(width: 80, height: 48)
    private var matrix: simd_float3x3 {
        .init(columns: (.init(60, 0, 0), .init(0, 70, 0), .init(25.5, 18.5, 1)))
    }

    private func expectedK(_ rotation: MediaRotation) -> [[Double]] {
        switch rotation {
        case .none: return [[120, 0, 51], [0, 140, 37], [0, 0, 1]]
        case .clockwise90: return [[140, 0, 58], [0, 120, 51], [0, 0, 1]]
        case .clockwise180: return [[120, 0, 108], [0, 140, 58], [0, 0, 1]]
        case .clockwise270: return [[140, 0, 37], [0, 120, 108], [0, 0, 1]]
        }
    }

    private func buffer(width: Int, height: Int) throws -> CVPixelBuffer {
        var pb: CVPixelBuffer?
        let result = CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
                                        [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
        XCTAssertEqual(result, kCVReturnSuccess)
        return try XCTUnwrap(pb)
    }

    private func testPattern() throws -> CVPixelBuffer {
        let pb = try buffer(width: sourceSize.width, height: sourceSize.height)
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(pb)).assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        let colors: [(UInt8, UInt8, UInt8)] = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for y in 0..<sourceSize.height {
            for x in 0..<sourceSize.width {
                let quadrant = (y < sourceSize.height / 2 ? 0 : 2) + (x < sourceSize.width / 2 ? 0 : 1)
                let (r, g, b) = colors[quadrant]
                let offset = y * stride + x * 4
                base[offset] = b
                base[offset + 1] = g
                base[offset + 2] = r
                base[offset + 3] = 255
            }
        }
        return pb
    }

    private func assertPixels(_ pb: CVPixelBuffer, rotation: MediaRotation,
                              file: StaticString = #filePath, line: UInt = #line) throws {
        let expectedIndices: [Int]
        switch rotation {
        case .none: expectedIndices = [0, 1, 2, 3]
        case .clockwise90: expectedIndices = [2, 0, 3, 1]
        case .clockwise180: expectedIndices = [3, 2, 1, 0]
        case .clockwise270: expectedIndices = [1, 3, 0, 2]
        }
        let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]
        let w = CVPixelBufferGetWidth(pb)
        let h = CVPixelBufferGetHeight(pb)
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        let base = try XCTUnwrap(CVPixelBufferGetBaseAddress(pb)).assumingMemoryBound(to: UInt8.self)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        let points = [(w / 4, h / 4), (3 * w / 4, h / 4), (w / 4, 3 * h / 4), (3 * w / 4, 3 * h / 4)]
        for (index, point) in points.enumerated() {
            let offset = point.1 * stride + point.0 * 4
            let rgb = [Int(base[offset + 2]), Int(base[offset + 1]), Int(base[offset])]
            for channel in 0..<3 {
                XCTAssertEqual(Double(rgb[channel]), Double(colors[expectedIndices[index]][channel]), accuracy: 40,
                               "Wrong pixel rotation at \(rotation.rawValue)°, quadrant \(index)", file: file, line: line)
            }
        }
    }

    func testJPEGHasRotatedPixelsAndMatchingIntrinsics() throws {
        let pattern = try testPattern()
        for rotation in MediaRotation.allCases {
            let photo = try PhotoCapture.encode(pixelBuffer: pattern, intrinsics: matrix,
                                                referenceResolution: referenceSize, rotation: rotation)
            let source = try XCTUnwrap(CGImageSourceCreateWithData(photo.jpegData as CFData, nil))
            let properties = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [String: Any])
            // Decode raw JPEG pixels, without applying any EXIF orientation tag.
            let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
            XCTAssertEqual(image.width, photo.imageSize.width)
            XCTAssertEqual(image.height, photo.imageSize.height)
            XCTAssertEqual(photo.imageSize, rotation.outputSize(for: sourceSize))
            XCTAssertEqual(photo.intrinsics.rowMajorK, expectedK(rotation))
            XCTAssertEqual((properties[kCGImagePropertyOrientation as String] as? Int) ?? 1, 1)
            let decoded = try buffer(width: image.width, height: image.height)
            CIContext().render(CIImage(cgImage: image), to: decoded)
            try assertPixels(decoded, rotation: rotation)
        }
    }

    func testH264HasRotatedPixelsMatchingFramesAndNoDisplayTransform() async throws {
        let pattern = try testPattern()
        for rotation in MediaRotation.allCases {
            let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mp4")
            defer { try? FileManager.default.removeItem(at: url) }
            let recorder = VideoRecorder(outputURL: url, rotation: rotation)
            for index in 0..<4 {
                recorder.append(pixelBuffer: pattern, timestamp: 10 + Double(index) / 30,
                                intrinsics: matrix, referenceResolution: referenceSize)
                // Feed at camera speed so this tests the encoder, rather than deliberately saturating it.
                try await Task.sleep(nanoseconds: 40_000_000)
            }
            let output = try await withCheckedThrowingContinuation { continuation in
                recorder.finish { continuation.resume(with: $0) }
            }
            XCTAssertFalse(output.frames.isEmpty)
            XCTAssertEqual(output.imageSize, rotation.outputSize(for: sourceSize))
            let asset = AVURLAsset(url: url)
            let tracks = try await asset.loadTracks(withMediaType: .video)
            let track = try XCTUnwrap(tracks.first)
            let size = try await track.load(.naturalSize)
            let transform = try await track.load(.preferredTransform)
            XCTAssertEqual(Int(size.width), output.imageSize.width)
            XCTAssertEqual(Int(size.height), output.imageSize.height)
            XCTAssertEqual(transform, .identity)
            let reader = try AVAssetReader(asset: asset)
            let readerOutput = AVAssetReaderTrackOutput(track: track, outputSettings: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            ])
            reader.add(readerOutput)
            XCTAssertTrue(reader.startReading())
            var decodedCount = 0
            while let sample = readerOutput.copyNextSampleBuffer() {
                let pb = try XCTUnwrap(CMSampleBufferGetImageBuffer(sample))
                try assertPixels(pb, rotation: rotation)
                XCTAssertLessThan(decodedCount, output.frames.count)
                guard decodedCount < output.frames.count else { break }
                let frame = output.frames[decodedCount]
                XCTAssertEqual(frame.index, decodedCount)
                XCTAssertEqual(frame.K, expectedK(rotation))
                XCTAssertEqual(frame.t, CMSampleBufferGetPresentationTimeStamp(sample).seconds, accuracy: 0.00001)
                decodedCount += 1
            }
            XCTAssertEqual(reader.status, .completed)
            XCTAssertEqual(decodedCount, output.frames.count)
        }
    }

    func test4KVideoCanBeEncodedAsPortraitPixels() async throws {
        let pb = try buffer(width: 3840, height: 2160)
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 3840, height: 2160))
        CIContext().render(image, to: pb)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".mp4")
        defer { try? FileManager.default.removeItem(at: url) }
        let recorder = VideoRecorder(outputURL: url, rotation: .clockwise90)
        for index in 0..<4 {
            recorder.append(pixelBuffer: pb, timestamp: Double(index) / 30, intrinsics: matrix,
                            referenceResolution: referenceSize)
            try await Task.sleep(nanoseconds: 40_000_000)
        }
        let output = try await withCheckedThrowingContinuation { continuation in
            recorder.finish { continuation.resume(with: $0) }
        }
        XCTAssertFalse(output.frames.isEmpty)
        XCTAssertEqual(output.imageSize, .init(width: 2160, height: 3840))
        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let size = try await track.load(.naturalSize)
        XCTAssertEqual(size, CGSize(width: 2160, height: 3840))
    }
}
