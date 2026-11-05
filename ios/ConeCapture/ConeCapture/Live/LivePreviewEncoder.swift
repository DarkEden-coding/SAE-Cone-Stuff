import Foundation
import CoreImage
import CoreVideo

/// Low-rate camera preview for the laptop viewer (SPEC "Camera preview images"): the gravity-upright
/// inference image, downscaled to `longSide` px, JPEG, at most `fps` per second. Encoding runs on its own
/// queue and drops frames while busy, so it never delays detection. Thread-safe.
final class LivePreviewEncoder: @unchecked Sendable {
    var onImage: ((Data) -> Void)?

    private let rotation: MediaRotation
    private let imageSize: PixelSize        // inference image the overlays refer to
    private let size: PixelSize             // JPEG size
    private let interval: Double
    private let quality: Double
    private let queue = DispatchQueue(label: "cone.live.preview", qos: .utility)
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)!

    private let lock = NSLock()
    private var busy = false
    private var lastT = -Double.infinity

    init(rotation: MediaRotation, imageSize: PixelSize, longSide: Int = 480, fps: Double = 15, quality: Double = 0.5) {
        self.rotation = rotation
        self.imageSize = imageSize
        let scale = Double(longSide) / Double(max(imageSize.width, imageSize.height))
        size = PixelSize(width: Int((Double(imageSize.width) * scale).rounded()),
                         height: Int((Double(imageSize.height) * scale).rounded()))
        interval = 1 / fps
        self.quality = quality
    }

    /// Called for every camera frame; `t` is the live session clock.
    func offer(_ buffer: CVPixelBuffer, t: Double) {
        lock.lock()
        // Small slack so a 30/60 fps camera yields a steady 15 fps.
        guard !busy, t - lastT >= interval * 0.9 else { lock.unlock(); return }
        busy = true
        lastT = t
        lock.unlock()
        queue.async { [self] in
            defer { lock.lock(); busy = false; lock.unlock() }
            autoreleasepool {
                guard let jpeg = encode(buffer) else { return }
                let header = LiveCoding.encode(LivePreviewHeader(t: t, size: size.asArray, imageSize: imageSize.asArray))
                var message = header
                message.append(0x0A)
                message.append(jpeg)
                onImage?(message)
            }
        }
    }

    private func encode(_ buffer: CVPixelBuffer) -> Data? {
        var image = CIImage(cvPixelBuffer: buffer).oriented(rotation.imageOrientation)
        image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        image = image.transformed(by: CGAffineTransform(scaleX: CGFloat(size.width) / image.extent.width,
                                                        y: CGFloat(size.height) / image.extent.height))
        let options = [CIImageRepresentationOption(rawValue: kCGImageDestinationLossyCompressionQuality as String): quality]
        return context.jpegRepresentation(of: image.cropped(to: CGRect(x: 0, y: 0, width: size.width, height: size.height)),
                                          colorSpace: colorSpace, options: options)
    }
}

struct LivePreviewHeader: Encodable {
    var type = "image"
    var t: Double
    var size: [Int]
    var imageSize: [Int]

    enum CodingKeys: String, CodingKey { case type, t, size, imageSize = "image_size" }
}
