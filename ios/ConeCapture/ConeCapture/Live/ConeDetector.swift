import Foundation
import CoreML
import CoreImage
import CoreVideo
import ConeSLAM

/// YOLO26n-pose on CoreML (`Resources/ConePose{W}x{H}.mlpackage`, see scripts/export_model.sh).
///
/// The camera buffer is rotated to the gravity-upright quarter turn, scaled to the inference image
/// (1280x720 or 720x1280) and letterboxed (grey 114, centred) to the model input (1280x736 / 736x1280).
/// Detections come back in inference-image pixels. Not thread-safe: use from one queue.
final class ConeDetector {
    enum DetectorError: LocalizedError {
        case missingModel(String), noOutput, buffer
        var errorDescription: String? {
            switch self {
            case .missingModel(let n): return "Cone model \(n) is not in the app bundle (run scripts/export_model.sh)."
            case .noOutput: return "Cone model returned no output."
            case .buffer: return "Cannot allocate the model input buffer."
            }
        }
    }

    let imageSize: PixelSize              // inference image, before letterboxing
    let inputSize: PixelSize              // model input
    private let model: MLModel
    private let inputName: String
    private let outputName: String
    private let padding: Vec2
    private var pool: CVPixelBufferPool?
    private let context = CIContext(options: [.cacheIntermediates: false, .workingColorSpace: NSNull()])
    private let background: CIImage

    init(portrait: Bool) throws {
        inputSize = portrait ? PixelSize(width: 736, height: 1280) : PixelSize(width: 1280, height: 736)
        imageSize = portrait ? PixelSize(width: 720, height: 1280) : PixelSize(width: 1280, height: 720)
        padding = Vec2(Double(inputSize.width - imageSize.width) / 2, Double(inputSize.height - imageSize.height) / 2)
        let name = "ConePose\(inputSize.width)x\(inputSize.height)"
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else {
            throw DetectorError.missingModel(name)
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try MLModel(contentsOf: url, configuration: configuration)
        inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        outputName = model.modelDescription.outputDescriptionsByName.keys.first ?? ""
        let grey = CGFloat(114) / 255
        background = CIImage(color: CIColor(red: grey, green: grey, blue: grey))
            .cropped(to: CGRect(x: 0, y: 0, width: inputSize.width, height: inputSize.height))
        let attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: inputSize.width,
            kCVPixelBufferHeightKey as String: inputSize.height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:],
        ]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
    }

    /// Runs the model on a native camera buffer rotated by `rotation`.
    func detect(_ pixelBuffer: CVPixelBuffer, rotation: MediaRotation, conf: Double) throws -> [Detection] {
        guard let pool else { throw DetectorError.buffer }
        var input: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &input)
        guard let input else { throw DetectorError.buffer }

        var image = CIImage(cvPixelBuffer: pixelBuffer).oriented(rotation.imageOrientation)
        image = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        let scale = CGFloat(imageSize.width) / image.extent.width
        image = image.transformed(by: CGAffineTransform(scaleX: scale, y: CGFloat(imageSize.height) / image.extent.height))
            .transformed(by: CGAffineTransform(translationX: padding.x, y: padding.y))
        context.render(image.composited(over: background), to: input)

        let features = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: input)])
        let output = try model.prediction(from: features)
        guard let array = output.featureValue(for: outputName)?.multiArrayValue, array.shape.count == 3 else {
            throw DetectorError.noOutput
        }
        let anchors = array.shape[2].intValue
        let channelStride = array.strides[1].intValue, anchorStride = array.strides[2].intValue
        func run<T: MLShapedArrayScalar>(_ type: T.Type, _ convert: @escaping (T) -> Float) -> [Detection] {
            array.withUnsafeBufferPointer(ofType: T.self) { buffer in
                PoseDecoder.decode(anchors: anchors, conf: conf, offset: padding) { c, a in
                    convert(buffer[c * channelStride + a * anchorStride])
                }
            }
        }
        switch array.dataType {
        case .float16: return run(Float16.self) { Float($0) }
        case .float32: return run(Float.self) { $0 }
        default: return run(Double.self) { Float($0) }
        }
    }
}
