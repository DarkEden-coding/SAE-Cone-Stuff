import Foundation
import ARKit
import AVFoundation

/// Helpers around `ARConfiguration.VideoFormat`.
enum VideoFormats {
    /// Back-camera formats, highest resolution first, then highest fps.
    static var supported: [ARConfiguration.VideoFormat] {
        ARWorldTrackingConfiguration.supportedVideoFormats
            .filter { $0.captureDevicePosition == .back }
            .sorted {
                let a = $0.imageResolution.width * $0.imageResolution.height
                let b = $1.imageResolution.width * $1.imageResolution.height
                return a != b ? a > b : $0.framesPerSecond > $1.framesPerSecond
            }
    }

    // VERIFY: `recommendedVideoFormatFor4KResolution` is a class property on ARConfiguration subclasses (iOS 16+).
    static var recommended4K: ARConfiguration.VideoFormat? {
        ARWorldTrackingConfiguration.recommendedVideoFormatFor4KResolution
    }

    // VERIFY: same for `recommendedVideoFormatForHighResolutionFrameCapturing` (iOS 16+).
    static var recommendedHighResStill: ARConfiguration.VideoFormat? {
        ARWorldTrackingConfiguration.recommendedVideoFormatForHighResolutionFrameCapturing
    }

    static func id(_ f: ARConfiguration.VideoFormat) -> String {
        "\(Int(f.imageResolution.width))x\(Int(f.imageResolution.height))@\(f.framesPerSecond)-\(f.captureDeviceType.rawValue)"
    }

    static func label(_ f: ARConfiguration.VideoFormat) -> String {
        let lens = f.captureDeviceType.rawValue.contains("UltraWide") ? "Ultra-wide" : "Wide"
        return "\(Int(f.imageResolution.width))\u{00D7}\(Int(f.imageResolution.height)) @ \(f.framesPerSecond) fps (\(lens))"
    }

    static func format(forID id: String) -> ARConfiguration.VideoFormat? {
        supported.first { Self.id($0) == id }
    }

    /// 4K recommended format if present, else the highest-resolution one.
    static var defaultFormat: ARConfiguration.VideoFormat? {
        recommended4K ?? supported.first
    }
}

@MainActor
final class AppSettings: ObservableObject {
    private enum Key {
        static let baseURL = "baseURL"
        static let formatID = "formatID"
        static let autofocus = "autofocus"
        static let photoMaxRes = "photoMaxResolution"
        static let captureName = "captureName"
        static let autoUpload = "autoUpload"
        static let deleteAfterUpload = "deleteAfterUpload"
    }
    private static let tokenAccount = "bearerToken"

    @Published var baseURL: String { didSet { UserDefaults.standard.set(baseURL, forKey: Key.baseURL) } }
    @Published var token: String { didSet { Keychain.set(token, account: Self.tokenAccount) } }
    @Published var formatID: String { didSet { UserDefaults.standard.set(formatID, forKey: Key.formatID) } }
    /// Default false: locked focus keeps intrinsics (fx, fy) stable between frames.
    @Published var autofocus: Bool { didSet { UserDefaults.standard.set(autofocus, forKey: Key.autofocus) } }
    /// Photo mode uses ARKit's recommended high-resolution-still format instead of the video format.
    @Published var photoMaxResolution: Bool { didSet { UserDefaults.standard.set(photoMaxResolution, forKey: Key.photoMaxRes) } }
    @Published var captureName: String { didSet { UserDefaults.standard.set(captureName, forKey: Key.captureName) } }
    @Published var autoUpload: Bool { didSet { UserDefaults.standard.set(autoUpload, forKey: Key.autoUpload) } }
    @Published var deleteAfterUpload: Bool { didSet { UserDefaults.standard.set(deleteAfterUpload, forKey: Key.deleteAfterUpload) } }

    init() {
        let d = UserDefaults.standard
        baseURL = d.string(forKey: Key.baseURL) ?? ""
        token = Keychain.get(Self.tokenAccount) ?? ""
        formatID = d.string(forKey: Key.formatID) ?? ""
        autofocus = d.object(forKey: Key.autofocus) as? Bool ?? false
        photoMaxResolution = d.object(forKey: Key.photoMaxRes) as? Bool ?? true
        captureName = d.string(forKey: Key.captureName) ?? ""
        autoUpload = d.object(forKey: Key.autoUpload) as? Bool ?? true
        deleteAfterUpload = d.object(forKey: Key.deleteAfterUpload) as? Bool ?? false

        if VideoFormats.format(forID: formatID) == nil, let def = VideoFormats.defaultFormat {
            formatID = VideoFormats.id(def)
        }
    }

    /// Changes in this string require restarting the ARSession.
    var sessionConfigKey: String { "\(formatID)|\(autofocus)|\(photoMaxResolution)" }

    var selectedFormat: ARConfiguration.VideoFormat? {
        VideoFormats.format(forID: formatID) ?? VideoFormats.defaultFormat
    }

    var trimmedName: String? {
        let t = captureName.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    var trimmedToken: String? {
        let t = token.trimmingCharacters(in: .whitespacesAndNewlines)
        return t.isEmpty ? nil : t
    }

    /// Normalised base URL string without trailing slash ("" if unset).
    var normalizedBase: String {
        var s = baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if s.isEmpty { return "" }
        if !s.contains("://") { s = "https://" + s }
        while s.hasSuffix("/") { s.removeLast() }
        return s
    }

    func endpoint(path: String) -> URL? {
        let b = normalizedBase
        guard !b.isEmpty else { return nil }
        return URL(string: b + path)
    }

    func jobPageURL(id: String) -> URL? {
        let b = normalizedBase
        guard !b.isEmpty else { return nil }
        return URL(string: b + "/#/job/" + id)
    }
}
