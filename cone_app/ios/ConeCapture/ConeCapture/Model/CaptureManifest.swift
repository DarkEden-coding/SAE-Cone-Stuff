import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Codable model of capture.json, schema `cone-capture/1` (see cone_app/SPEC.md — binding).
struct CaptureManifest: Encodable {
    static let schemaID = "cone-capture/1"

    enum Kind: String, Codable {
        case photo
        case video
    }

    struct Device: Codable, Equatable {
        var model: String
        var system: String
        var appVersion: String

        enum CodingKeys: String, CodingKey {
            case model, system
            case appVersion = "app_version"
        }
    }

    struct Frame: Codable, Equatable {
        var index: Int
        var t: Double
        var K: [[Double]]
    }

    var kind: Kind
    var device: Device
    var createdAt: Date
    var imageSize: PixelSize
    var frames: [Frame]
    var name: String?
    var notes: String?
    // `distortion` is always encoded as JSON null (pinhole, ARKit images are undistorted).

    enum CodingKeys: String, CodingKey {
        case schema, kind, device
        case createdAt = "created_at"
        case imageSize = "image_size"
        case frames, distortion, name, notes
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(Self.schemaID, forKey: .schema)
        try c.encode(kind, forKey: .kind)
        try c.encode(device, forKey: .device)
        try c.encode(Self.iso8601String(createdAt), forKey: .createdAt)
        try c.encode(imageSize.asArray, forKey: .imageSize)
        try c.encode(frames, forKey: .frames)
        try c.encodeNil(forKey: .distortion)
        if let name, !name.isEmpty { try c.encode(name, forKey: .name) }
        if let notes, !notes.isEmpty { try c.encode(notes, forKey: .notes) }
    }

    func jsonData() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try enc.encode(self)
    }

    /// "2026-10-15T12:00:00Z" (UTC, whole seconds).
    static func iso8601String(_ date: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        f.timeZone = TimeZone(identifier: "UTC")
        return f.string(from: date)
    }
}

extension CaptureManifest.Device {
    /// Hardware identifier (e.g. "iPhone15,2"), OS string and app version.
    static var current: CaptureManifest.Device {
        var info = utsname()
        uname(&info)
        let model = withUnsafePointer(to: &info.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: 1) { String(cString: $0) }
        }
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0"
        return .init(model: model, system: Self.systemString, appVersion: version)
    }

    private static var systemString: String {
        #if canImport(UIKit)
        return "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)"
        #else
        return "unknown"
        #endif
    }
}
