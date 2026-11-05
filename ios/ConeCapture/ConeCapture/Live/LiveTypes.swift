import Foundation

// Wire types of the `cone-live/1` stream (cone_app/SPEC.md, "Live mode"). Every message is one JSON
// object; the same bytes go to the WebSocket and to the on-phone `live.jsonl` log.

typealias Matrix = [[Double]]

struct LiveDevice: Codable, Equatable {
    var model: String
    var system: String
    var appVersion: String

    enum CodingKeys: String, CodingKey { case model, system, appVersion = "app_version" }
}

struct LiveHello: Codable, Equatable {
    var type = "hello"
    var schema = "cone-live/1"
    var sessionID: String
    var t: Double
    var startedAt: String
    var name: String?
    var device: LiveDevice
    var imageSize: [Int]
    var K: Matrix
    var model: [String: LiveJSON]
    var options: [String: LiveJSON]

    enum CodingKeys: String, CodingKey {
        case type, schema, t, name, device, model, options, K
        case sessionID = "session_id", startedAt = "started_at", imageSize = "image_size"
    }
}

struct LiveDetection: Codable, Equatable {
    var classID: Int
    var conf: Double
    var box: [Double]               // x1, y1, x2, y2 in inference-image pixels
    var keypoints: [[Double]]       // 8 x (x, y, confidence), corrected
    var pCam: [Double]?
    var landmarkID: Int?

    enum CodingKeys: String, CodingKey {
        case conf, keypoints
        case classID = "class_id", box = "box_xyxy", pCam = "p_cam", landmarkID = "landmark_id"
    }

    /// Writes explicit nulls (SPEC: `p_cam`/`landmark_id` are `| null`), unlike synthesized Encodable.
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(classID, forKey: .classID)
        try c.encode(conf, forKey: .conf)
        try c.encode(box, forKey: .box)
        try c.encode(keypoints, forKey: .keypoints)
        try c.encode(pCam, forKey: .pCam)
        try c.encode(landmarkID, forKey: .landmarkID)
    }
}

struct LiveRecordingRef: Codable, Equatable {
    var recordingID: String
    var t: Double

    enum CodingKeys: String, CodingKey { case t, recordingID = "recording_id" }
}

struct LiveStats: Codable, Equatable {
    var detMs: Double
    var slamMs: Double
    var fps: Double

    enum CodingKeys: String, CodingKey { case fps, detMs = "det_ms", slamMs = "slam_ms" }
}

struct LiveFrame: Codable, Equatable {
    var type = "frame"
    var i: Int
    var t: Double
    var status: String              // waiting | init | tracked | lost
    var T: Matrix?
    var K: Matrix
    var imageSize: [Int]
    var detections: [LiveDetection]
    var nInliers: Int
    var rec: LiveRecordingRef?
    var stats: LiveStats

    enum CodingKeys: String, CodingKey {
        case type, i, t, status, K, detections, rec, stats
        case T = "T_world_cam", imageSize = "image_size", nInliers = "n_inliers"
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(type, forKey: .type)
        try c.encode(i, forKey: .i)
        try c.encode(t, forKey: .t)
        try c.encode(status, forKey: .status)
        try c.encode(T, forKey: .T)                 // null while waiting / lost
        try c.encode(K, forKey: .K)
        try c.encode(imageSize, forKey: .imageSize)
        try c.encode(detections, forKey: .detections)
        try c.encode(nInliers, forKey: .nInliers)
        try c.encode(rec, forKey: .rec)
        try c.encode(stats, forKey: .stats)
    }
}

struct LiveLandmark: Codable, Equatable {
    var id: Int
    var classID: Int
    var p: [Double]
    var sigma: [Double]
    var nObs: Int
    var confirmed: Bool

    enum CodingKeys: String, CodingKey {
        case id, confirmed
        case classID = "class_id", p = "p_world", sigma = "sigma_m", nObs = "n_obs"
    }
}

struct LiveKeyframe: Codable, Equatable {
    var i: Int
    var t: Double
    var T: Matrix

    enum CodingKeys: String, CodingKey { case i, t, T = "T_world_cam" }
}

struct LiveMap: Codable, Equatable {
    var type = "map"
    var t: Double
    var landmarks: [LiveLandmark]
    var keyframes: [LiveKeyframe]
}

struct LiveRecordingEvent: Codable, Equatable {
    var type = "recording"
    var t: Double
    var event: String               // start | stop
    var recordingID: String
    var name: String?

    enum CodingKeys: String, CodingKey { case type, t, event, name, recordingID = "recording_id" }
}

struct LiveBye: Codable, Equatable {
    var type = "bye"
    var t: Double
}

/// Small JSON value for free-form `model` / `options` objects.
enum LiveJSON: Codable, Equatable {
    case bool(Bool), number(Double), string(String), array([LiveJSON])

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let d = try? c.decode(Double.self) { self = .number(d) }
        else if let a = try? c.decode([LiveJSON].self) { self = .array(a) }
        else { self = .string(try c.decode(String.self)) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .bool(let b): try c.encode(b)
        case .number(let d): try c.encode(d)
        case .string(let s): try c.encode(s)
        case .array(let a): try c.encode(a)
        }
    }
}

enum LiveCoding {
    /// Compact, single-line JSON (no pretty printing: one message per JSONL line).
    static func encode<T: Encodable>(_ value: T) -> Data {
        let encoder = JSONEncoder()
        encoder.nonConformingFloatEncodingStrategy = .convertToString(positiveInfinity: "inf",
                                                                      negativeInfinity: "-inf", nan: "nan")
        return (try? encoder.encode(value)) ?? Data("{}".utf8)
    }
}

/// What the on-screen minimap draws, in world metres (z-up; x forward at init, y left).
struct MinimapSnapshot: Equatable {
    struct Cone: Equatable { var id: Int; var classID: Int; var x: Double; var y: Double; var confirmed: Bool }
    struct Camera: Equatable { var x: Double; var y: Double; var yaw: Double }   // yaw of the optical axis, rad
    var cones: [Cone] = []
    var camera: Camera?
    /// No pose this frame: `camera` is the last known pose, drawn grey.
    var cameraStale = false
    var trail: [SIMD2<Double>] = []
    var status: String = "waiting"
}
