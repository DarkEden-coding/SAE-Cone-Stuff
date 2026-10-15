import Foundation

/// GET /api/health response (fields beyond `ok` are optional so older/newer servers still decode).
struct HealthInfo: Decodable {
    var ok: Bool
    var model: String?
    var imgsz: Int?
    var device: String?
    var version: String?
    var auth: Bool?
}

enum ServerAPI {
    enum APIError: LocalizedError {
        case badURL
        case http(Int, String)
        var errorDescription: String? {
            switch self {
            case .badURL: return "Invalid server URL."
            case .http(let code, let body): return "HTTP \(code)\(body.isEmpty ? "" : ": " + body)"
            }
        }
    }

    static func health(url: URL?, token: String?) async throws -> HealthInfo {
        guard let url else { throw APIError.badURL }
        var req = URLRequest(url: url, timeoutInterval: 10)
        // /api/health is unauthenticated per SPEC, but sending the header is harmless.
        if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        let (data, resp) = try await URLSession.shared.data(for: req)
        if let http = resp as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw APIError.http(http.statusCode, errorDetail(from: data))
        }
        return try JSONDecoder().decode(HealthInfo.self, from: data)
    }

    /// Extracts FastAPI's `{"detail": ...}` or falls back to the (truncated) raw body.
    static func errorDetail(from data: Data) -> String {
        if let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any], let d = obj["detail"] {
            if let s = d as? String { return s }
            if let json = try? JSONSerialization.data(withJSONObject: d), let s = String(data: json, encoding: .utf8) {
                return String(s.prefix(300))
            }
        }
        return String((String(data: data, encoding: .utf8) ?? "").prefix(300))
    }
}

/// Builds a multipart/form-data body as a file on disk (videos can be GBs; never held in memory).
enum MultipartBuilder {
    static func build(boundary: String,
                      captureJSON: Data,
                      mediaURL: URL,
                      mediaFileName: String,
                      mimeType: String,
                      to dest: URL) throws {
        let fm = FileManager.default
        try? fm.removeItem(at: dest)
        try fm.createDirectory(at: dest.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard fm.createFile(atPath: dest.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        let out = try FileHandle(forWritingTo: dest)
        defer { try? out.close() }

        func text(_ s: String) throws { try out.write(contentsOf: Data(s.utf8)) }

        try text("--\(boundary)\r\nContent-Disposition: form-data; name=\"capture\"\r\nContent-Type: application/json\r\n\r\n")
        try out.write(contentsOf: captureJSON)
        try text("\r\n--\(boundary)\r\nContent-Disposition: form-data; name=\"media\"; filename=\"\(mediaFileName)\"\r\nContent-Type: \(mimeType)\r\n\r\n")

        let input = try FileHandle(forReadingFrom: mediaURL)
        defer { try? input.close() }
        while true {
            let chunk: Data? = try autoreleasepool { try input.read(upToCount: 4 * 1024 * 1024) }
            guard let chunk, !chunk.isEmpty else { break }
            try out.write(contentsOf: chunk)
        }
        try text("\r\n--\(boundary)--\r\n")
    }
}
