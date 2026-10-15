import Foundation

enum UploadStatus: String, Codable {
    case pending, uploading, uploaded, failed
}

/// Local metadata for one capture folder (`meta.json`; separate from the spec'd capture.json).
struct Recording: Codable, Identifiable, Equatable {
    var id: String                       // folder name (timestamp)
    var kind: CaptureManifest.Kind
    var createdAt: Date
    var name: String?
    var mediaFileName: String
    var mediaBytes: Int64
    var status: UploadStatus
    var jobID: String?
    var errorMessage: String?
    var attempts: Int
    var mediaDeleted: Bool
}

/// Documents/Recordings/<timestamp>/{video.mp4|photo.jpg, capture.json, meta.json}
@MainActor
final class RecordingStore: ObservableObject {
    @Published private(set) var recordings: [Recording] = []
    /// Upload progress 0...1 keyed by recording id (transient).
    @Published var progress: [String: Double] = [:]

    let rootURL: URL
    private let fm = FileManager.default

    private static let folderFormatter: DateFormatter = {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f
    }()

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        rootURL = docs.appendingPathComponent("Recordings", isDirectory: true)
        try? fm.createDirectory(at: rootURL, withIntermediateDirectories: true)
        load()
    }

    var pendingCount: Int {
        recordings.filter { $0.status == .pending || $0.status == .failed || $0.status == .uploading }.count
    }

    // MARK: Paths

    func folderURL(_ id: String) -> URL { rootURL.appendingPathComponent(id, isDirectory: true) }
    func mediaURL(_ r: Recording) -> URL { folderURL(r.id).appendingPathComponent(r.mediaFileName) }
    func captureURL(_ r: Recording) -> URL { folderURL(r.id).appendingPathComponent("capture.json") }
    private func metaURL(_ id: String) -> URL { folderURL(id).appendingPathComponent("meta.json") }

    // MARK: CRUD

    func makeFolder() throws -> (id: String, url: URL) {
        let base = Self.folderFormatter.string(from: Date())
        var id = base
        var n = 2
        while fm.fileExists(atPath: folderURL(id).path) {
            id = "\(base)-\(n)"
            n += 1
        }
        let url = folderURL(id)
        try fm.createDirectory(at: url, withIntermediateDirectories: true)
        return (id, url)
    }

    func recording(_ id: String) -> Recording? { recordings.first { $0.id == id } }

    func add(_ r: Recording) {
        recordings.insert(r, at: 0)
        recordings.sort { $0.createdAt > $1.createdAt }
        save(r)
    }

    func update(_ id: String, _ mutate: (inout Recording) -> Void) {
        guard let i = recordings.firstIndex(where: { $0.id == id }) else { return }
        mutate(&recordings[i])
        save(recordings[i])
    }

    func delete(_ id: String) {
        try? fm.removeItem(at: folderURL(id))
        recordings.removeAll { $0.id == id }
        progress[id] = nil
    }

    /// Deletes only the (large) media file, keeping capture.json/meta so the job link survives.
    func deleteMedia(_ id: String) {
        guard let r = recording(id) else { return }
        try? fm.removeItem(at: mediaURL(r))
        update(id) { $0.mediaDeleted = true }
    }

    // MARK: Persistence

    private func save(_ r: Recording) {
        let enc = JSONEncoder()
        enc.dateEncodingStrategy = .iso8601
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? enc.encode(r) {
            try? data.write(to: metaURL(r.id), options: .atomic)
        }
    }

    private func load() {
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        let dirs = (try? fm.contentsOfDirectory(at: rootURL, includingPropertiesForKeys: nil)) ?? []
        var out: [Recording] = []
        for d in dirs {
            let meta = d.appendingPathComponent("meta.json")
            if let data = try? Data(contentsOf: meta), let r = try? dec.decode(Recording.self, from: data) {
                out.append(r)
            }
        }
        recordings = out.sorted { $0.createdAt > $1.createdAt }
    }
}
