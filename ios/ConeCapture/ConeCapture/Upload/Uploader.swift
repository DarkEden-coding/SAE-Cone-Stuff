import Foundation
import UIKit

/// Background URLSession uploader for POST {base}/api/captures (multipart: `media` + `capture`).
///
/// - The multipart body is written to Caches/Uploads/<id>.body and sent with `uploadTask(with:fromFile:)`.
/// - Task identity survives app relaunch through `taskDescription` (= recording id).
/// - Delegate callbacks arrive on the session's private serial queue; UI/store changes hop to the main actor.
final class Uploader: NSObject, ObservableObject, URLSessionDataDelegate {
    static let sessionIdentifier = "com.example.ConeCapture.upload"
    private static let maxAutoRetries = 3

    private struct JobResponse: Decodable { var id: String }

    private let store: RecordingStore
    private let settings: AppSettings
    private var session: URLSession!
    private let bufferLock = NSLock()
    private var buffers: [Int: Data] = [:]

    @MainActor
    init(store: RecordingStore, settings: AppSettings) {
        self.store = store
        self.settings = settings
        super.init()
        let cfg = URLSessionConfiguration.background(withIdentifier: Self.sessionIdentifier)
        cfg.sessionSendsLaunchEvents = true
        cfg.isDiscretionary = false
        cfg.waitsForConnectivity = true
        cfg.timeoutIntervalForResource = 60 * 60 * 24 * 3
        session = URLSession(configuration: cfg, delegate: self, delegateQueue: nil)
    }

    static func bodyURL(id: String) -> URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Uploads", isDirectory: true)
            .appendingPathComponent("\(id).body")
    }

    // MARK: Public API (main actor)

    /// Marks any recording stuck in `uploading` without a live URLSession task as failed.
    @MainActor
    func reconcile() {
        // Delay so that completions delivered by the system right after relaunch are processed first.
        // VERIFY: ordering of relaunch delegate events vs. getAllTasks; the delay is a conservative workaround.
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 3_000_000_000)
            let live = await liveTaskIDs()
            for r in store.recordings where r.status == .uploading && !live.contains(r.id) {
                store.update(r.id) { $0.status = .failed; $0.errorMessage = "Upload interrupted." }
                store.progress[r.id] = nil
            }
        }
    }

    private func liveTaskIDs() async -> Set<String> {
        await withCheckedContinuation { (cont: CheckedContinuation<Set<String>, Never>) in
            session.getAllTasks { tasks in
                cont.resume(returning: Set(tasks.compactMap(\.taskDescription)))
            }
        }
    }

    /// Manual retry: resets the attempt counter.
    @MainActor
    func retry(_ id: String) {
        store.update(id) { $0.attempts = 0 }
        upload(id)
    }

    @MainActor
    func uploadAllPending() {
        for r in store.recordings where (r.status == .pending || r.status == .failed) && !r.mediaDeleted {
            upload(r.id)
        }
    }

    @MainActor
    func cancel(_ id: String) {
        session.getAllTasks { tasks in
            tasks.filter { $0.taskDescription == id }.forEach { $0.cancel() }
        }
    }

    @MainActor
    func upload(_ id: String) {
        guard let rec = store.recording(id), rec.status != .uploading, !rec.mediaDeleted else { return }
        guard let endpoint = settings.endpoint(path: "/api/captures") else {
            fail(id, "Set a valid server URL in Settings.", transient: false)
            return
        }
        let token = settings.trimmedToken
        let mediaURL = store.mediaURL(rec)
        let captureURL = store.captureURL(rec)
        let bodyURL = Self.bodyURL(id: id)
        let fileName = rec.mediaFileName
        let mime = rec.kind == .video ? "video/mp4" : "image/jpeg"
        let boundary = "ConeCapture-" + UUID().uuidString

        store.update(id) { $0.status = .uploading; $0.errorMessage = nil }
        store.progress[id] = 0

        Task.detached(priority: .utility) { [weak self] in
            do {
                let captureJSON = try Data(contentsOf: captureURL)
                try MultipartBuilder.build(boundary: boundary, captureJSON: captureJSON, mediaURL: mediaURL,
                                           mediaFileName: fileName, mimeType: mime, to: bodyURL)
                var req = URLRequest(url: endpoint)
                req.httpMethod = "POST"
                req.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
                if let token { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
                await self?.startTask(req, bodyURL: bodyURL, id: id)
            } catch {
                await self?.fail(id, "Could not prepare upload: \(error.localizedDescription)", transient: false)
            }
        }
    }

    @MainActor
    private func startTask(_ req: URLRequest, bodyURL: URL, id: String) {
        let task = session.uploadTask(with: req, fromFile: bodyURL)
        task.taskDescription = id
        task.resume()
    }

    // MARK: Completion handling (main actor)

    @MainActor
    private func complete(id: String, status: Int?, body: Data, error: Error?) {
        try? FileManager.default.removeItem(at: Self.bodyURL(id: id))
        store.progress[id] = nil

        if let error {
            let ns = error as NSError
            if ns.domain == NSURLErrorDomain && ns.code == NSURLErrorCancelled {
                if store.recording(id) != nil {
                    store.update(id) { $0.status = .failed; $0.errorMessage = "Cancelled." }
                }
            } else {
                fail(id, error.localizedDescription, transient: true)
            }
            return
        }
        guard let status else { fail(id, "No HTTP response.", transient: true); return }
        if (200..<300).contains(status), let job = try? JSONDecoder().decode(JobResponse.self, from: body) {
            store.update(id) {
                $0.status = .uploaded
                $0.jobID = job.id
                $0.errorMessage = nil
            }
            if settings.deleteAfterUpload { store.deleteMedia(id) }
        } else {
            let detail = ServerAPI.errorDetail(from: body)
            fail(id, "HTTP \(status)\(detail.isEmpty ? "" : ": " + detail)", transient: status >= 500)
        }
    }

    @MainActor
    private func fail(_ id: String, _ message: String, transient: Bool) {
        guard let rec = store.recording(id) else { return }
        let attempts = rec.attempts + 1
        if transient && attempts < Self.maxAutoRetries {
            store.update(id) {
                $0.status = .pending
                $0.attempts = attempts
                $0.errorMessage = "\(message) \u{2014} retrying (\(attempts)/\(Self.maxAutoRetries - 1))"
            }
            let delay = UInt64(5 * attempts) * 1_000_000_000
            Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: delay)
                if let r = self?.store.recording(id), r.status == .pending { self?.upload(id) }
            }
        } else {
            store.update(id) {
                $0.status = .failed
                $0.attempts = attempts
                $0.errorMessage = message
            }
        }
    }

    // MARK: URLSession delegate (session queue)

    func urlSession(_ session: URLSession, task: URLSessionTask, didSendBodyData bytesSent: Int64,
                    totalBytesSent: Int64, totalBytesExpectedToSend: Int64) {
        guard let id = task.taskDescription, totalBytesExpectedToSend > 0 else { return }
        let p = Double(totalBytesSent) / Double(totalBytesExpectedToSend)
        Task { @MainActor [store] in store.progress[id] = p }
    }

    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        bufferLock.lock()
        buffers[dataTask.taskIdentifier, default: Data()].append(data)
        bufferLock.unlock()
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        bufferLock.lock()
        let body = buffers.removeValue(forKey: task.taskIdentifier) ?? Data()
        bufferLock.unlock()
        guard let id = task.taskDescription else { return }
        let status = (task.response as? HTTPURLResponse)?.statusCode
        Task { @MainActor [weak self] in
            self?.complete(id: id, status: status, body: body, error: error)
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async {
            AppDelegate.backgroundCompletionHandler?()
            AppDelegate.backgroundCompletionHandler = nil
        }
    }
}
