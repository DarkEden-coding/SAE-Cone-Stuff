import Foundation

/// Camera preview images on their own WebSocket, `{base}/api/live/ingest/images?session_id=` (SPEC "Camera
/// preview images"), so a 20 KB JPEG never sits in front of pose metadata on the wire.
///
/// One image in flight at most; a newer image replaces the unsent one (latest wins). Reconnects with
/// backoff; the server answers 4404 until the session's hello has arrived on the metadata socket, which the
/// backoff simply rides out. Thread-safe and non-blocking.
final class LiveImageStreamer: @unchecked Sendable {
    private let url: URL?
    private let authorization: String?
    private let q = DispatchQueue(label: "LiveImageStreamer", qos: .utility)
    private let session = URLSession(configuration: .default)

    // Confined to `q`.
    private var task: URLSessionWebSocketTask?
    private var generation = 0
    private var pending: Data?
    private var sending = false
    private var running = false
    private var backoff = LiveStreamer.minBackoff

    init(baseURL: String, token: String?, sessionID: String) {
        url = LiveStreamer.endpointURL(base: baseURL, path: "/api/live/ingest/images", token: token,
                                       query: [("session_id", sessionID)])
        authorization = token.flatMap { $0.isEmpty ? nil : "Bearer \($0)" }
    }

    func start() {
        q.async { self.running = true; self.connect() }
    }

    func send(_ image: Data) {
        q.async {
            guard self.running else { return }
            self.pending = image
            self.pump()
        }
    }

    func stop() {
        q.async {
            self.running = false
            self.generation += 1
            self.task?.cancel(with: .normalClosure, reason: nil)
            self.task = nil
            self.pending = nil
        }
    }

    private func connect() {
        guard running, task == nil, let url else { return }
        var request = URLRequest(url: url, timeoutInterval: 10)
        if let authorization { request.setValue(authorization, forHTTPHeaderField: "Authorization") }
        generation += 1
        let gen = generation
        let t = session.webSocketTask(with: request)
        task = t
        sending = false
        t.resume()
        receive(t, gen)
        pump()
    }

    private func receive(_ t: URLSessionWebSocketTask, _ gen: Int) {
        t.receive { [weak self] result in
            guard let self else { return }
            self.q.async {
                guard gen == self.generation else { return }
                if case .success = result { self.receive(t, gen) } else { self.fail(gen) }
            }
        }
    }

    private func pump() {
        guard let task, !sending, let image = pending else { return }
        pending = nil
        sending = true
        let gen = generation
        task.send(.data(image)) { [weak self] error in
            guard let self else { return }
            self.q.async {
                guard gen == self.generation else { return }
                self.sending = false
                if error != nil { self.fail(gen); return }
                self.backoff = LiveStreamer.minBackoff
                self.pump()
            }
        }
    }

    private func fail(_ gen: Int) {
        guard gen == generation else { return }
        generation += 1
        task?.cancel(with: .goingAway, reason: nil)
        task = nil
        sending = false
        guard running else { return }
        let delay = backoff
        backoff = min(backoff * 2, LiveStreamer.maxBackoff)
        q.asyncAfter(deadline: .now() + delay) { self.connect() }
    }
}
