import Foundation

/// WebSocket client for `{base}/api/live/ingest` (cone-live/1).
///
/// Thread-safe and non-blocking: every public method just hops onto a private serial queue, so the
/// camera/SLAM pipeline can call it from anywhere. All mutable state lives on that queue; the
/// `@Published` values are mirrored to the main queue (throttled) for UI.
///
/// Delivery rules per `Kind`:
/// - `.frame`: droppable. At most `maxPendingFrames` wait; the oldest are dropped first.
/// - `.map`: latest wins. Only the newest map is ever pending (and it is resent after a reconnect).
/// - `.control`: never dropped, sent in order (hello / recording / bye).
/// - `.image`: camera preview, sent as a binary message. Latest wins, lowest priority: an image goes out
///   only when nothing else is pending, and a newer image replaces an unsent one.
///
/// On every (re)connect the order is: stored hello, latest map, pending control messages.
final class LiveStreamer: ObservableObject {
    enum Kind { case frame, map, control, image }
    enum ConnectionState: Equatable { case offline, connecting, connected }

    static let maxPendingFrames = 8
    static let minBackoff: TimeInterval = 0.5
    static let maxBackoff: TimeInterval = 10
    private static let byeGrace: TimeInterval = 2
    private static let publishInterval: TimeInterval = 0.25

    @Published private(set) var state: ConnectionState = .offline
    @Published private(set) var sentCount = 0
    @Published private(set) var droppedCount = 0

    private struct Item {
        var data: Data
        var kind: Kind
    }

    private let baseURL: String
    private let token: String?
    private let q = DispatchQueue(label: "LiveStreamer", qos: .userInitiated)
    private let session = URLSession(configuration: .default)

    // All below: confined to `q`.
    private var hello: Data?
    private var latestMap: Data?
    private var pendingMap: Data?
    private var pendingImage: Data?
    private var queue: [Item] = []          // frames and controls, in arrival order
    private var task: URLSessionWebSocketTask?
    private var generation = 0              // invalidates callbacks of dead connections
    private var needHello = false
    private var sending = false
    private var running = false
    private var stopping = false
    private var backoff = LiveStreamer.minBackoff
    private var currentState: ConnectionState = .offline
    private var sent = 0
    private var dropped = 0
    private var publishPending = false

    /// `baseURL` is `AppSettings.normalizedBase` (may be empty), `token` is `AppSettings.trimmedToken`.
    init(baseURL: String, token: String?) {
        self.baseURL = baseURL
        self.token = token
    }

    /// `wss://host/api/live/ingest?token=...` from an http(s) base URL; nil if the base is unusable.
    static func ingestURL(base: String, token: String?) -> URL? {
        guard var c = URLComponents(string: base), let scheme = c.scheme?.lowercased(), c.host != nil else { return nil }
        switch scheme {
        case "https", "wss": c.scheme = "wss"
        case "http", "ws": c.scheme = "ws"
        default: return nil
        }
        while c.path.hasSuffix("/") { c.path.removeLast() }
        c.path += "/api/live/ingest"
        if let token, !token.isEmpty {
            // URLQueryItem leaves "+" and "&" alone; encode strictly so the server decodes the same token.
            var allowed = CharacterSet.alphanumerics
            allowed.insert(charactersIn: "-._~")
            c.percentEncodedQuery = "token=" + (token.addingPercentEncoding(withAllowedCharacters: allowed) ?? "")
        }
        return c.url
    }

    // MARK: Public API

    func start(hello: Data) {
        q.async {
            self.hello = hello
            self.running = true
            self.stopping = false
            self.backoff = Self.minBackoff
            self.queue.removeAll()
            self.connect()
        }
    }

    func send(_ message: Data, kind: Kind) {
        q.async {
            guard self.running else { return }
            switch kind {
            case .frame:
                self.queue.append(Item(data: message, kind: .frame))
                self.trimFrames()
            case .map:
                self.latestMap = message
                if self.pendingMap != nil { self.dropped += 1 }
                self.pendingMap = message
            case .control:
                self.queue.append(Item(data: message, kind: .control))
            case .image:
                if self.pendingImage != nil { self.dropped += 1 }
                self.pendingImage = message
            }
            self.pump()
        }
    }

    /// Sends `bye` after everything pending (up to a short grace period), then closes for good.
    func stop(bye: Data) {
        q.async {
            guard self.running else { return }
            self.stopping = true
            self.queue.append(Item(data: bye, kind: .control))
            let gen = self.generation
            self.q.asyncAfter(deadline: .now() + Self.byeGrace) {
                if self.stopping, self.generation == gen { self.finish() }
            }
            self.pump()
        }
    }

    // MARK: Connection (on q)

    private func connect() {
        guard running, !stopping, task == nil else { return }
        guard let url = Self.ingestURL(base: baseURL, token: token) else {
            setState(.offline)
            return
        }
        var req = URLRequest(url: url, timeoutInterval: 10)
        if let token, !token.isEmpty { req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }

        generation += 1
        let gen = generation
        let t = session.webSocketTask(with: req)
        task = t
        needHello = true
        sending = false
        pendingMap = latestMap
        pendingImage = nil
        queue.removeAll { $0.kind == .frame }       // stale after an outage
        setState(.connecting)
        t.resume()
        receiveLoop(t, gen)
        pump()
    }

    private func receiveLoop(_ t: URLSessionWebSocketTask, _ gen: Int) {
        t.receive { [weak self] result in
            guard let self else { return }
            self.q.async {
                guard gen == self.generation else { return }
                switch result {
                case .success: self.receiveLoop(t, gen)    // ack/ping content is ignored
                case .failure: self.fail(gen)
                }
            }
        }
    }

    private func pump() {
        guard let task, !sending else { return }
        let gen = generation
        let data: Data
        var binary = false
        var restore: Item?
        if needHello {
            guard let hello else { return }
            data = hello
        } else if let map = pendingMap {
            data = map
        } else if !queue.isEmpty {
            let item = queue.removeFirst()
            data = item.data
            restore = item.kind == .control ? item : nil
        } else if let image = pendingImage, !stopping {
            data = image
            binary = true
            pendingImage = nil
        } else {
            if stopping, currentState == .connected { finish() }
            return
        }
        let wasHello = needHello
        sending = true
        let message: URLSessionWebSocketTask.Message = binary ? .data(data) : .string(String(decoding: data, as: UTF8.self))
        task.send(message) { [weak self] error in
            guard let self else { return }
            self.q.async {
                guard gen == self.generation else { return }
                self.sending = false
                if error != nil {
                    if let restore { self.queue.insert(restore, at: 0) }   // controls survive the outage
                    self.fail(gen)
                    return
                }
                if wasHello {
                    self.needHello = false
                    self.backoff = Self.minBackoff
                    self.setState(.connected)
                } else if self.pendingMap == data {
                    self.pendingMap = nil
                }
                self.sent += 1
                self.publish()
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
        if stopping {
            finish()
            return
        }
        setState(.offline)
        let delay = backoff
        backoff = min(backoff * 2, Self.maxBackoff)
        q.asyncAfter(deadline: .now() + delay) { self.connect() }
    }

    private func finish() {
        generation += 1
        task?.cancel(with: .normalClosure, reason: nil)
        task = nil
        sending = false
        running = false
        stopping = false
        queue.removeAll()
        pendingMap = nil
        pendingImage = nil
        setState(.offline)
    }

    private func trimFrames() {
        var count = queue.reduce(0) { $0 + ($1.kind == .frame ? 1 : 0) }
        while count > Self.maxPendingFrames, let i = queue.firstIndex(where: { $0.kind == .frame }) {
            queue.remove(at: i)
            count -= 1
            dropped += 1
        }
    }

    // MARK: Published state (on q -> main)

    private func setState(_ s: ConnectionState) {
        currentState = s
        publish(force: true)
    }

    private func publish(force: Bool = false) {
        if force {
            publishNow()
        } else if !publishPending {
            publishPending = true
            q.asyncAfter(deadline: .now() + Self.publishInterval) {
                self.publishPending = false
                self.publishNow()
            }
        }
    }

    private func publishNow() {
        let (s, n, d) = (currentState, sent, dropped)
        DispatchQueue.main.async {
            self.state = s
            self.sentCount = n
            self.droppedCount = d
        }
    }
}
