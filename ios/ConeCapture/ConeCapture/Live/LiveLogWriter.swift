import Foundation

/// Append-only JSONL writer. `append` is cheap and never blocks on disk: lines are buffered on a private
/// serial queue and flushed at least once a second, when the buffer grows large, and on `close()`.
final class LiveLogWriter: @unchecked Sendable {
    private static let flushInterval: TimeInterval = 1
    private static let flushBytes = 64 * 1024

    private let queue = DispatchQueue(label: "LiveLogWriter", qos: .utility)
    private let handle: FileHandle
    private var buffer = Data()
    private var closed = false
    private var timer: DispatchSourceTimer?

    init(url: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !fm.fileExists(atPath: url.path), !fm.createFile(atPath: url.path, contents: nil) {
            throw CocoaError(.fileWriteUnknown)
        }
        handle = try FileHandle(forWritingTo: url)
        try handle.seekToEnd()

        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now() + Self.flushInterval, repeating: Self.flushInterval)
        t.setEventHandler { [weak self] in self?.flush() }
        t.resume()
        timer = t
    }

    deinit { timer?.cancel() }

    /// Appends one message as one line. No-op after `close()`.
    func append(_ message: Data) {
        queue.async {
            guard !self.closed else { return }
            self.buffer.append(message)
            self.buffer.append(0x0A)
            if self.buffer.count >= Self.flushBytes { self.flush() }
        }
    }

    /// Flushes everything buffered and closes the file. Blocks until written; safe to call twice.
    func close() {
        queue.sync {
            guard !closed else { return }
            flush()
            closed = true
            timer?.cancel()
            timer = nil
            try? handle.close()
        }
    }

    private func flush() {   // on queue
        guard !buffer.isEmpty else { return }
        try? handle.write(contentsOf: buffer)
        buffer.removeAll(keepingCapacity: true)
    }
}
