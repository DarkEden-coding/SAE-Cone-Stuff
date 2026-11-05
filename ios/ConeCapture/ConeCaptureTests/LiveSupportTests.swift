import XCTest
@testable import ConeCapture

final class LiveSupportTests: XCTestCase {
    private func tempURL(_ name: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("LiveSupportTests-\(UUID().uuidString)", isDirectory: true)
            .appendingPathComponent(name)
    }

    func testLogWriterWritesOneLinePerMessageAndFlushesOnClose() throws {
        let url = tempURL("live.jsonl")
        let w = try LiveLogWriter(url: url)
        w.append(Data(#"{"type":"hello"}"#.utf8))
        w.append(Data(#"{"type":"frame","i":0}"#.utf8))
        w.close()
        w.append(Data(#"{"type":"late"}"#.utf8))   // no-op after close
        w.close()                                   // idempotent
        let text = try String(contentsOf: url, encoding: .utf8)
        XCTAssertEqual(text, "{\"type\":\"hello\"}\n{\"type\":\"frame\",\"i\":0}\n")
    }

    func testLogWriterFlushesWithinASecond() throws {
        let url = tempURL("live.jsonl")
        let w = try LiveLogWriter(url: url)
        w.append(Data("{}".utf8))
        let flushed = expectation(description: "flushed")
        DispatchQueue.global().asyncAfter(deadline: .now() + 2.2) {
            if (try? String(contentsOf: url, encoding: .utf8)) == "{}\n" { flushed.fulfill() }
        }
        wait(for: [flushed], timeout: 4)
        w.close()
    }

    func testIngestURLConvertsSchemeAndAddsToken() {
        XCTAssertEqual(LiveStreamer.ingestURL(base: "https://m.tail.ts.net:8444", token: nil)?.absoluteString,
                       "wss://m.tail.ts.net:8444/api/live/ingest")
        XCTAssertEqual(LiveStreamer.ingestURL(base: "http://192.168.1.5:8000", token: "abc")?.absoluteString,
                       "ws://192.168.1.5:8000/api/live/ingest?token=abc")
        XCTAssertEqual(LiveStreamer.ingestURL(base: "https://h/prefix/", token: "a b+c&d")?.absoluteString,
                       "wss://h/prefix/api/live/ingest?token=a%20b%2Bc%26d")
    }

    func testIngestURLRejectsEmptyBase() {
        XCTAssertNil(LiveStreamer.ingestURL(base: "", token: "t"))
    }

    func testRecordingDecodesOldMetaWithoutLiveSession() throws {
        let old = """
        {"id":"20260101-120000","kind":"video","createdAt":"2026-01-01T12:00:00Z","mediaFileName":"video.mp4",
         "mediaBytes":1234,"status":"pending","attempts":0,"mediaDeleted":false}
        """
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        let r = try dec.decode(Recording.self, from: Data(old.utf8))
        XCTAssertNil(r.liveSessionID)
        XCTAssertEqual(r.id, "20260101-120000")
    }

    func testRecordingRoundTripsLiveSession() throws {
        var r = Recording(id: "x", kind: .video, createdAt: Date(timeIntervalSince1970: 0), name: nil,
                          mediaFileName: "video.mp4", mediaBytes: 1, status: .pending, jobID: nil,
                          errorMessage: nil, attempts: 0, mediaDeleted: false)
        r.liveSessionID = "sess-1"
        let back = try JSONDecoder().decode(Recording.self, from: JSONEncoder().encode(r))
        XCTAssertEqual(back.liveSessionID, "sess-1")
    }
}
