import XCTest
import simd
@testable import ConeSLAM

private struct ProjectionFixture: Decodable {
    struct Case: Decodable {
        var class_id: Int
        var p_world: [Double]
        var T: [[Double]]
        var keypoints: [[Double]]
        var p_cam: [Double]
        var up_cam: [Double]
    }
    var K: [[Double]]
    var cases: [Case]
}

private struct SequenceFixture: Decodable {
    struct Det: Decodable { var c: Int; var s: Double; var b: [Double]; var k: [[Double]] }
    struct Frame: Decodable { var t: Double; var up: [Double]; var d: [Det] }
    var K: [[Double]]
    var size: [Double]
    var frames: [Frame]
    var truth: [[[Double]]]
    var landmarks: [[Double]]
    var classes: [Int]
    var seen: [Int]
}

private func loadFixture<T: Decodable>(_ name: String) throws -> T {
    let url = try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
}

private func v3(_ a: [Double]) -> Vec3 { Vec3(a[0], a[1], a[2]) }

final class GeometryTests: XCTestCase {
    func testProjectionMatchesPython() throws {
        let fixture: ProjectionFixture = try loadFixture("projection")
        let K = Intrinsics(rowMajor: fixture.K)
        XCTAssertGreaterThan(fixture.cases.count, 30)
        for c in fixture.cases {
            let pose = try XCTUnwrap(Pose(matrix: c.T))
            let kp = projectConeKeypoints(classID: c.class_id, pWorld: v3(c.p_world), pose: pose, K: K)
            for (a, b) in zip(kp, c.keypoints) {
                XCTAssertEqual(a.x, b[0], accuracy: 0.01)
                XCTAssertEqual(a.y, b[1], accuracy: 0.01)
            }
        }
    }

    func testSolverRecoversConePosition() throws {
        let fixture: ProjectionFixture = try loadFixture("projection")
        let K = Intrinsics(rowMajor: fixture.K)
        for c in fixture.cases {
            let kp = c.keypoints.map { SIMD3($0[0], $0[1], 1) } + Array(repeating: SIMD3<Double>(0, 0, 0), count: 8 - c.keypoints.count)
            let xs = c.keypoints.map { $0[0] }, ys = c.keypoints.map { $0[1] }
            let det = Detection(classID: c.class_id, conf: 0.9, box: SIMD4(xs.min()!, ys.min()!, xs.max()!, ys.max()!), keypoints: kp)
            let m = try XCTUnwrap(ConeSolver.solve(det, K: K, up: v3(c.up_cam)))
            let truth = v3(c.p_cam)
            XCTAssertLessThan(simd_length(m.pCam - truth), 0.01 * simd_length(truth) + 0.01, "class \(c.class_id) at \(truth)")
            XCTAssertLessThan(m.reprojRMS, 0.05)
        }
    }

    func testHungarianMatchesBruteForce() {
        var generator = SplitMix(seed: 5)
        for (n, m) in [(3, 3), (3, 5), (5, 3), (4, 6), (1, 4)] {
            let cost = (0..<(n * m)).map { _ in Double(generator.next() % 1000) / 10 }
            let assignment = linearSumAssignment(cost, rows: n, cols: m)
            XCTAssertEqual(assignment.count, min(n, m))
            let total = assignment.reduce(0) { $0 + cost[$1.0 * m + $1.1] }
            XCTAssertEqual(total, bruteForce(cost, n, m), accuracy: 1e-9)
        }
    }

    private func bruteForce(_ cost: [Double], _ n: Int, _ m: Int) -> Double {
        if n > m {
            var t = [Double](repeating: 0, count: n * m)
            for i in 0..<n { for j in 0..<m { t[j * n + i] = cost[i * m + j] } }
            return bruteForce(t, m, n)
        }
        var best = Double.infinity
        func go(_ i: Int, _ used: Set<Int>, _ sum: Double) {
            if i == n { best = min(best, sum); return }
            for j in 0..<m where !used.contains(j) { go(i + 1, used.union([j]), sum + cost[i * m + j]) }
        }
        go(0, [], 0)
        return best
    }

    func testRotationRoundTrip() {
        for w in [Vec3(0.1, -0.2, 0.3), Vec3(0, 0, 3.1), Vec3(1e-9, 0, 0), Vec3(2, 1, -0.5)] {
            XCTAssertLessThan(simd_length(logSO3(expSO3(w)) - w), 1e-6)
        }
    }

    func testLevelledKeepsHeadingAndMatchesUp() {
        let R = expSO3(Vec3(0.05, -0.03, 1.2)) * Mat3(rows: [Vec3(0, 0, 1), Vec3(-1, 0, 0), Vec3(0, -1, 0)])
        let up = Pose(R: R).upInCamera
        let noisy = simd_normalize(up + Vec3(0.01, -0.02, 0.005))
        let L = levelled(R, up: noisy)
        XCTAssertLessThan(simd_length(Pose(R: L).upInCamera - noisy), 1e-9)
        XCTAssertLessThan(rotationAngle(L * R.transpose), 0.05)
    }
}

final class LiveSLAMTests: XCTestCase {
    private struct Metrics: CustomStringConvertible {
        var ate: Double, path: Double, landmarkError: Double, nLandmarks: Int, trueLandmarks: Int
        var trackedFraction: Double, msPerFrame: Double, maxMs: Double
        var description: String {
            String(format: "ATE %.3f m / path %.1f m, landmarks %d (true %d) err %.3f m, tracked %.3f, %.2f ms/frame (max %.1f)",
                   ate, path, nLandmarks, trueLandmarks, landmarkError, trackedFraction, msPerFrame, maxMs)
        }
    }

    private func run(_ fixture: SequenceFixture, options: SLAMOptions = SLAMOptions(), dropFrames: Range<Int>? = nil,
                     gravity: Bool = true) -> (Metrics, [SLAMFrameResult]) {
        let slam = LiveSLAM(options: options)
        let K = Intrinsics(rowMajor: fixture.K)
        var results: [SLAMFrameResult] = []
        var times: [Double] = []
        for (i, f) in fixture.frames.enumerated() {
            let dets = dropFrames?.contains(i) == true ? [] : f.d.map { d in
                Detection(classID: d.c, conf: d.s, box: SIMD4(d.b[0], d.b[1], d.b[2], d.b[3]),
                          keypoints: d.k.map { SIMD3($0[0], $0[1], $0[2]) })
            }
            let start = Date()
            results.append(slam.process(SLAMFrame(t: f.t, K: K, imageSize: Vec2(fixture.size[0], fixture.size[1]),
                                                  up: gravity ? v3(f.up) : nil, detections: dets)))
            times.append(Date().timeIntervalSince(start) * 1000)
        }
        // Align the truth to the estimate at the initialisation frame (as test_slam.assess).
        let truth = fixture.truth.map { Pose(matrix: $0)! }
        let initIndex = results.firstIndex { $0.status == .initialized }!
        let estimated = results[initIndex].pose!, reference = truth[initIndex]
        let A = Pose(R: estimated.R * reference.R.transpose, t: estimated.t - estimated.R * reference.R.transpose * reference.t)
        let selected = results.indices.filter { results[$0].status == .tracked || results[$0].status == .initialized }
        let errors = selected.map { simd_length(results[$0].pose!.t - (A.R * truth[$0].t + A.t)) }
        let ate = (errors.reduce(0) { $0 + $1 * $1 } / Double(errors.count)).squareRoot()
        let path = zip(truth, truth.dropFirst()).reduce(0) { $0 + simd_length($1.1.t - $1.0.t) }
        // Match confirmed landmarks to the true cones that were seen.
        let predicted = slam.landmarks.filter(\.confirmed)
        let eligible = fixture.seen
        var cost = [Double](repeating: 1e6, count: predicted.count * eligible.count)
        for (a, lm) in predicted.enumerated() {
            for (b, j) in eligible.enumerated() where lm.classID == fixture.classes[j] {
                cost[a * eligible.count + b] = simd_length(lm.p - (A.R * v3(fixture.landmarks[j]) + A.t))
            }
        }
        let matched = linearSumAssignment(cost, rows: predicted.count, cols: eligible.count)
            .map { cost[$0.0 * eligible.count + $0.1] }.filter { $0 < 1e5 }
        let metrics = Metrics(ate: ate, path: path, landmarkError: matched.reduce(0, +) / Double(max(1, matched.count)),
                              nLandmarks: predicted.count, trueLandmarks: eligible.count,
                              trackedFraction: Double(selected.count) / Double(results.count),
                              msPerFrame: times.reduce(0, +) / Double(times.count), maxMs: times.max() ?? 0)
        return (metrics, results)
    }

    func testLoopFlatGround() throws {
        let fixture: SequenceFixture = try loadFixture("loop_flat")
        let (m, _) = run(fixture)
        print("flat loop:", m)
        XCTAssertGreaterThan(m.trackedFraction, 0.95)
        XCTAssertLessThan(m.ate, 0.04 * m.path)
        XCTAssertLessThan(abs(m.nLandmarks - m.trueLandmarks), Int(0.2 * Double(m.trueLandmarks)))
        XCTAssertLessThan(m.landmarkError, 0.5)
    }

    func testLoopWithoutGravity() throws {
        let fixture: SequenceFixture = try loadFixture("loop_flat")
        var options = SLAMOptions()
        options.gravitySigma = nil
        let (m, _) = run(fixture, options: options, gravity: false)
        print("flat loop, no gravity:", m)
        XCTAssertGreaterThan(m.trackedFraction, 0.9)
        XCTAssertLessThan(m.ate, 0.06 * m.path)
    }

    func testShortSE3() throws {
        let fixture: SequenceFixture = try loadFixture("short_se3")
        var options = SLAMOptions()
        options.flatGround = false
        let (m, _) = run(fixture, options: options)
        print("SE3 short:", m)
        XCTAssertGreaterThan(m.trackedFraction, 0.9)
        XCTAssertLessThan(m.ate, 0.1)
        XCTAssertLessThan(m.landmarkError, 1)
    }

    func testRecoversAfterGap() throws {
        let fixture: SequenceFixture = try loadFixture("loop_flat")
        var short = fixture
        short.frames = Array(fixture.frames.prefix(80))
        short.truth = Array(fixture.truth.prefix(80))
        let (_, results) = run(short, dropFrames: 15..<35)
        XCTAssertTrue(results[15..<35].allSatisfy { $0.status == .lost && $0.pose == nil })
        XCTAssertEqual(results.last?.status, .tracked)
    }

    func testNoDetectionsStaysWaiting() {
        let slam = LiveSLAM()
        let r = slam.process(SLAMFrame(t: 0, K: Intrinsics(fx: 1000, fy: 1000, cx: 640, cy: 360),
                                       imageSize: Vec2(1280, 720), up: Vec3(0, -1, 0), detections: []))
        XCTAssertEqual(r.status, .waiting)
        XCTAssertNil(r.pose)
        XCTAssertTrue(slam.landmarks.isEmpty)
    }
}

final class PoseDecoderTests: XCTestCase {
    private struct Expected: Decodable {
        struct Det: Decodable { var box: [Double]; var conf: Double; var cls: Int; var kpts: [[Double]] }
        var pad: [Double]
        var conf: Double
        var anchors: Int
        var detections: [Det]
    }

    /// Raw CoreML output on a real frame, decoded like ultralytics non_max_suppression.
    func testDecodeMatchesUltralytics() throws {
        let expected: Expected = try loadFixture("decode_expected")
        let url = try XCTUnwrap(Bundle.module.url(forResource: "decode_raw_f16", withExtension: "bin", subdirectory: "Fixtures"))
        let data = try Data(contentsOf: url)
        let raw: [Float] = data.withUnsafeBytes { buffer in buffer.bindMemory(to: Float16.self).map(Float.init) }
        XCTAssertEqual(raw.count, 33 * expected.anchors)
        let n = expected.anchors
        let dets = PoseDecoder.decode(anchors: n, conf: expected.conf, offset: Vec2(expected.pad[0], expected.pad[1])) { raw[$0 * n + $1] }
        XCTAssertEqual(dets.count, expected.detections.count)
        for (d, e) in zip(dets, expected.detections) {
            XCTAssertEqual(d.classID, e.cls)
            XCTAssertEqual(d.conf, e.conf, accuracy: 1e-3)
            for i in 0..<4 { XCTAssertEqual(d.box[i], e.box[i], accuracy: 0.05) }
            for k in 0..<8 {
                XCTAssertEqual(d.keypoints[k].x, e.kpts[k][0], accuracy: 0.05)
                XCTAssertEqual(d.keypoints[k].y, e.kpts[k][1], accuracy: 0.05)
                XCTAssertEqual(d.keypoints[k].z, e.kpts[k][2], accuracy: 1e-3)
            }
        }
    }
}
