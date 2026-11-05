import Foundation
import simd

// Frame-to-map tracking: pixel-space association, robust pose refinement and relocalisation.
// Port of the online parts of cone_app/slam.py, with the camera's up axis from gravity as a prior.

let worldUp = Vec3(0, 0, 1)
let associationFloor = 0.02          // fraction of focal length
let minInliers = 2                   // only with a consistent motion prior; otherwise 3+
let strongInliers = 5                // inliers that justify a pose far from the motion prior
let jumpFraction = 0.7
let tentativeInflation = 2.0
let relocSeeds = 7
let relocMaxHypotheses = 3000
let relocMaxLandmarks = 40

final class Landmark {
    let id: Int
    var classID: Int
    var p: Vec3
    var covariance: Mat3
    var first: Int
    var last: Int
    var n = 1
    var information: Mat3
    var rhs: Vec3
    var ba = false

    init(id: Int, classID: Int, p: Vec3, covariance: Mat3, frame: Int, information: Mat3) {
        self.id = id
        self.classID = classID
        self.p = p
        self.covariance = covariance
        first = frame
        last = frame
        self.information = information
        rhs = information * p
    }
}

/// Landmarks as arrays for association (slam.py `_MapView`).
struct MapView {
    var ids: [Int] = []
    var points: [Vec3] = []
    var classes: [Int] = []
    var n: [Int] = []
    var inflation: [Double] = []
    var count: Int { ids.count }

    init() {}

    init(_ landmarks: [Int: Landmark], minimumN: Int = 0) {
        for lm in landmarks.values.sorted(by: { $0.id < $1.id }) where lm.n >= minimumN {
            ids.append(lm.id); points.append(lm.p); classes.append(lm.classID); n.append(lm.n)
            // Young points carry PnP range bias; let established ones dominate.
            inflation.append(lm.n >= 3 || lm.ba ? 1 : tentativeInflation)
        }
    }

    func subset(_ keep: [Int]) -> MapView {
        var out = MapView()
        for i in keep {
            out.ids.append(ids[i]); out.points.append(points[i]); out.classes.append(classes[i])
            out.n.append(n[i]); out.inflation.append(inflation[i])
        }
        return out
    }
}

/// Soft priors on the refined pose.
struct PosePrior {
    var up: Vec3?                 // expected world up in camera frame (gravity)
    var tiltSigma = 0.02
    var height: Double?           // expected camera height (motion prior)
    var heightSigma = 0.03
    var meanHeight: Double?       // flat mode: operator height about its ~3 s mean
    var meanHeightSigma = 0.15
    var reference: Pose?          // motion-prior pose (for consistency checks)
    var dt = 0.0
}

/// Motion-only refinement problem over a fixed association (slam.py `_Problem`).
struct PoseProblem {
    let K: Intrinsics
    let points: [Vec3]
    let classes: [Int]
    let pixels: [[Vec2]]
    let valid: [[Bool]]
    let sigma: [Double]
    let prior: PosePrior?

    init(dets: Prepared, rows: [Int], view: MapView, cols: [Int], K: Intrinsics, prior: PosePrior?) {
        self.K = K
        points = cols.map { view.points[$0] }
        classes = cols.map { view.classes[$0] }
        pixels = rows.map { dets.pixels[$0] }
        valid = rows.map { dets.valid[$0] }
        sigma = zip(rows, cols).map { dets.sigma[$0.0] * view.inflation[$0.1] }
        self.prior = prior
    }

    func residuals(_ T: Pose) -> [Double] {
        var out: [Double] = []
        out.reserveCapacity(points.count * 16 + 5)
        var pred = [Vec2](repeating: .zero, count: 8)
        for i in 0..<points.count {
            projectCone(classID: classes[i], p: points[i], up: worldUp, pose: T, K: K, into: &pred)
            appendTerms(pred: pred, pixels: pixels[i], valid: valid[i], sigma: sigma[i], into: &out)
        }
        if let prior {
            if let up = prior.up {
                let e = (T.upInCamera - up) / prior.tiltSigma
                out.append(e.x); out.append(e.y); out.append(e.z)
            }
            if let h = prior.height { out.append((T.t.z - h) / prior.heightSigma) }
            if let m = prior.meanHeight { out.append((T.t.z - m) / prior.meanHeightSigma) }
        }
        return out
    }

    /// RMS of the normalised midpoint terms per cone.
    func perConeError(_ T: Pose) -> [Double] {
        var pred = [Vec2](repeating: .zero, count: 8)
        return (0..<points.count).map { i in
            projectCone(classID: classes[i], p: points[i], up: worldUp, pose: T, K: K, into: &pred)
            var sum = 0.0, count = 0
            for ring in 0..<4 {
                let l = 2 * ring, r = l + 1
                let vl = valid[i][l], vr = valid[i][r]
                guard vl || vr else { continue }
                let d = vl && vr ? ((pred[l] - pixels[i][l]) + (pred[r] - pixels[i][r])) / 2
                    : (vl ? pred[l] - pixels[i][l] : pred[r] - pixels[i][r]) * singleWeight
                sum += simd_length_squared(d / sigma[i])
                count += 1
            }
            return (sum / Double(max(1, count))).squareRoot()
        }
    }
}

private let poseEps: [Double] = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5]

/// Levenberg-Marquardt with Cauchy IRLS weights on a right SE(3) perturbation (slam.py `_refine`).
func refine(_ start: Pose, _ problem: PoseProblem, iterations: Int = 15) -> (Pose, Double) {
    var T = start
    var r = problem.residuals(T)
    var cost = cauchy(r)
    var lambda = 1e-3
    for _ in 0..<iterations {
        var J = [[Double]](repeating: [], count: 6)
        for a in 0..<6 {
            var d = [Double](repeating: 0, count: 6)
            d[a] = poseEps[a]
            let ra = problem.residuals(T.perturbedRight(d))
            J[a] = zip(ra, r).map { ($0 - $1) / poseEps[a] }
        }
        var H = [Double](repeating: 0, count: 36), g = [Double](repeating: 0, count: 6)
        for i in 0..<r.count {
            let w = 1 / (1 + (r[i] / robustScale) * (r[i] / robustScale))
            for a in 0..<6 {
                let wa = J[a][i] * w
                g[a] += wa * r[i]
                for b in a..<6 { H[a * 6 + b] += wa * J[b][i] }
            }
        }
        for a in 0..<6 { for b in 0..<a { H[a * 6 + b] = H[b * 6 + a] } }
        var improved = false
        var newCost = cost
        var dx = [Double](repeating: 0, count: 6)
        for _ in 0..<6 {
            var D = H
            for a in 0..<6 { D[a * 7] += lambda * H[a * 7] + 1e-9 }
            guard let step = choleskySolve(D, g.map { -$0 }, 6) else { lambda *= 4; continue }
            dx = step
            let candidate = T.perturbedRight(dx)
            let rc = problem.residuals(candidate)
            newCost = cauchy(rc)
            if newCost < cost {
                T = candidate; r = rc
                improved = true
                lambda = max(lambda / 3, 1e-7)
                break
            }
            lambda *= 4
        }
        let small = dx.map(abs).max() ?? 0 < 1e-4
        if !improved || (abs(cost - newCost) < 1e-4 * max(cost, 1) && small) {
            if improved { cost = newCost }
            break
        }
        cost = newCost
    }
    return (T, cost)
}

struct Association {
    var rows: [Int] = []
    var cols: [Int] = []
    var cost: [Double] = []
}

/// Hungarian association in pixel space (slam.py `_associate`). With `shift` the dominant common image
/// offset (a small camera rotation, the typical hand-held prediction error) is found by voting first.
func associate(_ T: Pose, _ dets: Prepared, _ view: MapView, _ K: Intrinsics, size: Vec2, gate: Double,
               shift: Bool = false) -> Association {
    let N = dets.count, M = view.count
    guard N > 0, M > 0 else { return Association() }
    let width = size.x, height = size.y
    var near: [Int] = []
    var depth = [Double](repeating: 0, count: M)
    for j in 0..<M {
        let cam = T.toCamera(view.points[j])
        depth[j] = cam.z
        guard cam.z > 0.3 else { continue }
        let u = cam.x / cam.z * K.fx + K.cx, v = cam.y / cam.z * K.fy + K.cy
        if u > -0.5 * width, u < 1.5 * width, v > -0.5 * height, v < 1.5 * height { near.append(j) }
    }
    if near.isEmpty { return Association() }
    if near.count < M {
        var a = associate(T, dets, view.subset(near), K, size: size, gate: gate, shift: shift)
        a.cols = a.cols.map { near[$0] }
        return a
    }
    var pred = [[Vec2]](repeating: [Vec2](repeating: .zero, count: 8), count: M)
    for j in 0..<M { projectCone(classID: view.classes[j], p: view.points[j], up: worldUp, pose: T, K: K, into: &pred[j]) }

    // first[i][j][ring], either[i][ring]
    var first = [Vec2](repeating: .zero, count: N * M * 4)
    var either = [Bool](repeating: false, count: N * 4)
    var eitherCount = [Int](repeating: 0, count: N)
    for i in 0..<N {
        for ring in 0..<4 {
            let e = dets.valid[i][2 * ring] || dets.valid[i][2 * ring + 1]
            either[i * 4 + ring] = e
            if e { eitherCount[i] += 1 }
        }
    }
    for i in 0..<N {
        let px = dets.pixels[i], ok = dets.valid[i]
        for j in 0..<M {
            for ring in 0..<4 where either[i * 4 + ring] {
                let l = 2 * ring, r = l + 1
                let dl = pred[j][l] - px[l], dr = pred[j][r] - px[r]
                first[(i * M + j) * 4 + ring] = ok[l] && ok[r] ? (dl + dr) / 2 : (ok[l] ? dl : dr)
            }
        }
    }
    let normaliser = dets.scale.map { $0 + associationFloor * K.fx }
    var compatible = [Bool](repeating: false, count: N * M)
    for i in 0..<N { for j in 0..<M { compatible[i * M + j] = ConeModel.compatible(dets.classes[i], view.classes[j]) && depth[j] > 0.3 } }

    if shift {
        var offsets = [Vec2](repeating: .zero, count: N * M)
        var candidates: [Vec2] = [.zero]
        for i in 0..<N {
            for j in 0..<M {
                var s = Vec2.zero
                for ring in 0..<4 where either[i * 4 + ring] { s += first[(i * M + j) * 4 + ring] }
                offsets[i * M + j] = s / Double(max(1, eitherCount[i]))
                if compatible[i * M + j] { candidates.append(offsets[i * M + j]) }
            }
        }
        if candidates.count > 1 {
            var bestSupport = -1.0, best = 0, zeroSupport = 0.0
            for (c, candidate) in candidates.enumerated() {
                var support = 0.0
                for i in 0..<N {
                    var closest = Double.infinity
                    for j in 0..<M where compatible[i * M + j] {
                        closest = min(closest, simd_length(offsets[i * M + j] - candidate) / normaliser[i])
                    }
                    support += max(0, 1 - closest / 0.6)
                }
                if c == 0 { zeroSupport = support }
                if support > bestSupport { bestSupport = support; best = c }
            }
            if bestSupport > zeroSupport + 0.5 {
                let offset = candidates[best]
                for k in 0..<first.count { first[k] -= offset }
            }
        }
    }

    var total = [Double](repeating: 1e6, count: N * M)
    var costs = [Double](repeating: 0, count: N * M)
    for i in 0..<N {
        for j in 0..<M {
            var sum = 0.0
            for ring in 0..<4 where either[i * 4 + ring] { sum += simd_length_squared(first[(i * M + j) * 4 + ring]) }
            let cost = (sum / Double(max(1, eitherCount[i]))).squareRoot() / normaliser[i]
            costs[i * M + j] = cost
            guard compatible[i * M + j], cost < gate else { continue }
            // Apparent-size consistency is a scale-free depth gate.
            var lo = Vec2(repeating: .infinity), hi = Vec2(repeating: -.infinity)
            for k in 0..<8 where dets.valid[i][k] { lo = simd_min(lo, pred[j][k]); hi = simd_max(hi, pred[j][k]) }
            let ratio = simd_length(hi - lo) / dets.scale[i]
            guard ratio > 0.5, ratio < 2 else { continue }
            total[i * M + j] = cost + 0.1 * abs(log(max(ratio, 1e-3)))
        }
    }
    var out = Association()
    for (i, j) in linearSumAssignment(total, rows: N, cols: M) where total[i * M + j] < 1e6 {
        out.rows.append(i); out.cols.append(j); out.cost.append(costs[i * M + j])
    }
    return out
}

struct TrackResult {
    var T: Pose
    var rows: [Int]
    var cols: [Int]
    var n: Int
    var error: Double
    var truncated: Int
    var missing: Int
    var visible: Int
    var consistent = false
    var assoc: [Int: Int] = [:]          // prepared row -> landmark id

    var score: Double {
        Double(n) + (consistent ? 0.75 : 0) - Double(truncated) * 0.5 - 0.1 * error - 0.5 * Double(missing)
    }
}

/// Landmarks a detector should clearly see from T (negative evidence).
func visible(_ T: Pose, _ view: MapView, _ K: Intrinsics, size: Vec2, minimumPx: Double) -> [Bool] {
    view.points.map { p in
        let cam = T.toCamera(p)
        guard cam.z > 0.5 else { return false }
        let u = K.fx * cam.x / cam.z + K.cx, v = K.fy * cam.y / cam.z + K.cy
        let margin = 0.05 * size.x
        return u > margin && u < size.x - margin && v > margin && v < size.y - margin && K.fy * 0.2 / cam.z > minimumPx
    }
}

/// Associate, refine and re-associate from one hypothesis (slam.py `_localize`).
func localize(_ start: Pose, _ dets: Prepared, _ view: MapView, _ K: Intrinsics, size: Vec2, prior: PosePrior?,
              visiblePx: Double, gates: [Double] = [1, 0.45, 0.3]) -> TrackResult? {
    var T = start
    var rows: [Int]?, cols: [Int]?
    for (step, gate) in gates.enumerated() {
        let a = associate(T, dets, view, K, size: size, gate: gate, shift: step == 0)
        if a.rows.count < minInliers { return nil }
        if let rows, let cols, rows == a.rows, cols == a.cols { break }
        rows = a.rows; cols = a.cols
        T = refine(T, PoseProblem(dets: dets, rows: a.rows, view: view, cols: a.cols, K: K, prior: prior)).0
    }
    let a = associate(T, dets, view, K, size: size, gate: gates.last!)
    guard a.rows.count >= minInliers else { return nil }
    var r = a.rows, c = a.cols
    var problem = PoseProblem(dets: dets, rows: r, view: view, cols: c, K: K, prior: prior)
    var errors = problem.perConeError(T)
    let good = errors.map { $0 < 3 }
    guard good.filter({ $0 }).count >= minInliers else { return nil }
    if good.contains(false) {
        r = zip(r, good).filter { $0.1 }.map { $0.0 }
        c = zip(c, good).filter { $0.1 }.map { $0.0 }
        problem = PoseProblem(dets: dets, rows: r, view: view, cols: c, K: K, prior: prior)
        T = refine(T, problem, iterations: 8).0
        errors = problem.perConeError(T)
    }
    let vis = visible(T, view, K, size: size, minimumPx: visiblePx)
    let established = (0..<view.count).filter { vis[$0] && view.n[$0] >= 3 }
    let matchedEstablished = c.filter { vis[$0] && view.n[$0] >= 3 }.count
    return TrackResult(T: T, rows: r, cols: c, n: r.count, error: median(errors),
                       truncated: r.filter { dets.truncated[$0] }.count,
                       missing: established.count - matchedEstablished, visible: established.count)
}

func isConsistent(_ T: Pose, _ prior: PosePrior?) -> Bool {
    guard let prior, let reference = prior.reference else { return false }
    return simd_length(T.t - reference.t) < 0.6 + 1.5 * prior.dt &&
        rotationAngle(T.R * reference.R.transpose) < 0.2 + 2.5 * prior.dt
}

func isAdmissible(_ r: TrackResult, usable: Int) -> Bool {
    // Too few cones to outvote the many mapped cones that should be seen.
    if r.n < strongInliers && r.missing > r.n { return false }
    if r.consistent { return r.n >= 3 || (r.n >= 2 && r.error < 1.5) }
    // A jump away from the motion prior needs broad support.
    return r.n >= max(strongInliers, Int(ceil(jumpFraction * Double(usable)))) &&
        Double(r.missing) <= 0.25 * Double(max(r.visible, 1))
}

/// Association-free pose hypotheses from class-compatible pairs of cones (slam.py `_relocalize_hypotheses`).
func relocalizationHypotheses(_ ms: [Int: ConeMeasurement], _ dets: Prepared, _ view: MapView, ups: [Vec3],
                              flat: Bool, top: Int = 6) -> [Pose] {
    let index = (0..<dets.count).filter { ms[$0] != nil && !dets.truncated[$0] }
    guard index.count >= 2, view.count >= 2 else { return [] }
    let q = index.map { ms[$0]!.pCam }
    let qc = index.map { dets.classes[$0] }
    let rangeQ = q.map { simd_length($0) }
    let P = view.points, Pc = view.classes
    // Seed pairs come from the nearest well-fitted cones; all cones score.
    let quality = index.indices.map { rangeQ[$0] + (ms[index[$0]]!.reprojRMS > 4 ? 5 : 0) }
    let seeds = Array(quality.indices.sorted { quality[$0] < quality[$1] }.prefix(relocSeeds)).sorted()
    var seedPairs: [(Int, Int)] = []
    for a in 0..<seeds.count { for b in (a + 1)..<max(a + 1, seeds.count) { seedPairs.append((seeds[a], seeds[b])) } }
    var mapPairs: [(Int, Int)] = []
    for a in 0..<P.count { for b in 0..<P.count where a != b { mapPairs.append((a, b)) } }

    var hypotheses: [(Double, Pose)] = []
    for u in ups {
        let R1 = levelling(u)
        let ql = q.map { R1 * $0 }
        let meanZ = ql.reduce(0) { $0 + $1.z } / Double(ql.count)
        var candidates: [(Int, Int, Double)] = []     // seed pair, map pair, scale
        for (pi, (I, J)) in seedPairs.enumerated() {
            let dq = ql[J] - ql[I]
            let hq = simd_length(Vec2(dq.x, dq.y))
            guard hq > 0.3 else { continue }
            for (pa, (A, B)) in mapPairs.enumerated() {
                guard ConeModel.compatible(qc[I], Pc[A]), ConeModel.compatible(qc[J], Pc[B]) else { continue }
                let dP = P[B] - P[A]
                let hP = simd_length(Vec2(dP.x, dP.y))
                let scale = hP / hq
                guard hP > 0.3, scale > 0.6, scale < 1.6 else { continue }
                if !flat && abs(dq.z * scale - dP.z) >= 0.3 + 0.2 * hP { continue }
                candidates.append((pi, pa, scale))
            }
        }
        if candidates.isEmpty { continue }
        if candidates.count > relocMaxHypotheses {
            var generator = SplitMix(seed: UInt64(candidates.count))
            candidates.shuffle(using: &generator)
            candidates = Array(candidates.prefix(relocMaxHypotheses))
        }
        var scored: [(Double, Pose)] = []
        scored.reserveCapacity(candidates.count)
        for (pi, pa, k) in candidates {
            let (I, J) = seedPairs[pi], (A, B) = mapPairs[pa]
            let dq = ql[J] - ql[I], dP = P[B] - P[A]
            let yawAngle = atan2(dP.y, dP.x) - atan2(dq.y, dq.x)
            let R = rotZ(yawAngle) * R1
            var t = (P[A] + P[B]) / 2 - k * (R * ((q[I] + q[J]) / 2))
            // Camera height follows from the levelled cone heights.
            if flat { t.z = -k * meanZ }
            var score = 0.0
            for n in 0..<q.count {
                let world = k * (R * q[n]) + t
                let tolerance = 0.2 * rangeQ[n] + 0.2
                var close = 1e3
                for m in 0..<P.count where ConeModel.compatible(qc[n], Pc[m]) {
                    let d = world - P[m]
                    let distance = flat ? simd_length(Vec2(d.x, d.y)) : simd_length(d)
                    close = min(close, distance / tolerance)
                }
                score += max(0, 1 - close * close)
            }
            scored.append((score, Pose(R: R, t: t)))
        }
        scored.sort { $0.0 > $1.0 }
        hypotheses += scored.prefix(200)
    }
    hypotheses.sort { $0.0 > $1.0 }
    var chosen: [Pose] = []
    for (_, T) in hypotheses {
        if chosen.allSatisfy({ simd_length(T.t - $0.t) > 0.3 || rotationAngle(T.R.transpose * $0.R) > 0.1 }) {
            chosen.append(T)
        }
        if chosen.count >= top { break }
    }
    return chosen
}

/// Deterministic generator for hypothesis subsampling.
struct SplitMix: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
