import Foundation
import simd

struct Observation {
    var frame: Int
    var landmark: Int
    var classID: Int
    var pixels: [Vec2]
    var valid: [Bool]
    var K: Intrinsics
    var sigma: Double
}

/// Sparse Levenberg-Marquardt bundle adjustment with the camera blocks eliminated (slam.py
/// `_bundle_adjust`), plus a soft gravity prior on every free keyframe's tilt.
///
/// Poses use a world-frame perturbation (R <- exp(w) R, t <- t + v). Each camera block is inverted
/// independently, leaving a small dense Schur system over the landmarks. The first keyframe fixes the
/// gauge; in flat global passes only its ground position and yaw are pinned, so its tilt and height adapt.
func bundleAdjust(poses: inout [Int: Pose], ups: [Int: Vec3], landmarks: [Int: Landmark], keyframes: [Int],
                  observations allObservations: [Observation], flat: Bool, global: Bool, tiltSigma: Double,
                  iterations: Int? = nil) {
    let selected = Set(keyframes)
    var obs = allObservations.filter { selected.contains($0.frame) && landmarks[$0.landmark] != nil }
    guard obs.count >= 6, keyframes.count >= 2 else { return }
    var counts: [Int: Int] = [:]
    for o in obs { counts[o.landmark, default: 0] += 1 }
    // A single-view point is not constrained enough to belong in BA.
    obs = obs.filter { counts[$0.landmark]! >= 2 }
    guard !obs.isEmpty else { return }
    let anchor = keyframes[0]
    let partialAnchor = flat && global
    let framesUsed = Array(Set(obs.map(\.frame))).sorted()
    let freeFrames = framesUsed.filter { $0 != anchor || partialAnchor }
    guard !freeFrames.isEmpty else { return }
    let poseIndex = Dictionary(uniqueKeysWithValues: freeFrames.enumerated().map { ($1, $0) })
    let landmarkIDs = Array(Set(obs.map(\.landmark))).sorted()
    let lmIndex = Dictionary(uniqueKeysWithValues: landmarkIDs.enumerated().map { ($1, $0) })
    let ld = flat ? 2 : 3
    let P = freeFrames.count, L = landmarkIDs.count, n = obs.count
    var free = freeFrames.map { poses[$0]! }
    var points = landmarkIDs.map { landmarks[$0]!.p }
    if flat { for i in 0..<L { points[i].z = 0 } }
    let oi = obs.map { poseIndex[$0.frame] ?? -1 }
    let oj = obs.map { lmIndex[$0.landmark]! }
    let fixed = obs.map { poses[$0.frame]! }
    let tilt = freeFrames.map { ups[$0] }
    let byPose: [[Int]] = {
        var groups = [[Int]](repeating: [], count: P)
        for k in 0..<n where oi[k] >= 0 { groups[oi[k]].append(k) }
        return groups
    }()
    var pred = [Vec2](repeating: .zero, count: 8)

    func observationResiduals(_ k: Int, _ T: Pose, _ p: Vec3) -> [Double] {
        var out: [Double] = []
        out.reserveCapacity(16)
        projectCone(classID: obs[k].classID, p: p, up: worldUp, pose: T, K: obs[k].K, into: &pred)
        appendTerms(pred: pred, pixels: obs[k].pixels, valid: obs[k].valid, sigma: obs[k].sigma, into: &out)
        return out
    }
    func tiltResiduals(_ i: Int, _ T: Pose) -> [Double] {
        guard let up = tilt[i] else { return [] }
        let e = (T.upInCamera - up) / tiltSigma
        return [e.x, e.y, e.z]
    }
    func pose(_ k: Int, _ poses: [Pose]) -> Pose { oi[k] >= 0 ? poses[oi[k]] : fixed[k] }
    func allResiduals(_ poses: [Pose], _ points: [Vec3]) -> [[Double]] {
        (0..<n).map { observationResiduals($0, pose($0, poses), points[oj[$0]]) } + (0..<P).map { tiltResiduals($0, poses[$0]) }
    }
    func totalCost(_ r: [[Double]]) -> Double { r.reduce(0) { $0 + softL1($1) } }
    func moved(_ p: Vec3, _ dl: [Double], _ j: Int) -> Vec3 {
        var q = p
        for a in 0..<ld { q[a] += dl[j * ld + a] }
        return q
    }

    let eps: [Double] = [1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5]
    var r = allResiduals(free, points)
    var cost = totalCost(r)
    var lambda = 1e-3
    var finalS: [Double]?
    let maxIterations = iterations ?? (global ? 15 : 6)
    for _ in 0..<maxIterations {
        var Hpp = [Double](repeating: 0, count: P * 36), gp = [Double](repeating: 0, count: P * 6)
        var Hll = [Double](repeating: 0, count: L * ld * ld), gl = [Double](repeating: 0, count: L * ld)
        var Hpl = [[Double]](repeating: [], count: n)          // 6 x ld per observation
        for k in 0..<n {
            let rk = r[k], m = rk.count
            let w = rk.map { 1 / (1 + ($0 / robustScale) * ($0 / robustScale)).squareRoot() }
            let T = pose(k, free), p = points[oj[k]]
            var Jp = [[Double]](repeating: [Double](repeating: 0, count: m), count: 6)
            if oi[k] >= 0 {
                for a in 0..<6 {
                    var d = [Double](repeating: 0, count: 6)
                    d[a] = eps[a]
                    let ra = observationResiduals(k, T.perturbedLeft(d), p)
                    for e in 0..<m { Jp[a][e] = (ra[e] - rk[e]) / eps[a] }
                }
            }
            var Jl = [[Double]](repeating: [Double](repeating: 0, count: m), count: ld)
            for a in 0..<ld {
                var q = p
                q[a] += 1e-4
                let ra = observationResiduals(k, T, q)
                for e in 0..<m { Jl[a][e] = (ra[e] - rk[e]) / 1e-4 }
            }
            let j = oj[k]
            for a in 0..<ld {
                for e in 0..<m { gl[j * ld + a] += Jl[a][e] * w[e] * rk[e] }
                for b in 0..<ld {
                    var s = 0.0
                    for e in 0..<m { s += Jl[a][e] * w[e] * Jl[b][e] }
                    Hll[j * ld * ld + a * ld + b] += s
                }
            }
            var block = [Double](repeating: 0, count: 6 * ld)
            if oi[k] >= 0 {
                let i = oi[k]
                for a in 0..<6 {
                    for e in 0..<m { gp[i * 6 + a] += Jp[a][e] * w[e] * rk[e] }
                    for b in 0..<6 {
                        var s = 0.0
                        for e in 0..<m { s += Jp[a][e] * w[e] * Jp[b][e] }
                        Hpp[i * 36 + a * 6 + b] += s
                    }
                    for b in 0..<ld {
                        var s = 0.0
                        for e in 0..<m { s += Jp[a][e] * w[e] * Jl[b][e] }
                        block[a * ld + b] = s
                    }
                }
            }
            Hpl[k] = block
        }
        // Gravity prior: pose-only terms.
        for i in 0..<P where tilt[i] != nil {
            let ri = r[n + i]
            let w = ri.map { 1 / (1 + ($0 / robustScale) * ($0 / robustScale)).squareRoot() }
            var J = [[Double]](repeating: [0, 0, 0], count: 6)
            for a in 3..<6 {
                var d = [Double](repeating: 0, count: 6)
                d[a] = eps[a]
                let ra = tiltResiduals(i, free[i].perturbedLeft(d))
                for e in 0..<3 { J[a][e] = (ra[e] - ri[e]) / eps[a] }
            }
            for a in 3..<6 {
                for e in 0..<3 { gp[i * 6 + a] += J[a][e] * w[e] * ri[e] }
                for b in 3..<6 { for e in 0..<3 { Hpp[i * 36 + a * 6 + b] += J[a][e] * w[e] * J[b][e] } }
            }
        }
        if partialAnchor, let a = poseIndex[anchor] {
            for d in [0, 1, 5] { Hpp[a * 36 + d * 7] += 1e8 }      // ground position and yaw
        }

        var accepted = false
        var newCost = cost
        var newFree = free, newPoints = points, newR = r
        var S = [Double]()
        while true {
            var DpInv = [[Double]](repeating: [], count: P)
            var singular = false
            for i in 0..<P {
                var D = Array(Hpp[(i * 36)..<(i * 36 + 36)])
                for a in 0..<6 { D[a * 7] += lambda * D[a * 7] + 1e-9 }
                guard let inv = spdInverse(D, 6) else { singular = true; break }
                DpInv[i] = inv
            }
            let size = L * ld
            S = [Double](repeating: 0, count: size * size)
            if !singular {
                for j in 0..<L {
                    for a in 0..<ld {
                        for b in 0..<ld {
                            let h = Hll[j * ld * ld + a * ld + b]
                            S[(j * ld + a) * size + j * ld + b] = h + (a == b ? lambda * h + 1e-9 : 0)
                        }
                    }
                }
                for i in 0..<P {
                    let group = byPose[i], inv = DpInv[i]
                    // Y_b = Dp^-1 Hpl_b (6 x ld)
                    let Y = group.map { b -> [Double] in
                        var y = [Double](repeating: 0, count: 6 * ld)
                        for a in 0..<6 { for c in 0..<ld {
                            var s = 0.0
                            for e in 0..<6 { s += inv[a * 6 + e] * Hpl[b][e * ld + c] }
                            y[a * ld + c] = s
                        } }
                        return y
                    }
                    for (x, a) in group.enumerated() {
                        for (yIndex, b) in group.enumerated() {
                            let ja = oj[a], jb = oj[b]
                            for c in 0..<ld { for d in 0..<ld {
                                var s = 0.0
                                for e in 0..<6 { s += Hpl[a][e * ld + c] * Y[yIndex][e * ld + d] }
                                S[(ja * ld + c) * size + jb * ld + d] -= s
                            } }
                            _ = x
                        }
                    }
                }
            }
            var rhs = gl
            if !singular {
                for k in 0..<n where oi[k] >= 0 {
                    let i = oi[k], inv = DpInv[i]
                    var v = [Double](repeating: 0, count: 6)
                    for a in 0..<6 { for e in 0..<6 { v[a] += inv[a * 6 + e] * gp[i * 6 + e] } }
                    for c in 0..<ld {
                        var s = 0.0
                        for e in 0..<6 { s += Hpl[k][e * ld + c] * v[e] }
                        rhs[oj[k] * ld + c] -= s
                    }
                }
            }
            guard !singular, let dl = choleskySolve(S, rhs.map { -$0 }, size) else {
                lambda *= 10
                if lambda > 1e6 { break }
                continue
            }
            var coupling = [Double](repeating: 0, count: P * 6)
            for k in 0..<n where oi[k] >= 0 {
                for a in 0..<6 { for c in 0..<ld { coupling[oi[k] * 6 + a] += Hpl[k][a * ld + c] * dl[oj[k] * ld + c] } }
            }
            for i in 0..<P {
                var dp = [Double](repeating: 0, count: 6)
                for a in 0..<6 { for e in 0..<6 { dp[a] += DpInv[i][a * 6 + e] * (-gp[i * 6 + e] - coupling[i * 6 + e]) } }
                newFree[i] = free[i].perturbedLeft(dp)
            }
            for j in 0..<L { newPoints[j] = moved(points[j], dl, j) }
            newR = allResiduals(newFree, newPoints)
            newCost = totalCost(newR)
            if newCost < cost { accepted = true; break }
            lambda *= 10
            if lambda > 1e6 { break }
        }
        guard accepted else { break }
        let improvement = (cost - newCost) / max(cost, 1e-9)
        free = newFree; points = newPoints; r = newR; cost = newCost
        lambda = max(lambda / 5, 1e-7)
        finalS = S
        if improvement < 1e-4 { break }
    }
    for (f, i) in poseIndex { poses[f] = free[i] }
    for (id, j) in lmIndex {
        let lm = landmarks[id]!
        lm.p = points[j]
        lm.rhs = lm.information * lm.p
        lm.ba = true
    }
    // Marginal landmark covariance from the Schur complement.
    if let S = finalS, let covariance = spdInverse(S, L * ld) {
        let size = L * ld
        for (id, j) in lmIndex {
            let lm = landmarks[id]!
            for a in 0..<ld {
                for b in 0..<ld { lm.covariance[b][a] = a == b ? max(covariance[(j * ld + a) * size + j * ld + a], 0.01 * 0.01) : 0 }
            }
        }
    }
}
