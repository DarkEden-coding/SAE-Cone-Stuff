import Foundation
import simd

/// A cone's base position in the camera frame from one detection.
public struct ConeMeasurement: Equatable {
    public var classID: Int
    public var pCam: Vec3
    public var covariance: Mat3
    public var reprojRMS: Double
    public var nKeypoints: Int
}

/// Single-cone localisation. geometry.py uses planar PnP (cone orientation free); with the camera-frame
/// up axis known from gravity, the silhouette model of slam.py has only the 3 position unknowns, so this
/// is a small robust Levenberg-Marquardt fit on the keypoints, started from the apparent ring spacing.
public enum ConeSolver {
    public static func solve(_ d: Detection, K: Intrinsics, up upCam: Vec3, kpConf: Double = 0.5,
                             rangeScale: Double = 1) -> ConeMeasurement? {
        let c = d.classID
        let count = min(d.keypoints.count, ConeModel.nKeypoints(c))
        var pixels = [Vec2](repeating: .zero, count: 8), valid = [Bool](repeating: false, count: 8)
        for k in 0..<count where d.keypoints[k].z >= kpConf && d.keypoints[k].x.isFinite && d.keypoints[k].y.isFinite {
            pixels[k] = Vec2(d.keypoints[k].x, d.keypoints[k].y)
            valid[k] = true
        }
        let nValid = valid.filter { $0 }.count
        guard nValid >= 4, K.fx > 0, K.fy > 0, rangeScale > 0, rangeScale.isFinite else { return nil }
        let up = simd_normalize(upCam)
        guard var p = initialGuess(c, pixels: pixels, valid: valid, K: K, up: up) else { return nil }

        let identity = Pose()
        var pred = [Vec2](repeating: .zero, count: 8)
        func residuals(_ q: Vec3) -> [Double] {
            projectCone(classID: c, p: q, up: up, pose: identity, K: K, into: &pred)
            var r: [Double] = []
            r.reserveCapacity(16)
            for k in 0..<8 where valid[k] {
                let e = pred[k] - pixels[k]
                r.append(e.x); r.append(e.y)
            }
            return r
        }
        var r = residuals(p)
        var cost = cauchy(r)
        var lambda = 1e-3
        for _ in 0..<30 {
            // Forward-difference Jacobian, relative step along the range.
            let h = max(1e-4, 1e-5 * simd_length(p))
            var J = [[Double]](repeating: [], count: 3)
            for a in 0..<3 {
                var q = p
                q[a] += h
                J[a] = zip(residuals(q), r).map { ($0 - $1) / h }
            }
            let w = r.map { 1 / (1 + ($0 / robustScale) * ($0 / robustScale)) }
            var H = [Double](repeating: 0, count: 9), g = [Double](repeating: 0, count: 3)
            for i in 0..<r.count {
                for a in 0..<3 {
                    g[a] += J[a][i] * w[i] * r[i]
                    for b in 0..<3 { H[a * 3 + b] += J[a][i] * w[i] * J[b][i] }
                }
            }
            var improved = false
            var step = Vec3.zero
            for _ in 0..<8 {
                var D = H
                for a in 0..<3 { D[a * 4] += lambda * H[a * 4] + 1e-9 }
                guard let dx = choleskySolve(D, g.map { -$0 }, 3) else { lambda *= 10; continue }
                step = Vec3(dx[0], dx[1], dx[2])
                let candidate = p + step
                guard candidate.z > 0.05 else { lambda *= 10; continue }
                let rc = residuals(candidate)
                let cc = cauchy(rc)
                if cc < cost {
                    p = candidate; r = rc
                    improved = abs(cost - cc) > 1e-6 * max(cost, 1)
                    cost = cc
                    lambda = max(lambda / 3, 1e-7)
                    break
                }
                lambda *= 4
            }
            if !improved || simd_length(step) < 1e-5 { break }
        }
        let rms = (r.reduce(0) { $0 + $1 * $1 } / Double(max(1, r.count / 2))).squareRoot()
        p *= rangeScale
        let distance = simd_length(p)
        guard distance > 0.05, distance < 1000, p.z > 0 else { return nil }
        // Pixel uncertainty yields lateral r*sigma/f and axial r^2*sigma/(f*H), plus a 2% axial floor.
        let sigma = max(1, rms)
        let f = (K.fx * K.fy).squareRoot()
        let lateral = max(0.005, distance * sigma / f)
        let axial = distance * distance * sigma / (f * ConeModel.height(c)) + 0.02 * distance
        let ray = p / distance
        let outer = Mat3(columns: (ray * ray.x, ray * ray.y, ray * ray.z))
        let cov = lateral * lateral * matrix_identity_double3x3 + (axial * axial - lateral * lateral) * outer
        return ConeMeasurement(classID: c, pCam: p, covariance: cov, reprojRMS: rms, nKeypoints: nValid)
    }

    /// Range from the vertical spacing of the outermost valid rings (or the widest ring's width).
    static func initialGuess(_ c: Int, pixels: [Vec2], valid: [Bool], K: Intrinsics, up: Vec3) -> Vec3? {
        let rings = ConeModel.rings(c)
        var centres: [(z: Double, r: Double, px: Vec2, width: Double?)] = []
        for (k, ring) in rings.enumerated() {
            let l = 2 * k, r = l + 1
            if valid[l] && valid[r] {
                centres.append((ring.z, ring.r, (pixels[l] + pixels[r]) / 2, simd_length(pixels[r] - pixels[l])))
            } else if valid[l] || valid[r] {
                centres.append((ring.z, ring.r, valid[l] ? pixels[l] : pixels[r], nil))
            }
        }
        guard let top = centres.first, let bottom = centres.last else { return nil }
        let f = (K.fx * K.fy).squareRoot()
        var range: Double?
        let dz = top.z - bottom.z, dv = simd_length(bottom.px - top.px)
        if dz > 0.03, dv > 1 { range = f * dz / dv }
        if range == nil, let widest = centres.compactMap({ c in c.width.map { (c.r, $0) } }).max(by: { $0.1 < $1.1 }),
           widest.1 > 1 {
            range = f * 2 * widest.0 / widest.1
        }
        guard let distance = range, distance.isFinite, distance > 0.05 else { return nil }
        let mid = centres.reduce(Vec2.zero) { $0 + $1.px } / Double(centres.count)
        let midZ = centres.reduce(0) { $0 + $1.z } / Double(centres.count)
        let ray = simd_normalize(Vec3((mid.x - K.cx) / K.fx, (mid.y - K.cy) / K.fy, 1))
        return ray * distance - up * midZ
    }
}
