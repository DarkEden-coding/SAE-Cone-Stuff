import simd

public typealias Vec3 = SIMD3<Double>
public typealias Vec2 = SIMD2<Double>
public typealias Mat3 = simd_double3x3

/// Rigid camera pose `T_world_cam`: `p_world = R * p_cam + t`.
public struct Pose: Equatable {
    public var R: Mat3
    public var t: Vec3

    public init(R: Mat3 = matrix_identity_double3x3, t: Vec3 = .zero) {
        self.R = R
        self.t = t
    }

    /// World point into the camera frame.
    @inline(__always) public func toCamera(_ p: Vec3) -> Vec3 { R.transpose * (p - t) }

    /// Row-major 4x4 as in the JSON schemas.
    public var matrix: [[Double]] {
        (0..<3).map { r in [R[0][r], R[1][r], R[2][r], t[r]] } + [[0, 0, 0, 1]]
    }

    public init?(matrix m: [[Double]]) {
        guard m.count >= 3, m.allSatisfy({ $0.count >= 4 }) else { return nil }
        R = Mat3(rows: (0..<3).map { Vec3(m[$0][0], m[$0][1], m[$0][2]) })
        t = Vec3(m[0][3], m[1][3], m[2][3])
    }

    /// World up axis expressed in the camera frame (third row of R).
    public var upInCamera: Vec3 { Vec3(R[0][2], R[1][2], R[2][2]) }

    /// Right perturbation in the camera frame (as slam.py `_perturb`).
    func perturbedRight(_ d: [Double], _ o: Int = 0) -> Pose {
        let v = Vec3(d[o], d[o + 1], d[o + 2]), w = Vec3(d[o + 3], d[o + 4], d[o + 5])
        return Pose(R: R * expSO3(w), t: t + R * v)
    }

    /// Left (world-frame) perturbation (as slam.py `_bundle_adjust`).
    func perturbedLeft(_ d: [Double], _ o: Int = 0) -> Pose {
        let v = Vec3(d[o], d[o + 1], d[o + 2]), w = Vec3(d[o + 3], d[o + 4], d[o + 5])
        return Pose(R: expSO3(w) * R, t: t + v)
    }
}

public struct Intrinsics: Equatable {
    public var fx, fy, cx, cy: Double
    public init(fx: Double, fy: Double, cx: Double, cy: Double) {
        self.fx = fx; self.fy = fy; self.cx = cx; self.cy = cy
    }
    public init(rowMajor k: [[Double]]) {
        self.init(fx: k[0][0], fy: k[1][1], cx: k[0][2], cy: k[1][2])
    }
    public var rowMajor: [[Double]] { [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] }
    public func scaled(_ s: Double) -> Intrinsics { .init(fx: fx * s, fy: fy * s, cx: cx * s, cy: cy * s) }
}

func skew(_ w: Vec3) -> Mat3 {
    Mat3(rows: [Vec3(0, -w.z, w.y), Vec3(w.z, 0, -w.x), Vec3(-w.y, w.x, 0)])
}

public func expSO3(_ w: Vec3) -> Mat3 {
    let theta = simd_length(w)
    let K = skew(w)
    if theta < 1e-9 { return matrix_identity_double3x3 + K }
    let a = sin(theta) / theta, b = (1 - cos(theta)) / (theta * theta)
    return matrix_identity_double3x3 + a * K + b * (K * K)
}

public func logSO3(_ R: Mat3) -> Vec3 {
    let trace = R[0][0] + R[1][1] + R[2][2]
    let c = max(-1, min(1, (trace - 1) / 2))
    let theta = acos(c)
    // R[col][row]: v = (R32 - R23, R13 - R31, R21 - R12)
    let v = Vec3(R[1][2] - R[2][1], R[2][0] - R[0][2], R[0][1] - R[1][0])
    if theta < 1e-7 { return v / 2 }
    if theta > .pi - 1e-4 {
        // Near pi: axis from the diagonal.
        let d = Vec3(R[0][0], R[1][1], R[2][2])
        var axis = Vec3(sqrt(max(0, (d.x + 1) / 2)), sqrt(max(0, (d.y + 1) / 2)), sqrt(max(0, (d.z + 1) / 2)))
        if R[1][0] + R[0][1] < 0 { axis.y = -axis.y }
        if R[2][0] + R[0][2] < 0 { axis.z = -axis.z }
        return simd_normalize(axis) * theta
    }
    return v * (theta / (2 * sin(theta)))
}

public func rotationAngle(_ R: Mat3) -> Double { simd_length(logSO3(R)) }

/// Minimal rotation taking unit vector u onto +z (slam.py `_level`).
func levelling(_ u: Vec3) -> Mat3 {
    let n = simd_normalize(u)
    let axis = simd_cross(n, Vec3(0, 0, 1))
    let s = simd_length(axis), c = n.z
    if s < 1e-9 { return c > 0 ? matrix_identity_double3x3 : Mat3(diagonal: Vec3(1, -1, -1)) }
    return expSO3(axis / s * atan2(s, c))
}

func rotZ(_ yaw: Double) -> Mat3 {
    let c = cos(yaw), s = sin(yaw)
    return Mat3(rows: [Vec3(c, -s, 0), Vec3(s, c, 0), Vec3(0, 0, 1)])
}

/// The rotation closest to `R` (same heading) whose camera-frame world-up equals `up`.
func levelled(_ R: Mat3, up: Vec3) -> Mat3 {
    let L = levelling(up)                // camera -> a level frame
    // R = Rz(yaw) * L for some yaw; take the yaw that best matches R's heading.
    let M = R * L.transpose
    let yaw = atan2(M[0][1] - M[1][0], M[0][0] + M[1][1])
    return rotZ(yaw) * L
}

/// Heading of the optical axis projected on the ground plane (world z-up).
public func yaw(of pose: Pose) -> Double {
    let forward = pose.R * Vec3(0, 0, 1)
    return atan2(forward.y, forward.x)
}

// MARK: Small dense linear algebra

/// Solve the symmetric positive definite system A x = b (row-major n x n) by Cholesky. nil if not SPD.
func choleskySolve(_ A: [Double], _ b: [Double], _ n: Int) -> [Double]? {
    var L = [Double](repeating: 0, count: n * n)
    for j in 0..<n {
        var d = A[j * n + j]
        for k in 0..<j { d -= L[j * n + k] * L[j * n + k] }
        guard d > 1e-300, d.isFinite else { return nil }
        let ljj = d.squareRoot()
        L[j * n + j] = ljj
        for i in (j + 1)..<max(j + 1, n) {
            var s = A[i * n + j]
            for k in 0..<j { s -= L[i * n + k] * L[j * n + k] }
            L[i * n + j] = s / ljj
        }
    }
    var y = b
    for i in 0..<n {
        var s = y[i]
        for k in 0..<i { s -= L[i * n + k] * y[k] }
        y[i] = s / L[i * n + i]
    }
    for i in stride(from: n - 1, through: 0, by: -1) {
        var s = y[i]
        for k in (i + 1)..<max(i + 1, n) { s -= L[k * n + i] * y[k] }
        y[i] = s / L[i * n + i]
    }
    return y
}

/// Inverse of an SPD matrix (row-major), nil if singular.
func spdInverse(_ A: [Double], _ n: Int) -> [Double]? {
    var out = [Double](repeating: 0, count: n * n)
    for c in 0..<n {
        var e = [Double](repeating: 0, count: n)
        e[c] = 1
        guard let col = choleskySolve(A, e, n) else { return nil }
        for r in 0..<n { out[r * n + c] = col[r] }
    }
    return out
}

func inverse3(_ m: Mat3) -> Mat3? {
    abs(m.determinant) > 1e-300 ? m.inverse : nil
}

func median(_ values: [Double]) -> Double {
    guard !values.isEmpty else { return .nan }
    let s = values.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}
