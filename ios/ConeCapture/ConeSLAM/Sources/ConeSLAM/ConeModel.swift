import Foundation
import simd

/// Cone classes and keypoint rings (port of synth/cone_model.py + cone_app/geometry.py).
public enum ConeModel {
    public static let classNames = [0: "BLUE", 1: "ORANGE_BIG", 2: "ORANGE", 3: "UNDEFINED", 4: "YELLOW"]

    /// Real small cones are the 325 mm FS cone scaled uniformly (geometry.py SMALL_CONE_HEIGHT_M).
    public static var smallConeHeight = 0.3556

    private static let smallRings: [(z: Double, r: Double)] = [(0.220, 0), (0.140, 0), (0.045, 0)].map {
        ($0.0, bodyRadius(z: $0.0, z0: 0.045, r0: 0.080, z1: 0.316, r1: 0.025))
    }
    private static let bigRings: [(z: Double, r: Double)] = [0.370, 0.285, 0.200, 0.115].map {
        ($0, bodyRadius(z: $0, z0: 0.055, r0: 0.100, z1: 0.492, r1: 0.030))
    }

    private static func bodyRadius(z: Double, z0: Double, r0: Double, z1: Double, r1: Double) -> Double {
        r0 + (z - z0) / (z1 - z0) * (r1 - r0)
    }

    public static func scale(_ classID: Int) -> Double { classID == 1 ? 1 : smallConeHeight / 0.325 }

    public static func height(_ classID: Int) -> Double { classID == 1 ? 0.505 : smallConeHeight }

    public static func nKeypoints(_ classID: Int) -> Int { classID == 1 ? 8 : 6 }

    /// (z, r) per ring, top to bottom, metres.
    public static func rings(_ classID: Int) -> [(z: Double, r: Double)] {
        let k = scale(classID)
        return (classID == 1 ? bigRings : smallRings).map { ($0.z * k, $0.r * k) }
    }

    /// Class 3 (UNDEFINED) is compatible with every class.
    @inline(__always) static func compatible(_ a: Int, _ b: Int) -> Bool { a == b || a == 3 || b == 3 }
}

/// Precomputed ring table: index [class][keypoint] -> (z, signed radius); unused keypoints are zero.
struct RingTable {
    static let shared = RingTable()
    let z: [[Double]]
    let r: [[Double]]

    init() {
        var z = [[Double]](repeating: [Double](repeating: 0, count: 8), count: 5)
        var r = z
        for c in 0..<5 {
            for (k, ring) in ConeModel.rings(c).enumerated() {
                for s in 0..<2 {
                    z[c][2 * k + s] = ring.z
                    r[c][2 * k + s] = (s == 0 ? -1 : 1) * ring.r
                }
            }
        }
        self.z = z
        self.r = r
    }
}

/// Silhouette keypoints of a cone standing at `p` (base centre) with axis `up`, seen by `pose`
/// (vectorised slam.py `_pixels`). Writes 8 pixels; entries past nKeypoints are garbage.
@inline(__always)
func projectCone(classID c: Int, p: Vec3, up: Vec3, pose: Pose, K: Intrinsics, into out: inout [Vec2]) {
    let table = RingTable.shared
    var lateral = simd_cross(up, p - pose.t)
    let n = simd_length(lateral)
    lateral = n > 1e-10 ? lateral / n : lateral
    // First tangent must be image-left even under large roll/yaw.
    let camX = Vec3(pose.R[0][0], pose.R[0][1], pose.R[0][2])
    if simd_dot(lateral, camX) < 0 { lateral = -lateral }
    let Rt = pose.R.transpose
    let count = ConeModel.nKeypoints(c)
    let zs = table.z[c], rs = table.r[c]
    for k in 0..<count {
        let w = p + zs[k] * up + rs[k] * lateral
        let cam = Rt * (w - pose.t)
        let depth = max(cam.z, 0.05)
        out[k] = Vec2(K.fx * cam.x / depth + K.cx, K.fy * cam.y / depth + K.cy)
    }
}

/// Public convenience (tests, overlays): keypoints with world up = +z.
public func projectConeKeypoints(classID: Int, pWorld: Vec3, pose: Pose, K: Intrinsics) -> [Vec2] {
    var out = [Vec2](repeating: .zero, count: 8)
    projectCone(classID: classID, p: pWorld, up: Vec3(0, 0, 1), pose: pose, K: K, into: &out)
    return Array(out.prefix(ConeModel.nKeypoints(classID)))
}
