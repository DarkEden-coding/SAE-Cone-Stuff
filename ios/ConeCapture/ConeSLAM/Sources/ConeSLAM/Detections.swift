import Foundation
import simd

/// One detector output in image pixels: box and 8 keypoints (x, y, confidence).
public struct Detection: Equatable {
    public var classID: Int
    public var conf: Double
    public var box: SIMD4<Double>            // x1, y1, x2, y2
    public var keypoints: [SIMD3<Double>]    // 8 entries

    public init(classID: Int, conf: Double, box: SIMD4<Double>, keypoints: [SIMD3<Double>]) {
        self.classID = classID
        self.conf = conf
        self.box = box
        self.keypoints = keypoints
    }
}

/// Box-height dependent keypoint bias correction (port of geometry.KeypointCorrection).
public struct KeypointCorrection {
    let centres: [Double]                    // log of bin centre heights
    let offsets: [[[Double]]]                // [bin][kp][inward, below] as fraction of box height
    let rangeScales: [Double]                // per bin, empty if unknown
    /// Multiplies the box height before the table lookup (table px per image px).
    public var heightScale: Double = 1

    public init(json data: Data) throws {
        guard var object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CocoaError(.coderReadCorrupt)
        }
        if object["edges_px"] == nil {
            let tables = object.values.compactMap { $0 as? [String: Any] }.filter { $0["edges_px"] != nil }
            guard tables.count == 1 else { throw CocoaError(.coderReadCorrupt) }
            object = tables[0]
        }
        guard let edges = object["edges_px"] as? [Double], edges.count >= 2,
              let raw = object["offsets_inward_below_frac_h"] as? [[[Any]]], raw.count == edges.count - 1
        else { throw CocoaError(.coderReadCorrupt) }
        centres = (0..<(edges.count - 1)).map { log((edges[$0] * edges[$0 + 1]).squareRoot()) }
        offsets = raw.map { bin in bin.map { kp in kp.map { ($0 as? Double) ?? .nan } } }
        if let scales = object["range_scale"] as? [Any] {
            rangeScales = scales.map { ($0 as? Double) ?? .nan }
        } else if let bias = object["range_bias_frac"] as? [Any] {
            rangeScales = bias.map { b in (b as? Double).map { $0 > -1 ? 1 / (1 + $0) : .nan } ?? .nan }
        } else {
            rangeScales = []
        }
    }

    private func interp(_ values: [Double], _ height: Double, _ fallback: Double) -> Double {
        guard values.count == centres.count else { return fallback }
        let pairs = zip(centres, values).filter { $0.1.isFinite }
        guard let first = pairs.first, let last = pairs.last else { return fallback }
        let x = log(max(height, 1e-6))
        if x <= first.0 { return first.1 }
        if x >= last.0 { return last.1 }
        for i in 1..<pairs.count where x <= pairs[i].0 {
            let (x0, y0) = pairs[i - 1], (x1, y1) = pairs[i]
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        }
        return last.1
    }

    public func apply(_ d: Detection) -> Detection {
        var out = d
        let h = d.box.w - d.box.y
        guard h.isFinite, h > 0 else { return out }
        let hh = h * heightScale
        let n = min(d.keypoints.count, offsets.first?.count ?? 0)
        for k in 0..<n {
            let inward = interp(offsets.map { $0[k][0] }, hh, 0)
            let below = interp(offsets.map { $0[k][1] }, hh, 0)
            out.keypoints[k].x -= h * (k % 2 == 0 ? 1 : -1) * inward
            out.keypoints[k].y -= h * below
        }
        return out
    }

    public func rangeScale(boxHeight h: Double) -> Double {
        guard h.isFinite, h > 0, !rangeScales.isEmpty else { return 1 }
        let s = interp(rangeScales, h * heightScale, 1)
        return s.isFinite && s > 0 ? s : 1
    }
}

/// Per-frame detections usable for tracking (slam.py `_Detections`).
struct Prepared {
    var index: [Int] = []                    // index into the frame's detection list
    var classes: [Int] = []
    var pixels: [[Vec2]] = []                // 8 per detection
    var valid: [[Bool]] = []
    var sigma: [Double] = []
    var scale: [Double] = []                 // apparent size, px
    var truncated: [Bool] = []
    var count: Int { index.count }

    init() {}

    init(_ detections: [Detection], width: Double, height: Double, kpConf: Double) {
        for i in deduplicated(detections) {
            let d = detections[i]
            let n = ConeModel.nKeypoints(d.classID)
            var px = [Vec2](repeating: .zero, count: 8), ok = [Bool](repeating: false, count: 8)
            var lo = Vec2(repeating: .infinity), hi = Vec2(repeating: -.infinity)
            for k in 0..<min(n, d.keypoints.count) {
                let kp = d.keypoints[k]
                guard kp.x.isFinite, kp.y.isFinite, kp.z >= kpConf else { continue }
                px[k] = Vec2(kp.x, kp.y)
                ok[k] = true
                lo = simd_min(lo, px[k]); hi = simd_max(hi, px[k])
            }
            guard ok.filter({ $0 }).count >= 3 else { continue }
            let margin = 4.0
            let edge = d.box.x < margin || d.box.y < margin || d.box.z > width - margin || d.box.w > height - margin
            let extent = max(simd_length(hi - lo), 8)
            index.append(i)
            classes.append(d.classID)
            pixels.append(px)
            valid.append(ok)
            scale.append(extent)
            sigma.append((1.5 + 0.02 * extent) * (edge ? 3 : 1))
            truncated.append(edge)
        }
    }
}

/// Drop overlapping boxes of compatible classes (detector duplicates); returns kept indices in order.
func deduplicated(_ detections: [Detection], iou threshold: Double = 0.3) -> [Int] {
    let order = detections.indices.sorted { detections[$0].conf > detections[$1].conf }
    var kept: [Int] = []
    for i in order {
        let a = detections[i].box
        let duplicate = kept.contains { j in
            let b = detections[j].box
            guard ConeModel.compatible(detections[i].classID, detections[j].classID) else { return false }
            let w = min(a.z, b.z) - max(a.x, b.x), h = min(a.w, b.w) - max(a.y, b.y)
            let inter = max(w, 0) * max(h, 0)
            let union = (a.z - a.x) * (a.w - a.y) + (b.z - b.x) * (b.w - b.y) - inter
            return union > 0 && inter / union > threshold
        }
        if !duplicate { kept.append(i) }
    }
    return kept.sorted()
}

// MARK: Residual terms

let halfWeight = 0.35     // half-width residuals are blur/bias prone
let singleWeight = 0.7    // lone keypoint of a ring pair
let robustScale = 2.5     // Cauchy / soft-L1 scale in sigma units

/// Up to 16 normalised residuals (per ring: midpoint x/y, then half-width x/y) appended to `out`,
/// using only the entries slam.py's `_term_mask` keeps. Order is fixed for a given `valid`.
@inline(__always)
func appendTerms(pred: [Vec2], pixels: [Vec2], valid: [Bool], sigma: Double, into out: inout [Double]) {
    for ring in 0..<4 {
        let l = 2 * ring, r = l + 1
        let vl = valid[l], vr = valid[r]
        if vl && vr {
            let mid = ((pred[l] - pixels[l]) + (pred[r] - pixels[r])) / 2
            let half = ((pred[r] - pixels[r]) - (pred[l] - pixels[l])) / 2 * halfWeight
            out.append(mid.x / sigma); out.append(mid.y / sigma)
            out.append(half.x / sigma); out.append(half.y / sigma)
        } else if vl || vr {
            let k = vl ? l : r
            let d = (pred[k] - pixels[k]) * singleWeight
            out.append(d.x / sigma); out.append(d.y / sigma)
        }
    }
}

func termCount(_ valid: [Bool]) -> Int {
    (0..<4).reduce(0) { n, ring in
        let vl = valid[2 * ring], vr = valid[2 * ring + 1]
        return n + (vl && vr ? 4 : (vl || vr ? 2 : 0))
    }
}

@inline(__always) func cauchy(_ r: [Double]) -> Double {
    r.reduce(0) { $0 + log1p(($1 / robustScale) * ($1 / robustScale)) }
}

@inline(__always) func softL1(_ r: [Double]) -> Double {
    r.reduce(0) { $0 + 2 * ((1 + ($1 / robustScale) * ($1 / robustScale)).squareRoot() - 1) } * robustScale * robustScale
}
