import simd

/// Decodes the raw YOLO pose head (one-to-many, `[1, 4 + classes + 3 * keypoints, anchors]`, boxes as
/// centre xywh in model-input pixels, class scores and keypoint confidences already sigmoid) and applies
/// class-aware NMS like ultralytics `non_max_suppression` (best class per anchor).
public enum PoseDecoder {
    /// - Parameters:
    ///   - value: reads channel `c` of anchor `a`.
    ///   - offset: letterbox padding (left, top) subtracted from every coordinate.
    public static func decode(anchors: Int, classes: Int = 5, keypoints: Int = 8, conf: Double, iou: Double = 0.7,
                              maxDetections: Int = 300, offset: Vec2 = .zero,
                              value: (_ channel: Int, _ anchor: Int) -> Float) -> [Detection] {
        var candidates: [(score: Float, cls: Int, anchor: Int)] = []
        let threshold = Float(conf)
        for a in 0..<anchors {
            var best: Float = -1, cls = 0
            for c in 0..<classes {
                let s = value(4 + c, a)
                if s > best { best = s; cls = c }
            }
            if best > threshold { candidates.append((best, cls, a)) }
        }
        candidates.sort { $0.score > $1.score }
        var boxes: [SIMD4<Double>] = []
        var kept: [(score: Float, cls: Int, anchor: Int, box: SIMD4<Double>)] = []
        for c in candidates.prefix(30_000) {
            let cx = Double(value(0, c.anchor)), cy = Double(value(1, c.anchor))
            let w = Double(value(2, c.anchor)), h = Double(value(3, c.anchor))
            let box = SIMD4(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            let overlaps = zip(kept, boxes).contains { k, b in k.cls == c.cls && iouOf(b, box) > iou }
            if overlaps { continue }
            kept.append((c.score, c.cls, c.anchor, box))
            boxes.append(box)
            if kept.count >= maxDetections { break }
        }
        let base = 4 + classes
        return kept.map { k in
            let kp = (0..<keypoints).map { i -> SIMD3<Double> in
                SIMD3(Double(value(base + 3 * i, k.anchor)) - offset.x,
                      Double(value(base + 3 * i + 1, k.anchor)) - offset.y,
                      Double(value(base + 3 * i + 2, k.anchor)))
            }
            let b = k.box - SIMD4(offset.x, offset.y, offset.x, offset.y)
            return Detection(classID: k.cls, conf: Double(k.score), box: b, keypoints: kp)
        }
    }

    @inline(__always) static func iouOf(_ a: SIMD4<Double>, _ b: SIMD4<Double>) -> Double {
        let w = min(a.z, b.z) - max(a.x, b.x), h = min(a.w, b.w) - max(a.y, b.y)
        let inter = max(w, 0) * max(h, 0)
        let union = (a.z - a.x) * (a.w - a.y) + (b.z - b.x) * (b.w - b.y) - inter
        return union > 0 ? inter / union : 0
    }
}
