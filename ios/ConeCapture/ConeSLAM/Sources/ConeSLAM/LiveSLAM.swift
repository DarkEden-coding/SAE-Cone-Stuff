import Foundation
import simd

/// One camera frame for the live SLAM.
public struct SLAMFrame {
    public var t: Double
    public var K: Intrinsics
    public var imageSize: Vec2
    /// World up (opposite to gravity) in the camera frame, unit length. nil: assume a level camera.
    public var up: Vec3?
    /// Keypoints already bias-corrected.
    public var detections: [Detection]
    /// Range scale per detection (keypoint correction table), 1 if unknown.
    public var rangeScales: [Double]?

    public init(t: Double, K: Intrinsics, imageSize: Vec2, up: Vec3?, detections: [Detection], rangeScales: [Double]? = nil) {
        self.t = t; self.K = K; self.imageSize = imageSize; self.up = up
        self.detections = detections; self.rangeScales = rangeScales
    }
}

public struct SLAMOptions {
    public var kpConf = 0.5
    public var maxRange = 40.0
    public var flatGround = true
    /// Gravity tilt prior (rad). nil disables it (cones-only tilt, like the server SLAM).
    public var gravitySigma: Double? = 0.015
    public var localWindow = 8
    public init() {}
}

public enum TrackingStatus: String {
    case waiting, initialized = "init", tracked, lost
}

public struct SLAMFrameResult {
    public var index: Int
    public var t: Double
    public var status: TrackingStatus
    public var pose: Pose?
    /// Per detection index: the cone's camera-frame measurement and its landmark.
    public var measurements: [Int: ConeMeasurement]
    public var landmarkOf: [Int: Int]
    public var nInliers: Int
    public var isKeyframe: Bool
    /// Bundle adjustment ran: landmark positions and keyframe poses moved.
    public var mapAdjusted: Bool
}

public struct MapLandmark: Equatable {
    public var id: Int
    public var classID: Int
    public var p: Vec3
    public var sigma: Vec3
    public var nObs: Int
    public var confirmed: Bool
}

public struct KeyframePose: Equatable {
    public var index: Int
    public var t: Double
    public var pose: Pose
}

/// Online cone-only SLAM for the phone: the online half of cone_app/slam.py (tracking, mapping,
/// windowed and periodic global bundle adjustment, duplicate merging), plus a gravity prior on tilt.
/// The offline passes (re-localising every frame against the final map, gap filling) are left to the
/// server. Not thread-safe: call from one queue.
public final class LiveSLAM {
    public let options: SLAMOptions

    private var landmarksByID: [Int: Landmark] = [:]
    private var keyframeIndices: [Int] = []
    private var keyframeTimes: [Int: Double] = [:]
    private var keyframePoses: [Int: Pose] = [:]
    private var keyframeUps: [Int: Vec3] = [:]
    private var observations: [Observation] = []
    private var coobserved: [Int: Set<Int>] = [:]
    private var seenTime: [Int: Double] = [:]
    private var nextID = 0
    private var frameNumber = 0
    private var initialized = false
    private var flat = false
    private var previous: (t: Double, pose: Pose)?
    private var velocity: (w: Vec3, v: Vec3)?
    private var lostCount = 0
    private var meanHeight: Double?
    private var nextGlobal = 15
    private var probation = 0

    private let fuseUntil = 3
    private let baEvery = 3
    private let keyframeSeconds = 0.5
    private let keyframeMetres = 0.5
    private let probationFrames = 3
    private let heightTau = 3.0

    public init(options: SLAMOptions = SLAMOptions()) {
        self.options = options
    }

    public var isInitialized: Bool { initialized }

    public var landmarks: [MapLandmark] {
        landmarksByID.values.sorted { $0.id < $1.id }.map { lm in
            let s = Vec3(lm.covariance[0][0], lm.covariance[1][1], lm.covariance[2][2])
            return MapLandmark(id: lm.id, classID: lm.classID, p: lm.p, sigma: simd_max(s, .zero).squareRoot(),
                               nObs: lm.n, confirmed: lm.n >= 3)
        }
    }

    public var keyframes: [KeyframePose] {
        keyframeIndices.map { KeyframePose(index: $0, t: keyframeTimes[$0]!, pose: keyframePoses[$0]!) }
    }

    public func process(_ frame: SLAMFrame) -> SLAMFrameResult {
        let fi = frameNumber
        frameNumber += 1
        let K = frame.K
        let up = frame.up.map(simd_normalize) ?? Vec3(0, -1, 0)
        let prep = Prepared(frame.detections, width: frame.imageSize.x, height: frame.imageSize.y, kpConf: options.kpConf)
        let visiblePx = 30 * frame.imageSize.x / 1920

        // PnP per prepared row.
        var ms: [Int: ConeMeasurement] = [:]
        for row in 0..<prep.count {
            let d = frame.detections[prep.index[row]]
            let scale = frame.rangeScales?[prep.index[row]] ?? 1
            if let m = ConeSolver.solve(d, K: K, up: up, kpConf: options.kpConf, rangeScale: scale) { ms[row] = m }
        }
        let good = ms.filter { simd_length($0.value.pCam) <= options.maxRange && $0.value.reprojRMS < 6 }
        var result = SLAMFrameResult(index: fi, t: frame.t, status: initialized ? .lost : .waiting, pose: nil,
                                     measurements: Dictionary(uniqueKeysWithValues: ms.map { (prep.index[$0.key], $0.value) }),
                                     landmarkOf: [:], nInliers: 0, isKeyframe: false, mapAdjusted: false)

        var T: Pose
        var assoc: [Int: Int] = [:]           // prepared row -> landmark id
        var quality: TrackResult?
        if !initialized {
            guard let start = initialPose(good, up: up) else { return result }
            T = start
            initialized = true
            result.status = .initialized
            addKeyframe(fi, t: frame.t, pose: T, up: frame.up)
        } else {
            var hypotheses: [Pose] = []
            var prior = PosePrior(up: options.gravitySigma == nil ? nil : frame.up, tiltSigma: options.gravitySigma ?? 0.02)
            if let previous {
                let dt = frame.t - previous.t
                if let velocity, lostCount == 0, dt > 0, dt < 0.5 {
                    hypotheses.append(Pose(R: expSO3(velocity.w * dt) * previous.pose.R, t: previous.pose.t + velocity.v * dt))
                }
                hypotheses.append(previous.pose)
                if let g = frame.up, options.gravitySigma != nil {
                    hypotheses = hypotheses.map { Pose(R: levelled($0.R, up: g), t: $0.t) }
                }
                prior.reference = hypotheses[0]
                prior.dt = min(abs(dt), 5)
                prior.height = hypotheses[0].t.z
                prior.heightSigma = 0.03 + 0.5 * prior.dt
                if prior.up == nil {
                    // Hand-held tilt changes smoothly: soft prior from the motion model.
                    prior.up = hypotheses[0].upInCamera
                    prior.tiltSigma = 0.02 + 0.5 * prior.dt
                }
                prior.meanHeight = flat ? meanHeight : nil
            }
            guard let tracked = track(prep, good, K: K, size: frame.imageSize, hypotheses: hypotheses,
                                      prior: previous == nil ? (prior.up == nil ? nil : prior) : prior,
                                      gravity: frame.up, visiblePx: visiblePx) else {
                lostCount += 1
                return result
            }
            T = tracked.T
            quality = tracked
            if !tracked.consistent {
                probation = probationFrames
            } else if probation > 0, tracked.n >= 4 {
                probation -= 1
            }
            assoc = tracked.assoc
            result.status = .tracked
            result.nInliers = tracked.n
            if let previous, lostCount == 0 {
                let dt = frame.t - previous.t
                if dt > 0 {
                    let w = logSO3(T.R * previous.pose.R.transpose) / dt, v = (T.t - previous.pose.t) / dt
                    velocity = velocity.map { (0.5 * $0.w + 0.5 * w, 0.5 * $0.v + 0.5 * v) } ?? (w, v)
                }
            } else {
                velocity = nil
            }
            lostCount = 0
        }

        // Mapping: fuse associated detections and seed tentative landmarks.
        let reliable = result.status == .initialized ||
            (probation == 0 && quality.map { $0.n >= 4 || ($0.n >= 3 && $0.consistent) } == true)
        if reliable {
            if meanHeight == nil || result.status == .initialized {
                meanHeight = T.t.z
            } else if let previous {
                let alpha = min(1, max(frame.t - previous.t, 0) / heightTau)
                meanHeight = (1 - alpha) * meanHeight! + alpha * T.t.z
            }
        }
        var occupied = Set<Int>()
        let view = MapView(landmarksByID)
        if view.count > 0 {
            occupied = Set(associate(T, prep, view, K, size: frame.imageSize, gate: 1.2).rows)
        }
        if reliable {
            for (row, m) in good.sorted(by: { $0.key < $1.key }) where !prep.truncated[row] {
                let (p, covariance) = measurementWorld(m, T, K)
                guard p.x.isFinite, p.y.isFinite, p.z.isFinite, simd_length(p - T.t) <= options.maxRange * 1.25 else { continue }
                if let id = assoc[row], let lm = landmarksByID[id] {
                    if lm.n < fuseUntil || !lm.ba {
                        // PnP ranges are biased by a few percent, so only young points are fused;
                        // bundle adjustment owns the rest. Huber-like attenuation limits bad PnP fits.
                        let error = simd_length(p - lm.p)
                        guard let inv = inverse3(covariance + Mat3(diagonal: Vec3(repeating: 0.03 * 0.03))) else { continue }
                        let info = inv * min(1, 0.5 / max(error, 1e-9))
                        lm.information += info
                        lm.rhs += info * p
                        if let solved = inverse3(lm.information) {
                            lm.p = solved * lm.rhs
                            if flat { lm.p.z = 0 }
                            lm.covariance = solved + Mat3(diagonal: Vec3(repeating: 0.02 * 0.02))
                        }
                    }
                    lm.n += 1
                    lm.last = fi
                    if lm.classID == 3, m.classID != 3 { lm.classID = m.classID }
                } else {
                    guard !occupied.contains(row), simd_length(m.pCam) <= min(options.maxRange, 25), m.reprojRMS <= 4,
                          let info = inverse3(covariance + Mat3(diagonal: Vec3(repeating: 0.02 * 0.02))) else { continue }
                    let lm = Landmark(id: nextID, classID: m.classID, p: p, covariance: covariance, frame: fi, information: info)
                    landmarksByID[nextID] = lm
                    assoc[row] = nextID
                    nextID += 1
                }
            }
        }
        for id in assoc.values {
            if let lm = landmarksByID[id], lm.last != fi { lm.n += 1; lm.last = fi }
            seenTime[id] = frame.t
        }
        let ids = Set(assoc.values)
        for id in ids { coobserved[id, default: []].formUnion(ids.subtracting([id])) }
        // Stale tentative points disappear (time based, independent of frame rate).
        for (id, lm) in landmarksByID where lm.n < 3 && frame.t - (seenTime[id] ?? frame.t) > 1 {
            removeLandmark(id)
            assoc = assoc.filter { $0.value != id }
        }

        if result.status != .initialized, reliable, let last = keyframeIndices.last, let lastPose = keyframePoses[last] {
            if frame.t - keyframeTimes[last]! >= keyframeSeconds || simd_length(T.t - lastPose.t) > keyframeMetres ||
                rotationAngle(T.R * lastPose.R.transpose) > 0.3 {
                addKeyframe(fi, t: frame.t, pose: T, up: frame.up)
            }
        }
        if keyframeIndices.last == fi {
            result.isKeyframe = true
            for (row, id) in assoc {
                observations.append(Observation(frame: fi, landmark: id, classID: prep.classes[row], pixels: prep.pixels[row],
                                                valid: prep.valid[row], K: K, sigma: prep.sigma[row]))
            }
            if keyframeIndices.count >= 4, keyframeIndices.count % baEvery == 0 {
                let global = keyframeIndices.count >= nextGlobal
                let window = global ? keyframeIndices : Array(keyframeIndices.suffix(options.localWindow))
                bundleAdjust(poses: &keyframePoses, ups: options.gravitySigma == nil ? [:] : keyframeUps,
                             landmarks: landmarksByID, keyframes: window, observations: observations, flat: flat,
                             global: global, tiltSigma: options.gravitySigma ?? 0.02)
                if global {
                    // Geometric spacing bounds the total cost of growing global passes.
                    nextGlobal = keyframeIndices.count * 2
                    mergeLandmarks()
                }
                T = keyframePoses[fi]!
                result.mapAdjusted = true
            }
        }
        previous = (frame.t, T)
        result.pose = T
        result.landmarkOf = Dictionary(uniqueKeysWithValues: assoc.compactMap { row, id in
            landmarksByID[id] == nil ? nil : (prep.index[row], id)
        })
        return result
    }

    // MARK: Initialisation

    /// Level world from gravity, ground height from the cones; needs a spread of at least 3 cones.
    private func initialPose(_ good: [Int: ConeMeasurement], up: Vec3) -> Pose? {
        let points = good.values.map(\.pCam)
        guard points.count >= 3 else { return nil }
        // Horizontal spread: second singular value of the levelled ground positions.
        let L = levelling(up)
        let ground = points.map { L * $0 }
        let mean = ground.reduce(Vec3.zero, +) / Double(ground.count)
        var sxx = 0.0, syy = 0.0, sxy = 0.0
        for g in ground { let d = g - mean; sxx += d.x * d.x; syy += d.y * d.y; sxy += d.x * d.y }
        let smaller = (sxx + syy) / 2 - (((sxx - syy) / 2) * ((sxx - syy) / 2) + sxy * sxy).squareRoot()
        guard smaller.squareRoot() > 0.4 else { return nil }
        let height = -median(points.map { simd_dot(up, $0) })
        flat = options.flatGround && height > 0.2 && height < 5
        return worldTransform(up: up, height: flat ? height : 0)
    }

    /// World x = camera forward projected on the ground, y = left, z = up (geometry.py `_world_transform`).
    private func worldTransform(up: Vec3, height: Double) -> Pose {
        var forward = Vec3(0, 0, 1) - up * up.z
        forward = simd_normalize(forward)
        let left = simd_cross(up, forward)
        return Pose(R: Mat3(rows: [forward, left, up]), t: Vec3(0, 0, height))
    }

    // MARK: Tracking

    private func track(_ prep: Prepared, _ ms: [Int: ConeMeasurement], K: Intrinsics, size: Vec2, hypotheses: [Pose],
                       prior: PosePrior?, gravity: Vec3?, visiblePx: Double) -> TrackResult? {
        let view = MapView(landmarksByID)
        let usable = prep.truncated.filter { !$0 }.count
        var best: TrackResult?
        var seen: [TrackResult] = []

        func consider(_ start: Pose) {
            guard var candidate = localize(start, prep, view, K, size: size, prior: prior, visiblePx: visiblePx) else { return }
            candidate.consistent = isConsistent(candidate.T, prior)
            seen.append(candidate)
            if isAdmissible(candidate, usable: usable), candidate.score > (best?.score ?? -1e9) { best = candidate }
        }

        let expected = max(minInliers, Int(0.7 * Double(min(prep.count, view.count))))
        for start in hypotheses {
            consider(start)
            if let best, best.n >= max(expected, prep.count - 1), best.error < 1 { break }
        }
        if best == nil || best!.n < expected || best!.error > 1.5 {
            let confirmed = MapView(landmarksByID, minimumN: 3)
            var relocView = confirmed.count >= 3 ? confirmed : view
            if let reference = prior?.reference, relocView.count > relocMaxLandmarks {
                // Only the neighbourhood the camera can have reached is searched.
                let distance = relocView.points.map { simd_length($0 - reference.t) }
                let reach = (ms.values.map { simd_length($0.pCam) }.max() ?? 0) + 2 + 1.5 * min(prior!.dt, 20)
                let keep = distance.indices.sorted { distance[$0] < distance[$1] }.prefix(relocMaxLandmarks).filter { distance[$0] < reach }
                relocView = relocView.subset(Array(keep))
            }
            var ups: [Vec3] = []
            if let gravity { ups.append(simd_normalize(gravity)) }
            if let first = hypotheses.first { ups.append(first.upInCamera) }
            if ups.isEmpty { ups.append(Vec3(0, -1, 0)) }
            for start in relocalizationHypotheses(ms, prep, relocView, ups: ups, flat: flat) { consider(start) }
        }
        guard var chosen = best else { return nil }
        if !chosen.consistent {
            // Cone layouts are often periodic (track boundaries): a jump is only trusted when no clearly
            // different pose explains nearly as much.
            for other in seen where other.n >= chosen.n - 2 &&
                (simd_length(other.T.t - chosen.T.t) > 1 || rotationAngle(other.T.R * chosen.T.R.transpose) > 0.35) {
                return nil
            }
        }
        chosen.assoc = Dictionary(uniqueKeysWithValues: zip(chosen.rows, chosen.cols).map { ($0, view.ids[$1]) })
        return chosen
    }

    // MARK: Mapping helpers

    private func measurementWorld(_ m: ConeMeasurement, _ T: Pose, _ K: Intrinsics) -> (Vec3, Mat3) {
        var p = T.R * m.pCam + T.t
        var covariance = T.R * m.covariance * T.R.transpose
        if flat {
            let ray = T.R * m.pCam
            if ray.z < -0.02, T.t.z > 0.2 {
                p = T.t - ray * T.t.z / ray.z
                let distance = simd_length(p - T.t)
                var direction = Vec2(p.x - T.t.x, p.y - T.t.y)
                direction /= max(simd_length(direction), 1e-9)
                let axial = max(0.03, distance * distance * max(1, m.reprojRMS) / (K.fx * T.t.z))
                let lateral = max(0.015, distance * max(1, m.reprojRMS) / K.fx)
                covariance = Mat3(diagonal: Vec3(repeating: lateral * lateral))
                for a in 0..<2 { for b in 0..<2 { covariance[b][a] += axial * axial * direction[a] * direction[b] } }
            }
            p.z = 0
            covariance[2][2] = 0.005 * 0.005
        }
        return (p, covariance)
    }

    private func addKeyframe(_ fi: Int, t: Double, pose: Pose, up: Vec3?) {
        keyframeIndices.append(fi)
        keyframeTimes[fi] = t
        keyframePoses[fi] = pose
        if let up { keyframeUps[fi] = simd_normalize(up) }
    }

    private func removeLandmark(_ id: Int) {
        landmarksByID[id] = nil
        coobserved[id] = nil
        seenTime[id] = nil
    }

    /// Consolidate nearby compatible points unless they were ever seen together (slam.py `_merge_landmarks`).
    @discardableResult
    func mergeLandmarks(distance: Double = 1.5) -> Bool {
        guard landmarksByID.count >= 2 else { return false }
        var remap = Dictionary(uniqueKeysWithValues: landmarksByID.keys.map { ($0, $0) })
        var members = landmarksByID.mapValues { [$0.p] }
        let ids = landmarksByID.keys.sorted { a, b in
            let la = landmarksByID[a]!, lb = landmarksByID[b]!
            return la.first != lb.first ? la.first < lb.first : (la.n != lb.n ? la.n > lb.n : a < b)
        }
        var merged = false
        for i in ids where remap[i] == i {
            let a = landmarksByID[i]!
            for j in ids where i != j && remap[j] == j {
                let b = landmarksByID[j]!
                let gap = simd_length(a.p - b.p)
                // Two physical cones cannot stand closer than about a base width.
                if coobserved[i, default: []].contains(j) && gap > 0.15 { continue }
                guard ConeModel.compatible(a.classID, b.classID), gap <= distance else { continue }
                // Complete-link clustering keeps chains of nearby duplicates from swallowing neighbours.
                guard !members[i]!.contains(where: { pa in members[j]!.contains { simd_length(pa - $0) > distance } }) else { continue }
                members[i]! += members[j]!
                a.p = (a.p * Double(a.n) + b.p * Double(b.n)) / Double(a.n + b.n)
                a.n += b.n
                a.first = min(a.first, b.first)
                a.last = max(a.last, b.last)
                a.information += b.information
                a.rhs = a.information * a.p
                if let inv = inverse3(a.information) { a.covariance = inv + Mat3(diagonal: Vec3(repeating: 0.02 * 0.02)) }
                if a.classID == 3 { a.classID = b.classID }
                remap[j] = i
                let others = coobserved[j, default: []]
                coobserved[i, default: []].formUnion(others)
                for other in others { coobserved[other, default: []].insert(i) }
                seenTime[i] = max(seenTime[i] ?? 0, seenTime[j] ?? 0)
                merged = true
            }
        }
        guard merged else { return false }
        // Observations may still name removed tentative points; those map to themselves.
        func root(_ j: Int) -> Int { var r = j; while let next = remap[r], next != r { r = next }; return r }
        for k in observations.indices { observations[k].landmark = root(observations[k].landmark) }
        for (j, i) in remap where j != i { removeLandmark(j); _ = i }
        for (id, set) in coobserved {
            coobserved[id] = Set(set.compactMap { other in remap[other] == nil ? nil : root(other) }).subtracting([id])
        }
        return true
    }

    /// Forget the map and start over (e.g. after a long loss).
    public func reset() {
        landmarksByID = [:]; keyframeIndices = []; keyframeTimes = [:]; keyframePoses = [:]; keyframeUps = [:]
        observations = []; coobserved = [:]; seenTime = [:]
        initialized = false; flat = false; previous = nil; velocity = nil; lostCount = 0; meanHeight = nil
        nextGlobal = 15; probation = 0
    }
}
