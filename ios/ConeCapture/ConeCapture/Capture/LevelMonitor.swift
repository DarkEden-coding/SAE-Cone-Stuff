import Foundation
import CoreMotion
import UIKit

/// Projects gravity into the displayed UI, then filters the shortest arc of an undirected line.
/// The resulting angle is used directly by SwiftUI (positive angles are clockwise on screen).
struct HorizonFilter {
    private var angle: Double?
    private var timestamp: TimeInterval?
    private var orientation: UIInterfaceOrientation = .unknown

    mutating func update(gravity g: CMAcceleration, orientation newOrientation: UIInterfaceOrientation,
                         timestamp now: TimeInterval) -> Double? {
        let x: Double
        let y: Double
        switch newOrientation {
        case .portrait: (x, y) = (g.x, g.y)
        case .portraitUpsideDown: (x, y) = (-g.x, -g.y)
        case .landscapeLeft: (x, y) = (g.y, -g.x)
        case .landscapeRight: (x, y) = (-g.y, g.x)
        default: return nil
        }
        // Near face-up/down, gravity has no reliable direction in the screen plane.
        guard hypot(x, y) >= 0.15 else { return nil }
        let target = Self.normalized(atan2(-x, -y) * 180 / .pi)
        let result: Double
        if let angle, let timestamp, orientation == newOrientation {
            let alpha = 1 - exp(-max(0, now - timestamp) / 0.2)
            result = Self.normalized(angle + alpha * Self.normalized(target - angle))
        } else {
            result = target
        }
        angle = result
        timestamp = now
        orientation = newOrientation
        return result
    }

    private static func normalized(_ degrees: Double) -> Double {
        var angle = degrees.truncatingRemainder(dividingBy: 180)
        if angle > 90 { angle -= 180 }
        if angle < -90 { angle += 180 }
        return angle
    }
}

/// Recent CoreMotion gravity samples (device frame, unit g), thread-safe. Live mode reads gravity at each
/// camera frame's timestamp (both clocks are system uptime). Never uploaded.
final class GravityHistory: @unchecked Sendable {
    private let lock = NSLock()
    private var samples: [(t: TimeInterval, g: SIMD3<Double>)] = []

    func append(t: TimeInterval, g: SIMD3<Double>) {
        lock.lock(); defer { lock.unlock() }
        samples.append((t, g))
        if samples.count > 200 { samples.removeFirst(samples.count - 200) }
    }

    /// Linearly interpolated gravity at `t`; nil without a sample within 0.1 s.
    func gravity(at t: TimeInterval) -> SIMD3<Double>? {
        lock.lock(); defer { lock.unlock() }
        guard let last = samples.last else { return nil }
        if t >= last.t { return t - last.t < 0.1 ? last.g : nil }
        guard let i = samples.lastIndex(where: { $0.t <= t }) else {
            return samples[0].t - t < 0.1 ? samples[0].g : nil
        }
        let a = samples[i], b = samples[i + 1]
        let f = (t - a.t) / max(b.t - a.t, 1e-6)
        return a.g + (b.g - a.g) * f
    }
}

/// Level indicator from CoreMotion gravity ONLY. Never uploaded, never ARKit pose.
@MainActor
final class LevelMonitor: ObservableObject {
    /// Screen-space horizon tilt in degrees (0 = level).
    @Published var rollDegrees: Double = 0
    /// Camera elevation in degrees (0 = horizontal, negative = looking down).
    @Published var pitchDegrees: Double = 0
    /// Independent of the UI and its rotation lock. Sampled once at photo/video start.
    private(set) var captureRotation: MediaRotation = .clockwise90
    /// 100 Hz gravity for the live SLAM's tilt prior.
    nonisolated let history = GravityHistory()

    private let motion = CMMotionManager()
    private let motionQueue: OperationQueue = {
        let q = OperationQueue()
        q.name = "cone.capture.motion"
        q.maxConcurrentOperationCount = 1
        q.qualityOfService = .userInitiated
        return q
    }()
    var interfaceOrientation: UIInterfaceOrientation = .portrait
    private var filter = HorizonFilter()
    private var lastUI: TimeInterval = 0

    func start() {
        guard motion.isDeviceMotionAvailable, !motion.isDeviceMotionActive else { return }
        filter = HorizonFilter()
        motion.deviceMotionUpdateInterval = 1.0 / 100.0
        let history = history
        motion.startDeviceMotionUpdates(to: motionQueue) { [weak self] data, _ in
            guard let data else { return }
            let g = data.gravity
            history.append(t: data.timestamp, g: SIMD3(g.x, g.y, g.z))
            DispatchQueue.main.async { MainActor.assumeIsolated {
                guard let self, data.timestamp - self.lastUI >= 0.05 else { return }
                self.lastUI = data.timestamp
                self.captureRotation = .fromGravity(x: g.x, y: g.y, fallback: self.captureRotation)
                if let roll = self.filter.update(gravity: g, orientation: self.interfaceOrientation, timestamp: data.timestamp) {
                    self.rollDegrees = roll
                }
                self.pitchDegrees = asin(max(-1.0, min(1.0, g.z))) * 180 / .pi
            } }
        }
    }

    func stop() { motion.stopDeviceMotionUpdates() }
}
