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

/// Level indicator from CoreMotion gravity ONLY. Never uploaded, never ARKit pose.
@MainActor
final class LevelMonitor: ObservableObject {
    /// Screen-space horizon tilt in degrees (0 = level).
    @Published var rollDegrees: Double = 0
    /// Camera elevation in degrees (0 = horizontal, negative = looking down).
    @Published var pitchDegrees: Double = 0
    /// Independent of the UI and its rotation lock. Sampled once at photo/video start.
    private(set) var captureRotation: MediaRotation = .clockwise90

    private let motion = CMMotionManager()
    var interfaceOrientation: UIInterfaceOrientation = .portrait
    private var filter = HorizonFilter()

    func start() {
        guard motion.isDeviceMotionAvailable, !motion.isDeviceMotionActive else { return }
        filter = HorizonFilter()
        motion.deviceMotionUpdateInterval = 1.0 / 20.0
        motion.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
            guard let data else { return }
            let g = data.gravity
            MainActor.assumeIsolated {
                guard let self else { return }
                self.captureRotation = .fromGravity(x: g.x, y: g.y, fallback: self.captureRotation)
                if let roll = self.filter.update(gravity: g, orientation: self.interfaceOrientation, timestamp: data.timestamp) {
                    self.rollDegrees = roll
                }
                self.pitchDegrees = asin(max(-1.0, min(1.0, g.z))) * 180 / .pi
            }
        }
    }

    func stop() { motion.stopDeviceMotionUpdates() }
}
