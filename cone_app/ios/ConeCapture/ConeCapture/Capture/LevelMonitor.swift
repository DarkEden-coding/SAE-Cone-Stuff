import Foundation
import CoreMotion

/// Level indicator from CoreMotion gravity ONLY. Never uploaded, never ARKit pose.
@MainActor
final class LevelMonitor: ObservableObject {
    /// Horizon tilt in degrees (0 = level) for a landscape-held phone.
    @Published var rollDegrees: Double = 0
    /// Camera elevation in degrees (0 = horizontal, negative = looking down).
    @Published var pitchDegrees: Double = 0

    private let motion = CMMotionManager()

    func start() {
        guard motion.isDeviceMotionAvailable, !motion.isDeviceMotionActive else { return }
        motion.deviceMotionUpdateInterval = 1.0 / 20.0
        motion.startDeviceMotionUpdates(to: .main) { [weak self] data, _ in
            guard let g = data?.gravity else { return }
            // In landscape gravity lies mostly along device x (+/-). Tilt = rotation about the view axis.
            // VERIFY: sign of roll on device (flip if the bubble moves the wrong way).
            let roll = g.x < 0 ? atan2(g.y, -g.x) : atan2(-g.y, g.x)
            let pitch = asin(max(-1.0, min(1.0, g.z)))   // screen-up flat => g.z = -1 => camera looks down => -90
            MainActor.assumeIsolated {
                self?.rollDegrees = roll * 180 / .pi
                self?.pitchDegrees = pitch * 180 / .pi
            }
        }
    }

    func stop() { motion.stopDeviceMotionUpdates() }
}
