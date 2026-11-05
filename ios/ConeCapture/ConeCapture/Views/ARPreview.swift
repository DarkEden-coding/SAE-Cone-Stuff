import SwiftUI
import ARKit
import SceneKit

/// Live camera preview. ARSCNView draws the camera background itself (it uses the session internally for
/// that; this app never reads any pose from it). Our own FrameSink remains the session's delegate.
struct ARPreview: UIViewRepresentable {
    let session: ARSession
    var onOrientationChange: (UIInterfaceOrientation, CGSize) -> Void

    func makeUIView(context: Context) -> OrientedARView {
        let v = OrientedARView(frame: .zero)
        v.onOrientationChange = onOrientationChange
        v.session = session   // VERIFY: assigning a pre-made ARSession to ARSCNView keeps session.delegate intact.
        v.scene = SCNScene()
        v.automaticallyUpdatesLighting = false
        v.autoenablesDefaultLighting = false
        v.rendersCameraGrain = false
        v.rendersMotionBlur = false
        v.antialiasingMode = .none
        v.preferredFramesPerSecond = 30
        v.contentMode = .scaleAspectFill
        return v
    }

    func updateUIView(_ uiView: OrientedARView, context: Context) {
        uiView.onOrientationChange = onOrientationChange
    }
}

/// Use the actual UI orientation, including when the phone's rotation lock is enabled.
final class OrientedARView: ARSCNView {
    /// Interface orientation and this view's size (the live overlay's display-transform viewport).
    var onOrientationChange: ((UIInterfaceOrientation, CGSize) -> Void)?
    private var lastOrientation: UIInterfaceOrientation = .unknown
    private var lastSize: CGSize = .zero

    override func layoutSubviews() {
        super.layoutSubviews()
        guard let orientation = window?.windowScene?.interfaceOrientation, orientation != .unknown,
              orientation != lastOrientation || bounds.size != lastSize else { return }
        lastOrientation = orientation
        lastSize = bounds.size
        let size = bounds.size
        // Avoid publishing SwiftUI state during its layout pass.
        DispatchQueue.main.async { [weak self] in self?.onOrientationChange?(orientation, size) }
    }
}
