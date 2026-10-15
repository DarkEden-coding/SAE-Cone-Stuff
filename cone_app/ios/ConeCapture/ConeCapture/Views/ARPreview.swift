import SwiftUI
import ARKit
import SceneKit

/// Live camera preview. ARSCNView draws the camera background itself (it uses the session internally for
/// that; this app never reads any pose from it). Our own FrameSink remains the session's delegate.
struct ARPreview: UIViewRepresentable {
    let session: ARSession

    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView(frame: .zero)
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

    func updateUIView(_ uiView: ARSCNView, context: Context) {}
}
