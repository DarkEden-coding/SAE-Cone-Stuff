import SwiftUI

/// Live detections drawn over the camera preview (view coordinates from LivePipeline).
struct LiveOverlayView: View {
    var cones: [LiveDisplay.Cone]

    static func color(_ classID: Int) -> Color {
        switch classID {
        case 0: return Color(red: 0.2, green: 0.45, blue: 1)
        case 1: return Color(red: 1, green: 0.35, blue: 0.1)
        case 2: return .orange
        case 4: return .yellow
        default: return .gray
        }
    }

    var body: some View {
        Canvas { context, _ in
            for cone in cones {
                let color = Self.color(cone.classID)
                context.stroke(Path(cone.box), with: .color(color.opacity(cone.landmarkID == nil ? 0.6 : 1)),
                               lineWidth: cone.landmarkID == nil ? 1 : 2)
                for point in cone.keypoints.compactMap({ $0 }) {
                    context.fill(Path(ellipseIn: CGRect(x: point.x - 2, y: point.y - 2, width: 4, height: 4)), with: .color(.white))
                }
                var label = cone.rangeM.map { String(format: "%.1fm", $0) } ?? ""
                if let id = cone.landmarkID { label += " #\(id)" }
                if !label.isEmpty {
                    context.draw(Text(label).font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundColor(color),
                                 at: CGPoint(x: cone.box.midX, y: cone.box.minY - 7))
                }
            }
        }
        .allowsHitTesting(false)
    }
}
