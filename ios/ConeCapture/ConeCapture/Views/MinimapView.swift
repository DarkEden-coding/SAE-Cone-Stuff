import SwiftUI

/// Top-down, camera-centred, heading-up minimap of the live cone map.
/// World is z-up with x/y in the ground plane; heading-up means the camera's optical-axis yaw points up.
struct MinimapView: View {
    let snapshot: MinimapSnapshot

    static let side: CGFloat = 160
    private static let minAcross = 5.0      // metres across the square, at the closest zoom
    private static let maxAcross = 40.0
    private static let fovHalfAngle = 35.0 * Double.pi / 180

    var body: some View {
        Canvas { ctx, size in draw(&ctx, size) }
            .frame(width: Self.side, height: Self.side)
            .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
            .overlay(alignment: .topLeading) {
                Text(snapshot.status)
                    .font(.system(size: 10, weight: .semibold, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.85))
                    .padding(6)
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .accessibilityElement()
            .accessibilityLabel("Cone minimap, \(snapshot.cones.count) cones, \(snapshot.status)")
    }

    // MARK: Drawing

    private func draw(_ ctx: inout GraphicsContext, _ size: CGSize) {
        let origin = CGPoint(x: size.width / 2, y: size.height * 0.62)   // camera sits low: more room ahead
        let cam = snapshot.camera ?? .init(x: 0, y: 0, yaw: 0)
        let (sinY, cosY) = (sin(cam.yaw), cos(cam.yaw))

        let farthest = snapshot.cones.map { hypot($0.x - cam.x, $0.y - cam.y) }.max() ?? 0
        let across = min(max(farthest * 2 * 1.15, Self.minAcross), Self.maxAcross)
        let scale = (size.width - 2 * 12) / across          // points per metre

        func screen(_ x: Double, _ y: Double) -> CGPoint {
            let dx = x - cam.x, dy = y - cam.y
            let forward = dx * cosY + dy * sinY
            let right = dx * sinY - dy * cosY
            return CGPoint(x: origin.x + right * scale, y: origin.y - forward * scale)
        }

        if snapshot.trail.count > 1 {
            var path = Path()
            path.move(to: screen(snapshot.trail[0].x, snapshot.trail[0].y))
            for p in snapshot.trail.dropFirst() { path.addLine(to: screen(p.x, p.y)) }
            ctx.stroke(path, with: .color(.white.opacity(0.5)), lineWidth: 1)
        }

        // Unconfirmed first so confirmed cones draw on top.
        for cone in snapshot.cones.sorted(by: { !$0.confirmed && $1.confirmed }) {
            let p = screen(cone.x, cone.y)
            let r: CGFloat = cone.classID == 1 ? 6 : 4
            let rect = CGRect(x: p.x - r, y: p.y - r, width: 2 * r, height: 2 * r)
            let color = Self.color(for: cone.classID)
            if cone.confirmed {
                ctx.fill(Path(ellipseIn: rect), with: .color(color))
            } else {
                ctx.stroke(Path(ellipseIn: rect.insetBy(dx: 0.5, dy: 0.5)), with: .color(color.opacity(0.6)), lineWidth: 1.5)
            }
        }

        if snapshot.camera != nil {
            let wedgeLen = size.width * 0.4
            var wedge = Path()
            wedge.move(to: origin)
            wedge.addLine(to: CGPoint(x: origin.x - wedgeLen * tan(Self.fovHalfAngle), y: origin.y - wedgeLen))
            wedge.addLine(to: CGPoint(x: origin.x + wedgeLen * tan(Self.fovHalfAngle), y: origin.y - wedgeLen))
            wedge.closeSubpath()
            let tint: Color = snapshot.cameraStale ? .gray : .white
            ctx.fill(wedge, with: .color(tint.opacity(snapshot.cameraStale ? 0.1 : 0.15)))

            var tri = Path()
            tri.move(to: CGPoint(x: origin.x, y: origin.y - 7))
            tri.addLine(to: CGPoint(x: origin.x - 5, y: origin.y + 5))
            tri.addLine(to: CGPoint(x: origin.x + 5, y: origin.y + 5))
            tri.closeSubpath()
            ctx.fill(tri, with: .color(tint))
        }

        drawScaleBar(&ctx, size, scale: scale)
    }

    private func drawScaleBar(_ ctx: inout GraphicsContext, _ size: CGSize, scale: Double) {
        let metres = [0.5, 1, 2, 5, 10].last { $0 * scale <= 50 } ?? 0.5
        let length = metres * scale
        let y = size.height - 10
        var bar = Path()
        bar.move(to: CGPoint(x: 8, y: y - 3))
        bar.addLine(to: CGPoint(x: 8, y: y))
        bar.addLine(to: CGPoint(x: 8 + length, y: y))
        bar.addLine(to: CGPoint(x: 8 + length, y: y - 3))
        ctx.stroke(bar, with: .color(.white.opacity(0.8)), lineWidth: 1)
        let label = metres < 1 ? "\(metres) m" : "\(Int(metres)) m"
        ctx.draw(Text(label).font(.system(size: 9, design: .monospaced)).foregroundStyle(.white.opacity(0.8)),
                 at: CGPoint(x: 8 + length + 4, y: y), anchor: .leading)
    }

    static func color(for classID: Int) -> Color {
        switch classID {
        case 0: return .blue
        case 1, 2: return .orange
        case 3: return .gray
        case 4: return .yellow
        default: return .white
        }
    }
}

#Preview {
    let cones: [MinimapSnapshot.Cone] = (0..<8).flatMap { i -> [MinimapSnapshot.Cone] in
        let x = 3.0 + Double(i) * 1.8
        return [.init(id: 2 * i, classID: 0, x: x, y: 1.6 + Double(i) * 0.1, confirmed: i < 6),
                .init(id: 2 * i + 1, classID: 4, x: x, y: -1.6 + Double(i) * 0.1, confirmed: i < 7)]
    } + [.init(id: 100, classID: 1, x: 1.0, y: 0.2, confirmed: true)]
    let trail = (0..<20).map { SIMD2<Double>(Double($0) * -0.15, 0) }
    return MinimapView(snapshot: .init(cones: cones, camera: .init(x: 0, y: 0, yaw: 0), trail: trail, status: "tracked"))
        .padding()
        .background(.gray)
}
