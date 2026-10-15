import SwiftUI

struct CaptureView: View {
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var capture: CaptureSession
    @StateObject private var level = LevelMonitor()
    @Environment(\.scenePhase) private var scenePhase

    @State private var showSettings = false
    @State private var showRecordings = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            ARPreview(session: capture.arSession).ignoresSafeArea()

            HStack(alignment: .top) {
                hud
                Spacer()
            }
            .padding(12)

            LevelIndicator(roll: level.rollDegrees, pitch: level.pitchDegrees)

            HStack {
                Spacer()
                controls
            }
            .padding(.trailing, 16)

            if let toast = capture.toast {
                VStack {
                    Spacer()
                    Text(toast)
                        .font(.footnote.weight(.medium))
                        .padding(.horizontal, 12).padding(.vertical, 8)
                        .background(.ultraThinMaterial, in: Capsule())
                        .padding(.bottom, 12)
                }
                .transition(.opacity)
                .task(id: toast) {
                    try? await Task.sleep(nanoseconds: 2_500_000_000)
                    capture.toast = nil
                }
            }
        }
        .persistentSystemOverlays(.hidden)
        .statusBarHidden(true)
        .onAppear {
            capture.start()
            level.start()
            UIApplication.shared.isIdleTimerDisabled = true
        }
        .onDisappear {
            level.stop()
            UIApplication.shared.isIdleTimerDisabled = false
        }
        .onChange(of: settings.sessionConfigKey) { _, _ in
            if !capture.isRecording { capture.applyConfiguration() }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase != .active { capture.stopRecordingIfNeeded() }
            if phase == .active { capture.start() }
        }
        .sheet(isPresented: $showSettings) {
            NavigationStack { SettingsView() }
        }
        .sheet(isPresented: $showRecordings) {
            NavigationStack { RecordingsView() }
        }
        .alert("Capture", isPresented: Binding(
            get: { capture.errorMessage != nil },
            set: { if !$0 { capture.errorMessage = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(capture.errorMessage ?? "")
        }
    }

    // MARK: HUD

    private var hud: some View {
        VStack(alignment: .leading, spacing: 2) {
            if let h = capture.hud {
                Text("\(Int(h.resolution.width))\u{00D7}\(Int(h.resolution.height))  \(String(format: "%.1f", h.fps)) fps")
                Text(String(format: "fx %.1f  fy %.1f", h.intrinsics.fx, h.intrinsics.fy))
                Text(String(format: "cx %.1f  cy %.1f", h.intrinsics.cx, h.intrinsics.cy))
            } else {
                Text("Starting camera\u{2026}")
            }
            Text("Focus: \(capture.focusLabel)")
            Text(String(format: "Level: roll %+.1f\u{00B0}  pitch %+.1f\u{00B0}", level.rollDegrees, level.pitchDegrees))
            if capture.isRecording, let h = capture.hud {
                Text("\u{25CF} REC \(Self.clock(h.recordElapsed))  enc \(h.encoded)  drop \(h.dropped)")
                    .foregroundStyle(.red)
            }
        }
        .font(.system(size: 12, design: .monospaced))
        .foregroundStyle(.white)
        .padding(8)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 8))
    }

    private static func clock(_ s: Double) -> String {
        let t = Int(s)
        return String(format: "%02d:%02d", t / 60, t % 60)
    }

    // MARK: Controls

    private var controls: some View {
        VStack(spacing: 18) {
            Button { showSettings = true } label: { icon("gearshape.fill") }
                .disabled(capture.isRecording)

            Picker("Mode", selection: $capture.mode) {
                ForEach(CaptureSession.Mode.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            .frame(width: 130)
            .disabled(capture.isRecording || capture.isBusy)

            shutter

            Button { showRecordings = true } label: {
                ZStack(alignment: .topTrailing) {
                    icon("tray.full.fill")
                    if store.pendingCount > 0 {
                        Text("\(store.pendingCount)")
                            .font(.caption2.bold())
                            .padding(4)
                            .background(Color.orange, in: Circle())
                            .offset(x: 6, y: -6)
                    }
                }
            }
        }
        .padding(10)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 16))
    }

    private func icon(_ name: String) -> some View {
        Image(systemName: name)
            .font(.title2)
            .foregroundStyle(.white)
            .frame(width: 44, height: 44)
    }

    @ViewBuilder
    private var shutter: some View {
        Button {
            switch capture.mode {
            case .photo: capture.takePhoto()
            case .video: capture.toggleRecording()
            }
        } label: {
            ZStack {
                Circle().stroke(.white, lineWidth: 4).frame(width: 72, height: 72)
                if capture.mode == .video {
                    RoundedRectangle(cornerRadius: capture.isRecording ? 6 : 28)
                        .fill(.red)
                        .frame(width: capture.isRecording ? 30 : 56, height: capture.isRecording ? 30 : 56)
                } else {
                    Circle().fill(.white).frame(width: 58, height: 58)
                }
                if capture.isBusy { ProgressView().tint(.black) }
            }
        }
        .disabled(capture.isBusy)
        .opacity(capture.isBusy ? 0.6 : 1)
    }
}

/// Simple level: horizon line rotated by roll; turns green within 1 degree.
struct LevelIndicator: View {
    var roll: Double
    var pitch: Double

    var body: some View {
        let level = abs(roll) < 1.0
        ZStack {
            Rectangle()
                .fill(level ? Color.green : Color.white.opacity(0.8))
                .frame(width: 140, height: 2)
                .rotationEffect(.degrees(-roll))   // VERIFY: sign; flip together with LevelMonitor if needed
            Rectangle()
                .fill(Color.white.opacity(0.5))
                .frame(width: 24, height: 1)
        }
        .allowsHitTesting(false)
    }
}
