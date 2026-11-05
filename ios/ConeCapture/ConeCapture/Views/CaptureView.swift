import SwiftUI

struct CaptureView: View {
    @EnvironmentObject private var settings: AppSettings
    @EnvironmentObject private var store: RecordingStore
    @EnvironmentObject private var capture: CaptureSession
    @EnvironmentObject private var level: LevelMonitor
    @Environment(\.scenePhase) private var scenePhase

    @State private var showSettings = false
    @State private var showRecordings = false

    var body: some View {
        GeometryReader { geometry in
            let isPortrait = geometry.size.height > geometry.size.width
            ZStack {
                Color.black.ignoresSafeArea()
                ARPreview(session: capture.arSession) { orientation, size in
                    level.interfaceOrientation = orientation
                    capture.setViewport(size, orientation: orientation)
                }
                .ignoresSafeArea()

                if capture.mode == .live, let live = capture.live {
                    LiveOverlayView(cones: live.cones)
                        .ignoresSafeArea()
                    VStack(alignment: .trailing, spacing: 8) {
                        MinimapView(snapshot: live.minimap)
                            .frame(width: 160, height: 160)
                            .onTapGesture(count: 2) { capture.resetLiveMap() }
                            .accessibilityHint("Double tap to reset the live map")
                        Button {
                            capture.resetLiveMap()
                        } label: {
                            Label("Reset map", systemImage: "arrow.counterclockwise")
                                .font(.footnote.weight(.semibold))
                                .padding(.horizontal, 12).padding(.vertical, 8)
                                .background(.black.opacity(0.55), in: Capsule())
                                .foregroundStyle(.white)
                        }
                        .accessibilityLabel("Reset live map")
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topTrailing)
                    .padding(12)
                }

                hud
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(12)

                LevelIndicator(roll: level.rollDegrees)

                controls(isPortrait: isPortrait)
                    .frame(maxWidth: .infinity, maxHeight: .infinity,
                           alignment: isPortrait ? .bottom : .trailing)
                    .padding(isPortrait ? .bottom : .trailing, 16)

                if let toast = capture.toast {
                    VStack {
                        Spacer()
                        Text(toast)
                            .font(.footnote.weight(.medium))
                            .padding(.horizontal, 12).padding(.vertical, 8)
                            .background(.ultraThinMaterial, in: Capsule())
                            .padding(.bottom, isPortrait ? 180 : 12)
                    }
                    .transition(.opacity)
                    .task(id: toast) {
                        try? await Task.sleep(nanoseconds: 2_500_000_000)
                        capture.toast = nil
                    }
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
            if capture.mode == .live {
                if let live = capture.live {
                    Text(String(format: "LIVE %@  %.1f fps  det %.0f ms  slam %.0f ms",
                                live.status.rawValue, live.fps, live.detMs, live.slamMs))
                        .foregroundStyle(live.status == .tracked || live.status == .initialized ? .green : .orange)
                    Text("cones \(live.nLandmarks)  inliers \(live.nInliers)  skipped \(live.skipped)  \(streamLabel)")
                } else {
                    Text(capture.liveStarting ? "LIVE loading model\u{2026}" : "LIVE starting\u{2026}")
                        .foregroundStyle(.orange)
                }
            }
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

    private var streamLabel: String {
        guard settings.liveStream else { return "stream off" }
        guard let streamer = capture.livePipeline?.streamer else { return "no server" }
        return "stream \(streamer.state)"
    }

    private static func clock(_ s: Double) -> String {
        let t = Int(s)
        return String(format: "%02d:%02d", t / 60, t % 60)
    }

    // MARK: Controls

    private func controls(isPortrait: Bool) -> some View {
        Group {
            if isPortrait {
                HStack(spacing: 32) {
                    settingsButton
                    VStack(spacing: 12) {
                        modePicker
                        shutter
                    }
                    recordingsButton
                }
            } else {
                VStack(spacing: 18) {
                    settingsButton
                    modePicker
                    shutter
                    recordingsButton
                }
            }
        }
        .padding(12)
        .background(.black.opacity(0.35), in: RoundedRectangle(cornerRadius: 16))
    }

    private var settingsButton: some View {
        Button { showSettings = true } label: { icon("gearshape.fill") }
            .accessibilityLabel("Settings")
            .disabled(capture.isRecording || capture.isBusy)
    }

    private var modePicker: some View {
        Picker("Mode", selection: $capture.mode) {
            ForEach(CaptureSession.Mode.allCases) { Text($0.rawValue).tag($0) }
        }
        .pickerStyle(.segmented)
        .frame(width: 190)
        .disabled(capture.isRecording || capture.isBusy)
    }

    private var recordingsButton: some View {
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
        .accessibilityLabel("Recordings")
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
            case .video, .live: capture.toggleRecording()
            }
        } label: {
            ZStack {
                Circle().stroke(.white, lineWidth: 4).frame(width: 72, height: 72)
                if capture.mode != .photo {
                    RoundedRectangle(cornerRadius: capture.isRecording ? 6 : 28)
                        .fill(.red)
                        .frame(width: capture.isRecording ? 30 : 56, height: capture.isRecording ? 30 : 56)
                } else {
                    Circle().fill(.white).frame(width: 58, height: 58)
                }
                if capture.isBusy { ProgressView().tint(.black) }
            }
        }
        .accessibilityLabel(capture.mode == .photo ? "Take photo" : (capture.isRecording ? "Stop recording" : "Start recording"))
        .disabled(capture.isBusy)
        .opacity(capture.isBusy ? 0.6 : 1)
    }
}

/// Simple level: horizon line rotated by roll; turns green within 1 degree.
struct LevelIndicator: View {
    var roll: Double

    var body: some View {
        let level = abs(roll) < 1.0
        ZStack {
            Rectangle()
                .fill(level ? Color.green : Color.white.opacity(0.8))
                .frame(width: 140, height: 2)
                .rotationEffect(.degrees(roll))
            Rectangle()
                .fill(Color.white.opacity(0.5))
                .frame(width: 24, height: 1)
        }
        .allowsHitTesting(false)
    }
}
