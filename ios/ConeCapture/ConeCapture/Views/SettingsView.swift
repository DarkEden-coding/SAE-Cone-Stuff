import SwiftUI

struct SettingsView: View {
    @EnvironmentObject private var settings: AppSettings
    @Environment(\.dismiss) private var dismiss

    @State private var testing = false
    @State private var testResult: String?
    @State private var testOK = false

    var body: some View {
        Form {
            Section {
                TextField("https://machine.tailnet.ts.net:8444", text: $settings.baseURL)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                SecureField("Bearer token (optional)", text: $settings.token)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                Button {
                    Task { await testConnection() }
                } label: {
                    HStack {
                        Text("Test connection")
                        if testing { Spacer(); ProgressView() }
                    }
                }
                .disabled(testing)
                if let testResult {
                    Text(testResult).font(.footnote).foregroundStyle(testOK ? .green : .red)
                }
            } header: {
                Text("Server")
            } footer: {
                Text("The token is stored in the Keychain and sent as Authorization: Bearer.")
            }

            Section {
                Picker("Video format", selection: $settings.formatID) {
                    ForEach(VideoFormats.supported, id: \.self) { f in
                        Text(VideoFormats.label(f) + (isRecommended4K(f) ? " \u{2605}" : ""))
                            .tag(VideoFormats.id(f))
                    }
                }
                .pickerStyle(.navigationLink)
                Toggle("Photo: max still resolution", isOn: $settings.photoMaxResolution)
                Toggle("Autofocus", isOn: $settings.autofocus)
            } header: {
                Text("Camera")
            } footer: {
                Text("\u{2605} = ARKit recommended 4K format. Autofocus off locks focus so intrinsics stay stable. Changing these restarts the camera. Photo mode uses ARKit's high-resolution still format when enabled.")
            }

            Section("Capture") {
                TextField("Capture name (optional)", text: $settings.captureName)
                Toggle("Upload automatically after capture", isOn: $settings.autoUpload)
                Toggle("Delete media after confirmed upload", isOn: $settings.deleteAfterUpload)
            }

            Section("About") {
                LabeledContent("Version",
                               value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")
                LabeledContent("Schema", value: CaptureManifest.schemaID)
            }
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
        }
    }

    private func isRecommended4K(_ f: ARFormat) -> Bool {
        guard let r = VideoFormats.recommended4K else { return false }
        return VideoFormats.id(r) == VideoFormats.id(f)
    }

    private func testConnection() async {
        testing = true
        testResult = nil
        defer { testing = false }
        do {
            let info = try await ServerAPI.health(url: settings.endpoint(path: "/api/health"),
                                                  token: settings.trimmedToken)
            testOK = info.ok
            var parts = ["OK"]
            if let m = info.model { parts.append(m) }
            if let d = info.device { parts.append(d) }
            if let v = info.version { parts.append("v\(v)") }
            if info.auth == true { parts.append(settings.trimmedToken == nil ? "auth required (no token set!)" : "auth on") }
            testResult = parts.joined(separator: " \u{00B7} ")
        } catch {
            testOK = false
            testResult = error.localizedDescription
        }
    }
}

import ARKit
private typealias ARFormat = ARConfiguration.VideoFormat
