import SwiftUI
import UIKit

/// Receives the system's background-URLSession wake-up so the completion handler can be called
/// once `Uploader.urlSessionDidFinishEvents` fires.
final class AppDelegate: NSObject, UIApplicationDelegate {
    static var backgroundCompletionHandler: (() -> Void)?

    func application(_ application: UIApplication,
                     handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        Self.backgroundCompletionHandler = completionHandler
    }
}

/// Owns the long-lived objects and wires them together.
@MainActor
final class AppEnvironment: ObservableObject {
    let settings: AppSettings
    let store: RecordingStore
    let uploader: Uploader
    let capture: CaptureSession

    init() {
        let settings = AppSettings()
        let store = RecordingStore()
        let uploader = Uploader(store: store, settings: settings)
        let capture = CaptureSession()
        capture.configure(settings: settings, store: store, uploader: uploader)
        self.settings = settings
        self.store = store
        self.uploader = uploader
        self.capture = capture
        uploader.reconcile()
    }
}

@main
struct ConeCaptureApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    // VERIFY: @StateObject with a @MainActor class initialised in the App struct (fine on Xcode 15+/iOS 17 SDK).
    @StateObject private var env = AppEnvironment()

    var body: some Scene {
        WindowGroup {
            CaptureView()
                .environmentObject(env.settings)
                .environmentObject(env.store)
                .environmentObject(env.uploader)
                .environmentObject(env.capture)
                .preferredColorScheme(.dark)
        }
    }
}
