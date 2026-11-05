import XCTest

final class CaptureSmokeTests: XCTestCase {
    func testVideoAndPhotoCapture() throws {
        #if targetEnvironment(simulator)
        throw XCTSkip("ARKit capture requires a physical iPhone.")
        #endif
        continueAfterFailure = false
        let app = XCUIApplication()
        app.launch()

        let springboard = XCUIApplication(bundleIdentifier: "com.apple.springboard")
        let allowCamera = springboard.buttons["Allow"]
        if allowCamera.waitForExistence(timeout: 5) { allowCamera.tap() }

        let intrinsics = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'fx '")).firstMatch
        XCTAssertTrue(intrinsics.waitForExistence(timeout: 20), "Camera should deliver frames and intrinsics.")

        app.buttons["Settings"].tap()
        let automaticUpload = app.switches["Upload automatically after capture"]
        XCTAssertTrue(automaticUpload.waitForExistence(timeout: 5))
        let wasAutomatic = automaticUpload.value as? String == "1"
        if wasAutomatic { automaticUpload.tap() }
        app.buttons["Done"].tap()

        app.buttons["Video"].tap()
        app.buttons["Start recording"].tap()
        let recording = app.staticTexts.matching(NSPredicate(format: "label CONTAINS 'REC 00:02'")).firstMatch
        XCTAssertTrue(recording.waitForExistence(timeout: 10))
        app.buttons["Stop recording"].tap()
        let savedVideo = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'Saved '")).firstMatch
        XCTAssertTrue(savedVideo.waitForExistence(timeout: 15))

        app.buttons["Photo"].tap()
        app.buttons["Take photo"].tap()
        let savedPhoto = app.staticTexts.matching(NSPredicate(format: "label BEGINSWITH 'Photo saved '")).firstMatch
        XCTAssertTrue(savedPhoto.waitForExistence(timeout: 15))

        app.buttons["Recordings"].tap()
        XCTAssertTrue(app.staticTexts["Pending"].firstMatch.waitForExistence(timeout: 5))
        let screenshot = XCTAttachment(screenshot: app.screenshot())
        screenshot.lifetime = .keepAlways
        add(screenshot)
        app.buttons["Done"].tap()

        app.buttons["Video"].tap()

        if wasAutomatic {
            app.buttons["Settings"].tap()
            app.switches["Upload automatically after capture"].tap()
            app.buttons["Done"].tap()
        }
    }
}
