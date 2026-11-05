import XCTest
import CoreMotion
import UIKit
@testable import ConeCapture

final class HorizonFilterTests: XCTestCase {
    func testLevelInEveryInterfaceOrientation() throws {
        let samples: [(UIInterfaceOrientation, CMAcceleration)] = [
            (.portrait, .init(x: 0, y: -1, z: 0)),
            (.portraitUpsideDown, .init(x: 0, y: 1, z: 0)),
            (.landscapeLeft, .init(x: 1, y: 0, z: 0)),
            (.landscapeRight, .init(x: -1, y: 0, z: 0)),
        ]
        for (orientation, gravity) in samples {
            var filter = HorizonFilter()
            let angle = try XCTUnwrap(filter.update(gravity: gravity, orientation: orientation, timestamp: 0))
            XCTAssertEqual(angle, 0, accuracy: 0.001)
        }
    }

    func testHorizonCountersClockwiseTiltAtOneToOneScale() throws {
        let radians = 30.0 * .pi / 180
        let x = sin(radians)
        let y = -cos(radians)
        let samples: [(UIInterfaceOrientation, CMAcceleration)] = [
            (.portrait, .init(x: x, y: y, z: 0)),
            (.portraitUpsideDown, .init(x: -x, y: -y, z: 0)),
            (.landscapeLeft, .init(x: -y, y: x, z: 0)),
            (.landscapeRight, .init(x: y, y: -x, z: 0)),
        ]
        for (orientation, gravity) in samples {
            var filter = HorizonFilter()
            let angle = try XCTUnwrap(filter.update(gravity: gravity, orientation: orientation, timestamp: 0))
            XCTAssertEqual(angle, -30, accuracy: 0.001)
        }
    }

    func testSuddenTiltIsSmoothed() throws {
        var filter = HorizonFilter()
        _ = filter.update(gravity: .init(x: 0, y: -1, z: 0), orientation: .portrait, timestamp: 0)
        let angle = try XCTUnwrap(filter.update(gravity: .init(x: 0.5, y: -sqrt(0.75), z: 0),
                                               orientation: .portrait, timestamp: 0.05))
        XCTAssertLessThan(angle, 0)
        XCTAssertGreaterThan(angle, -10)
    }

    func testFaceUpReadingsDoNotSpinTheHorizon() {
        var filter = HorizonFilter()
        XCTAssertNil(filter.update(gravity: .init(x: 0.01, y: -0.01, z: -0.9999),
                                   orientation: .portrait, timestamp: 0))
    }

    func testCrossingVerticalUsesShortestLineRotation() throws {
        var filter = HorizonFilter()
        func gravity(_ degrees: Double) -> CMAcceleration {
            let radians = degrees * .pi / 180
            return .init(x: -sin(radians), y: -cos(radians), z: 0)
        }
        _ = filter.update(gravity: gravity(89), orientation: .portrait, timestamp: 0)
        let angle = try XCTUnwrap(filter.update(gravity: gravity(-89), orientation: .portrait, timestamp: 0.05))
        XCTAssertGreaterThan(abs(angle), 88)
    }

    func testOrientationChangeResetsOldCoordinateSystem() throws {
        var filter = HorizonFilter()
        _ = filter.update(gravity: .init(x: 0, y: -1, z: 0), orientation: .portrait, timestamp: 0)
        let angle = try XCTUnwrap(filter.update(gravity: .init(x: 1, y: 0, z: 0),
                                               orientation: .landscapeLeft, timestamp: 0.05))
        XCTAssertEqual(angle, 0, accuracy: 0.001)
    }
}
