// swift-tools-version:5.9
import PackageDescription

// Pure-Swift cone geometry and live cone-only SLAM. No UIKit/ARKit, so it builds and tests on macOS:
//   cd ios/ConeCapture/ConeSLAM && swift test
let package = Package(
    name: "ConeSLAM",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "ConeSLAM", targets: ["ConeSLAM"])],
    targets: [
        .target(
            name: "ConeSLAM",
            // Numeric code on small arrays is 10-50x slower unoptimised; keep Debug app builds usable.
            swiftSettings: [.unsafeFlags(["-O"], .when(configuration: .debug))]),
        .testTarget(name: "ConeSLAMTests", dependencies: ["ConeSLAM"], resources: [.copy("Fixtures")]),
    ]
)
