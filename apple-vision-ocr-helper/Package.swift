// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "apple-vision-ocr-helper",
    platforms: [.macOS(.v13)],
    targets: [
        .executableTarget(
            name: "apple-vision-ocr-helper",
            path: "Sources/main"
        )
    ]
)
