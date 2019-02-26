// swift-tools-version:4.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "NELFramework",
    products: [
        .library(
            name: "NELFramework",
            targets: ["NELFramework"]),
    ],
    targets: [
        .target(
            name: "CNELFramework",
            path: "api/swift/Sources/CNELFramework", 
            sources: [
                "*.h", "*.cpp", 
                "deps/core/*.h", "deps/core/*.cpp",
                "deps/math/*.h", "deps/math/*.cpp"],
            publicHeadersPath: "api/swift/Sources/CNELFramework/include/simulator.h",
            _cSettings: [
                .headerSearchPath("../../../../"),
                .headerSearchPath("../../../../deps"),
                .headerSearchPath("../../../../deps/core"),
                .headerSearchPath("../../../../deps/math"),
            ]),
        .target(
            name: "NELFramework",
            dependencies: ["CNELFramework"],
            path: "api/swift/Sources/NELFramework"),
    ]
)
