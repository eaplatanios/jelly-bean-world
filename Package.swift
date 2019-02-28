// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
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
            path: ".", 
            sources: [
                "./api/swift/Sources/CNELFramework",
                "./nel/simulator.cpp"],
            publicHeadersPath: "./api/swift/Sources/CNELFramework",
            cxxSettings: [
                .headerSearchPath("."),
                .headerSearchPath("./deps"),
                .unsafeFlags([
                    "-std=c++11", "-Wall", "-Wpedantic", "-Ofast", "-DNDEBUG", 
                    "-fno-stack-protector", "-mtune=native", "-march=native",
               ]),
            ]),
        .target(
            name: "NELFramework",
            dependencies: ["CNELFramework"],
            path: "api/swift/Sources/NELFramework"),
        .target(
            name: "NELExperiments",
            dependencies: ["NELFramework"],
            path: "api/swift/Sources/NELExperiments"),
    ]
)
