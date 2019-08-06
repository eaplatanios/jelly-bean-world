// swift-tools-version:5.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import Foundation
import PackageDescription

let package = Package(
  name: "JellyBeanWorld",
  products: [
    .library(
      name: "JellyBeanWorld",
      targets: ["JellyBeanWorld"]),
  ],
  dependencies: [
    .package(url: "https://github.com/eaplatanios/swift-rl.git", .branch("master")),
    .package(url: "https://github.com/apple/swift-log.git", from: "1.0.0"),
  ],
  targets: [
    .target(
      name: "CJellyBeanWorld",
      path: ".", 
      sources: [
        "api/swift/Sources/CJellyBeanWorld",
        "jbw/simulator.cpp",
        "jbw/status.cpp"],
      publicHeadersPath: "api/swift/Sources/CJellyBeanWorld",
      cxxSettings: [
        .headerSearchPath("jbw"),
        .headerSearchPath("jbw/deps"),
        .unsafeFlags([
          "-std=c++11", "-Wall", "-Wpedantic", "-Ofast", "-DNDEBUG", 
          "-fno-stack-protector", "-mtune=native", "-march=native",
        ]),
      ]),
    .target(
      name: "JellyBeanWorld",
      dependencies: ["CJellyBeanWorld", "ReinforcementLearning"],
      path: "api/swift/Sources/JellyBeanWorld"),
    .target(
      name: "JellyBeanWorldExperiments",
      dependencies: ["JellyBeanWorld", "Logging"],
      path: "api/swift/Sources/JellyBeanWorldExperiments"),
  ]
)
