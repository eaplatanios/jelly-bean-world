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
    .package(url: "https://github.com/apple/swift-package-manager.git", from: "0.5.0"),
  ],
  targets: [
    .target(
      name: "CJellyBeanWorld",
      path: "api/swift/Sources/CJellyBeanWorld",
      sources: [
        "simulator.cpp",
        "../../../../jbw/simulator.cpp"],
      cxxSettings: [
        .headerSearchPath("../../../../jbw"),
        .headerSearchPath("../../../../jbw/deps"),
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
      dependencies: ["JellyBeanWorld", "Logging", "SwiftPM"],
      path: "api/swift/Sources/JellyBeanWorldExperiments"),
  ]
)
