// Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import Foundation
import Python

fileprivate let mpl = Python.import("matplotlib")
fileprivate let plt = Python.import("matplotlib.pyplot")
fileprivate let sns = Python.import("seaborn")

fileprivate let redPalette = sns.color_palette("Reds_d", 4)
fileprivate let bluePalette = sns.color_palette("Blues_d", 5)

internal func makePlots(resultsDir: URL, rewardRatePeriod: Int) throws {
  let plotsDir = resultsDir.deletingLastPathComponent().appendingPathComponent("plots")
  if !FileManager.default.fileExists(atPath: plotsDir.path) {
    try FileManager.default.createDirectory(at: plotsDir, withIntermediateDirectories: true)
  }
  let configurationFiles = try FileManager.default.contentsOfDirectory(
    at: plotsDir,
    includingPropertiesForKeys: nil)
  try configurationFiles
    .filter { $0.lastPathComponent.hasSuffix(".tsv") }
    .map { try Plots(configurationFile: $0, resultsDir: resultsDir) }
    .forEach { try $0.makePlots(rewardRatePeriod: rewardRatePeriod) }
}

public struct Plots {
  public let configurationFile: URL
  public let resultsDir: URL
  public let title: String
  public let lineResultDirs: [(String, URL, Int)]

  public init(configurationFile: URL, resultsDir: URL) throws {
    self.configurationFile = configurationFile
    self.resultsDir = resultsDir
    let lines = (try String(contentsOf: configurationFile, encoding: .utf8))
      .components(separatedBy: .newlines)
    self.title = lines[0]
    self.lineResultDirs = lines
      .dropFirst(2)
      .map { $0.components(separatedBy: "\t") }
      .filter { $0.count == 3 }
      .map { ($0[0], resultsDir.appendingPathComponent($0[1]), Int($0[2])!) }
  }

  public func makePlots(rewardRatePeriod: Int) throws {
    // Set some plot styling parameters.
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    sns.set()
    sns.set_context("paper")
    sns.set_style("white")
    sns.set_style("ticks")

    // Create a new figure.
    let figure = plt.figure(figsize: [10.0, 6.0])
    let axes = plt.gca()

    for (plotIndex, (name, resultsDir, color)) in lineResultDirs.enumerated() {
      // Read the results from all saved result files.
      guard let resultFiles = try? FileManager.default.contentsOfDirectory(
        at: resultsDir,
        includingPropertiesForKeys: nil
      ) else { continue }
      let lines = resultFiles
        .filter { $0.lastPathComponent.hasSuffix("-rewards.tsv") }
        .compactMap { file -> Line? in
          guard let fileContent = try? String(contentsOf: file, encoding: .utf8) else {
            return nil
          }
          let rows = fileContent.components(separatedBy: .newlines)
            .dropFirst()
            .map { $0.components(separatedBy: "\t") }
            .filter { $0.count == 2 }
          if rows.isEmpty { return nil }
          let x = [Float](unsafeUninitializedCapacity: rows.count) {
            for i in rows.indices { $0[i] = Float(rows[i][0])! }
            $1 = rows.count
          }
          let y = [Float](unsafeUninitializedCapacity: rows.count) {
            for i in rows.indices { $0[i] = Float(rows[i][1])! }
            $1 = rows.count
          }
          return Line(x: x, y: y).movingAverage(period: rewardRatePeriod)
        }
      if lines.isEmpty { continue }
      // TODO: !!!
      // let rewardSchedules = resultFiles
      //   .filter { $0.lastPathComponent.hasSuffix("-reward-schedule.tsv") }
      //   .compactMap { file -> RewardSchedule? in
      //     guard let fileContent = try? String(contentsOf: file, encoding: .utf8) else {
      //       return nil
      //     }
      //     let rows = fileContent.components(separatedBy: .newlines)
      //       .dropFirst()
      //       .map { $0.components(separatedBy: "\t") }
      //       .filter { $0.count == 2 }
      //     if rows.isEmpty { return nil }
      //     let x = [Float](unsafeUninitializedCapacity: rows.count) {
      //       for i in rows.indices { $0[i] = Float(rows[i][0])! }
      //       $1 = rows.count
      //     }
      //     let rewardFunctions = [String](unsafeUninitializedCapacity: rows.count) {
      //       for i in rows.indices { $0[i] = rows[i][1] }
      //       $1 = rows.count
      //     }
      //     return RewardSchedule(x: x, rewardFunctions: rewardFunctions)
      //   }

      // Plot a line for this observation-network combination.
      let colorPalette = color == 0 ? bluePalette : redPalette
      lines.plotWithStandardError(on: axes, label: name, color: colorPalette[plotIndex])

      // TODO: !!!
      // if rewardSchedules.count > 1 {
      //   assert(rewardSchedules.allSatisfy { $0 == rewardSchedules.first! })
      //   rewardSchedules.first!.addVerticalAnnotations(on: axes)
      // }
    }

    // Use exponential notation for the x-axis labels.
    let xAxisFormatter = mpl.ticker.ScalarFormatter()
    xAxisFormatter.set_powerlimits([-3, 3])
    axes.xaxis.set_major_formatter(xAxisFormatter)

    // Add axis labels.
    axes.set_xlabel(
      "Time Step",
      color: "grey",
      fontname: "Lato",
      fontsize: 18,
      fontweight: "light")
    axes.set_ylabel(
      "Reward Rate (pts/s)",
      color: "grey",
      fontname: "Lato",
      fontsize: 18,
      fontweight: "light")
    axes.yaxis.set_tick_params(labelbottom: true)

    // Change the tick label sizes.
    plt.setp(axes.get_xticklabels(), fontname: "Lato", fontsize: 18, fontweight: "regular")
    plt.setp(axes.get_yticklabels(), fontname: "Lato", fontsize: 18, fontweight: "regular")

    // Set the figure title.
    figure.suptitle(
      title,
      x: 0.5,
      y: 1,
      fontname: "Lato",
      fontsize: 22,
      fontweight: "black")

    // Remove the grid.
    sns.despine()

    // Add a legend.
    plt.legend()

    // Save the figure.
    plt.savefig(
      configurationFile.deletingPathExtension().appendingPathExtension("pdf").path,
      bbox_inches: "tight")
  }
}

extension Experiment {
  public func makePlots(
    observations: [Observation],
    networks: [Network],
    rewardRatePeriod: Int
  ) throws {
    let resultsDir = self.resultsDir
      .appendingPathComponent(description)
      .appendingPathComponent(agent.description)

    // Set some plot styling parameters.
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    sns.set()
    sns.set_context("paper")
    sns.set_style("white")
    sns.set_style("ticks")

    // Create a new figure.
    let figure = plt.figure(figsize: [10.0, 6.0])
    let axes = plt.gca()

    // Plot the curves for all observation-network combinations.
    for observation in observations {
      // The following index is used for picking the color for the corresponding lines in the plot.
      let observationIndex: Int = {
        switch observation {
        case .vision: return 2
        case .scent: return 4
        case .visionAndScent: return 0
        }
      }()
      for network in networks {
        // Read the results from all saved result files.
        let resultsDir = resultsDir
          .appendingPathComponent(observation.description)
          .appendingPathComponent(network.description)
        guard let resultFiles = try? FileManager.default.contentsOfDirectory(
          at: resultsDir,
          includingPropertiesForKeys: nil
        ) else { continue }
        let lines = resultFiles
          .filter { $0.lastPathComponent.hasSuffix("-rewards.tsv") }
          .compactMap { file -> Line? in
            guard let fileContent = try? String(contentsOf: file, encoding: .utf8) else {
              return nil
            }
            let rows = fileContent.components(separatedBy: .newlines)
              .dropFirst()
              .map { $0.components(separatedBy: "\t") }
              .filter { $0.count == 2 }
            if rows.isEmpty { return nil }
            let x = [Float](unsafeUninitializedCapacity: rows.count) {
              for i in rows.indices { $0[i] = Float(rows[i][0])! }
              $1 = rows.count
            }
            let y = [Float](unsafeUninitializedCapacity: rows.count) {
              for i in rows.indices { $0[i] = Float(rows[i][1])! }
              $1 = rows.count
            }
            return Line(x: x, y: y).movingAverage(period: rewardRatePeriod)
          }
        if lines.isEmpty { continue }
        let rewardSchedules = resultFiles
          .filter { $0.lastPathComponent.hasSuffix("-reward-schedule.tsv") }
          .compactMap { file -> RewardSchedule? in
            guard let fileContent = try? String(contentsOf: file, encoding: .utf8) else {
              return nil
            }
            let rows = fileContent.components(separatedBy: .newlines)
              .dropFirst()
              .map { $0.components(separatedBy: "\t") }
              .filter { $0.count == 2 }
            if rows.isEmpty { return nil }
            let x = [Float](unsafeUninitializedCapacity: rows.count) {
              for i in rows.indices { $0[i] = Float(rows[i][0])! }
              $1 = rows.count
            }
            let rewardFunctions = [String](unsafeUninitializedCapacity: rows.count) {
              for i in rows.indices { $0[i] = rows[i][1] }
              $1 = rows.count
            }
            return RewardSchedule(x: x, rewardFunctions: rewardFunctions)
          }

        // Plot a line for this observation-network combination.
        let colorPalette = network == .plain ? bluePalette : redPalette
        lines.plotWithStandardError(
          on: axes,
          label: "\(network.description)-\(observation.description)",
          color: colorPalette[observationIndex])

        if !rewardSchedules.isEmpty {
          assert(rewardSchedules.allSatisfy { $0 == rewardSchedules.first! })
          rewardSchedules.first!.addVerticalAnnotations(on: axes)
        }
      }
    }

    // Use exponential notation for the x-axis labels.
    let xAxisFormatter = mpl.ticker.ScalarFormatter()
    xAxisFormatter.set_powerlimits([-3, 3])
    axes.xaxis.set_major_formatter(xAxisFormatter)

    // Add axis labels.
    axes.set_xlabel(
      "Time Step",
      color: "grey",
      fontname: "Lato",
      fontsize: 18,
      fontweight: "light")
    axes.set_ylabel(
      "Reward Rate (pts/s)",
      color: "grey",
      fontname: "Lato",
      fontsize: 18,
      fontweight: "light")
    axes.yaxis.set_tick_params(labelbottom: true)

    // Change the tick label sizes.
    plt.setp(axes.get_xticklabels(), fontname: "Lato", fontsize: 18, fontweight: "regular")
    plt.setp(axes.get_yticklabels(), fontname: "Lato", fontsize: 18, fontweight: "regular")

    // Set the figure title.
    figure.suptitle(
      "\(reward.description) \(agent.description)",
      x: 0.5,
      y: 1,
      fontname: "Lato",
      fontsize: 22,
      fontweight: "black")

    // Remove the grid.
    sns.despine()

    // Add a legend.
    plt.legend()

    // Save the figure.
    let observations = observations.map { $0.description } .joined(separator: "-")
    let networks = networks.map { $0.description } .joined(separator: "-")
    plt.savefig(
      resultsDir.appendingPathComponent("\(observations)-\(networks).pdf").path,
      bbox_inches: "tight")
  }
}

fileprivate struct Line: Equatable {
  fileprivate let x: [Float]
  fileprivate let y: [Float]
}

fileprivate struct RewardSchedule: Equatable {
  fileprivate let x: [Float]
  fileprivate let rewardFunctions: [String]
}

extension Line {
  fileprivate var cumulativeSum: Line {
    Line(x: x, y: [Float](y.scan(0, +).dropFirst()))
  }

  fileprivate func movingAverage(period: Int) -> Line {
    let period = Float(period)
    var yMovingAverage = [Float]()
    yMovingAverage.reserveCapacity(y.count)
    var sum = Float(0.0)
    var low = 0
    for i in x.indices {
      sum += y[i]
      while x[low] < x[i] - period {
        sum -= y[low]
        low += 1
      }
      yMovingAverage.append(sum / period)
    }
    return Line(x: x, y: yMovingAverage)
  }
}

extension Array where Element == Line {
  // TODO: This function can be made much faster.
  fileprivate func plotWithStandardError(
    on axes: PythonObject,
    label: String,
    color: PythonObject,
    pointCount: Int = 1000
  ) {
    precondition(!isEmpty)
    var x = [Float]()
    var yMean = [Float]()
    var yStandardDeviation = [Float]()
    x.reserveCapacity(pointCount)
    yMean.reserveCapacity(pointCount)
    yStandardDeviation.reserveCapacity(pointCount)
    var indices = [Int](repeating: 0, count: count)
    let xMin = map { $0.x.min() ?? 0.0 } .min() ?? 0.0
    let xMax = map { $0.x.max() ?? 0.0 } .max() ?? 0.0
    let interval = (xMax - xMin) / Float(pointCount - 1)
    for i in 0..<pointCount {
      let currentX = xMin + Float(i) * interval
      x.append(currentX)
      var ys = [Float]()
      ys.reserveCapacity(count)
      for j in self.indices {
        while self[j].x[indices[j]] < currentX && indices[j] < self[j].x.count - 1 {
          if self[j].x[indices[j] + 1] > currentX { break }
          indices[j] += 1
        }
        if self[j].x[indices[j]] == currentX {
          ys.append(self[j].y[indices[j]])
          continue
        }
        if indices[j] == self[j].x.count - 1 {
          continue
        }
        // TODO: Can do better than linear interpolation
        // by accounting for the in-between `y` values.
        let xLow = self[j].x[indices[j]]
        let xHigh = self[j].x[indices[j] + 1]
        let yLow = self[j].y[indices[j]]
        let yHigh = self[j].y[indices[j] + 1]
        ys.append((yLow * (xHigh - currentX) + yHigh * (currentX - xLow)) / (xHigh - xLow))
      }
      yMean.append(ys.mean)
      yStandardDeviation.append(ys.standardDeviation / Float(count).squareRoot())
    }
//    for i in yMean.indices {
//      if i == 0 { continue }
//      yMean[i] = yMean[i - 1] + yMean[i]
//    }
    axes.plot(x, yMean, label: label, color: color, linewidth: 2)
    axes.fill_between(
      x,
      zip(yMean, yStandardDeviation).map(-).map { Swift.max($0, 0.0) },
      zip(yMean, yStandardDeviation).map(+).map { Swift.max($0, 0.0) },
      color: color,
      alpha: 0.1,
      linewidth: 0)
    axes.axhline(y: yMean.max(), linestyle: "dashed", alpha: 0.7, color: color)
  }
}

extension RewardSchedule {
  fileprivate func addVerticalAnnotations(on axes: PythonObject) {
    zip(x, rewardFunctions).forEach { (x, rewardFunction) in
      axes.axvline(x: x, linestyle: "solid", linewidth: 1.0, color: "#222222", alpha: 0.5)
      axes.text(
        x: x + 0.01 * Float(axes.get_xlim()[1])!, y: 0.00,
        s: rewardFunction
          .split(separator: "∧")
          .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
          .joined(separator: "\n"),
        color: "#222222", alpha: 0.5,
        fontsize: 18, fontfamily: "DejaVu Sans", fontweight: 400)
    }
  }
}
