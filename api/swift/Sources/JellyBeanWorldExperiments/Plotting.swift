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

fileprivate let redPalette = sns.color_palette("Reds", 4)
fileprivate let bluePalette = sns.color_palette("Blues_d", 4)

public struct ResultsPlot {
  public let reward: Reward
  public let agent: Agent
  public let observations: [Observation]
  public let networks: [Network]
  public let rewardRatePeriod: Int
  public let resultsDir: URL

  public init(
    reward: Reward,
    agent: Agent,
    observations: [Observation],
    networks: [Network],
    rewardRatePeriod: Int,
    resultsDir: URL
  ) {
    self.reward = reward
    self.agent = agent
    self.observations = observations
    self.networks = networks
    self.rewardRatePeriod = rewardRatePeriod
    self.resultsDir = resultsDir
      .appendingPathComponent(reward.description)
      .appendingPathComponent(agent.description)
  }

  public func plot() throws {
    // Set some plot styling parameters.
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    sns.set()
    sns.set_context("paper")
    sns.set_style("white")
    sns.set_style("ticks")

    // Create a new figure.
    let figure = plt.figure(figsize: [10.0, 6.0])
    let ax = plt.gca()

    // Plot the curves for all observation-network combinations.
    for observation in observations {
      // The following index is used for picking the color for the corresponding lines in the plot.
      let observationIndex: Int = {
        switch observation {
        case .vision: return 1
        case .scent: return 2
        case .visionAndScent: return 0
        }
      }()
      for network in networks {
        // Read the results from all saved result files.
        let resultsDir = self.resultsDir
          .appendingPathComponent(observation.description)
          .appendingPathComponent(network.description)
        guard let resultFiles = try? FileManager.default.contentsOfDirectory(
          at: resultsDir,
          includingPropertiesForKeys: nil
        ) else { continue }
        let lines = resultFiles
          .compactMap { (try? Line(fromFile: $0))?
          .movingAverage(period: rewardRatePeriod) }
        if lines.isEmpty { continue }

        // Plot a line for this observation-network combination.
        let colorPalette = network == .plain ? bluePalette : redPalette
        lines.plotWithStandardDeviation(
          on: ax,
          label: "\(network.description)-\(observation.description)",
          color: colorPalette[observationIndex])
      }
    }

    // Use exponential notation for the x-axis labels.
    let xAxisFormatter = mpl.ticker.ScalarFormatter()
    xAxisFormatter.set_powerlimits([-3, 3])
    ax.xaxis.set_major_formatter(xAxisFormatter)

    // Add axis labels.
    ax.set_xlabel(
      "Time Step",
      color: "grey",
      fontname: "Lato",
      fontsize: 18,
      fontweight: "light")
    ax.set_ylabel(
      "Reward Rate (pts/s)",
      color: "grey",
      fontname: "Lato",
      fontsize: 18,
      fontweight: "light")
    ax.yaxis.set_tick_params(labelbottom: true)

    // Change the tick label sizes.
    plt.setp(ax.get_xticklabels(), fontname: "Lato", fontsize: 18, fontweight: "regular")
    plt.setp(ax.get_yticklabels(), fontname: "Lato", fontsize: 18, fontweight: "regular")

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
    let observations = self.observations.map { $0.description } .joined(separator: "-")
    let networks = self.networks.map { $0.description } .joined(separator: "-")
    plt.savefig(
      resultsDir.appendingPathComponent("\(observations)-\(networks).pdf").path,
      bbox_inches: "tight")
  }
}

fileprivate struct Line {
  fileprivate let x: [Float]
  fileprivate let y: [Float]
}

extension Line {
  fileprivate init(fromFile file: URL) throws {
    let rows = try String(contentsOf: file, encoding: .utf8)
      .components(separatedBy: .newlines)
      .dropFirst()
      .map { $0.components(separatedBy: "\t") }
      .filter { $0.count == 2 }
    self.x = [Float](unsafeUninitializedCapacity: rows.count) { buffer, initializedCount in
      for i in rows.indices { buffer[i] = Float(rows[i][0])! }
      initializedCount = rows.count
    }
    self.y = [Float](unsafeUninitializedCapacity: rows.count) { buffer, initializedCount in
      for i in rows.indices { buffer[i] = Float(rows[i][1])! }
      initializedCount = rows.count
    }
  }
}

extension Line {
  fileprivate var cumulativeSum: Line {
    Line(x: x, y: [Float](y.scan(0, +).dropFirst()))
  }

  fileprivate func movingAverage(period: Int) -> Line {
    // TODO: This can be made much faster.
    let period = Float(period)
    var yMovingAverage = [Float]()
    yMovingAverage.reserveCapacity(y.count)
    for i in x.indices {
      var buffer = [Float]()
      var j = i
      while j >= 0 && x[j] >= x[i] - period {
        buffer.append(y[j])
        j -= 1
      }
      // TODO: let interval = x[i] - (j >= 0 ? x[j] : 0.0)
      yMovingAverage.append(buffer.sum / period)
    }
    return Line(x: x, y: yMovingAverage)
  }
}

extension Array where Element == Line {
  fileprivate func plotWithStandardDeviation(
    on axes: PythonObject,
    label: String,
    color: PythonObject
  ) {
    precondition(!isEmpty)
    precondition(allSatisfy({ $0.x == first!.x }))
    let x = first!.x
    let y = map { $0.y } .transposed()
    let yMean = y.map { $0.mean }
    let yStandardDeviation = y.map { $0.standardDeviation }
    axes.plot(x, yMean, label: label, color: color, linewidth: 2)
    axes.fill_between(
      x,
      zip(yMean, yStandardDeviation).map(-),
      zip(yMean, yStandardDeviation).map(+),
      color: color,
      alpha: 0.1,
      linewidth: 0)
    axes.axhline(y: yMean.max(), linestyle: "dashed", alpha: 0.7, color: color)
  }
}
