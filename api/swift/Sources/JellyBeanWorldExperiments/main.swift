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
import JellyBeanWorld
import Logging
import ReinforcementLearning
import SPMUtility

let logger = Logger(label: "Jelly Bean World Experiment")

public enum Error: Swift.Error {
  case invalidArgument
}

public enum Reward: String, CustomStringConvertible {
  case collectJellyBeans

  public var description: String {
    switch self {
    case .collectJellyBeans: return "CollectJellyBeans"
    }
  }
}

extension Reward: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Reward.collectJellyBeans.rawValue, "Each collected jelly bean is worth 1 point.")
    ])
  }
}

public enum Agent: String, CustomStringConvertible {
  case reinforce, a2c, ppo, dqn

  public var description: String {
    switch self {
    case .reinforce: return "REINFORCE"
    case .a2c: return "A2C"
    case .ppo: return "PPO"
    case .dqn: return "DQN"
    }
  }
}

extension Agent: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Agent.reinforce.rawValue, "REINFORCE agent."),
      (Agent.a2c.rawValue, "Advantage Actor Critic (A2C) agent."),
      (Agent.ppo.rawValue, "Proximal Policy Optimization (PPO) agent."),
      (Agent.dqn.rawValue, "Deep Q-Network (DQN) agent.")
    ])
  }
}

public enum Observation: String, CustomStringConvertible {
  case vision, scent, visionAndScent

  public var description: String {
    switch self {
    case .vision: return "Vision"
    case .scent: return "Scent"
    case .visionAndScent: return "Vision+Scent"
    }
  }
}

extension Observation: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Observation.vision.rawValue, "Visual field."),
      (Observation.scent.rawValue, "Scent vector at the current cell."),
      (Observation.visionAndScent.rawValue, "Visual field and scent vector at the current cell.")
    ])
  }
}

public enum Network: String, CustomStringConvertible {
  case plain, contextual

  public var description: String {
    switch self {
    case .plain: return "Plain"
    case .contextual: return "Contextual"
    }
  }
}

extension Network: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Network.plain.rawValue, "Plain network."),
      (Network.contextual.rawValue, "Contextual network.")
    ])
  }
}

let parser = ArgumentParser(
  usage: "<options>",
  overview: "This executable can be used to run Jelly Bean World experiments.")
let rewardArg: OptionArgument<Reward> = parser.add(
  option: "--reward",
  kind: Reward.self,
  usage: "Reward schedule to use. Can be one of: `collectJellyBeans`.")
let agentArg: OptionArgument<Agent> = parser.add(
  option: "--agent",
  kind: Agent.self,
  usage: "Agent to use in the experiments. Can be one of: `reinforce`, `a2c`, `ppo`, and `dqn`.")
let observationArg: OptionArgument<Observation> = parser.add(
  option: "--observation",
  kind: Observation.self,
  usage: "Observations type. Can be one of: `vision`, `scent`, and `visionAndScent`.")
let networkArg: OptionArgument<Network> = parser.add(
  option: "--network",
  kind: Network.self,
  usage: "Network type. Can be one of: `plain` and `contextual`.")
let batchSizeArg: OptionArgument<Int> = parser.add(
  option: "--batch-size",
  kind: Int.self,
  usage: "Batch size to use for the experiments. Defaults to 32.")
let stepCountArg: OptionArgument<Int> = parser.add(
  option: "--step-count-per-update",
  kind: Int.self,
  usage: "Total number of steps to run. Defaults to 1000000.")
let stepCountPerUpdateArg: OptionArgument<Int> = parser.add(
  option: "--step-count-per-update",
  kind: Int.self,
  usage: "Number of steps between model updates. Defaults to 128.")
let resultsDirArg: OptionArgument<PathArgument> = parser.add(
  option: "--results-dir",
  kind: PathArgument.self,
  usage: "Path to the results directory.")

let fileManager = FileManager.default

// The first argument is always the executable, and so we drop it.
let arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
let parsedArguments = try parser.parse(arguments)
let currentDir = URL(fileURLWithPath: fileManager.currentDirectoryPath)
var resultsDir: Foundation.URL = {
  if let argument = parsedArguments.get(resultsDirArg) {
    return URL(fileURLWithPath: argument.path.pathString)
  }
  return currentDir.appendingPathComponent("temp/results")
}()
guard let reward = parsedArguments.get(rewardArg) else { throw Error.invalidArgument }
guard let agent = parsedArguments.get(agentArg) else { throw Error.invalidArgument }
guard let observation = parsedArguments.get(observationArg) else { throw Error.invalidArgument }
guard let network = parsedArguments.get(networkArg) else { throw Error.invalidArgument }
let batchSize = parsedArguments.get(batchSizeArg) ?? 32
let stepCount = parsedArguments.get(stepCountArg) ?? 1000000
let stepCountPerUpdate = parsedArguments.get(stepCountPerUpdateArg) ?? 128

// Create the results file.
resultsDir = resultsDir
  .appendingPathComponent(reward.description)
  .appendingPathComponent(agent.description)
  .appendingPathComponent(observation.description)
  .appendingPathComponent(network.description)
if !fileManager.fileExists(atPath: resultsDir.path) {
  try fileManager.createDirectory(
    at: resultsDir,
    withIntermediateDirectories: true)
}
let runIDs = try fileManager.contentsOfDirectory(at: resultsDir, includingPropertiesForKeys: nil)
  .filter { $0.pathExtension == "tsv" }
  .compactMap { Int($0.deletingPathExtension().lastPathComponent) }
var runID = 0
while runIDs.contains(runID) { runID += 1 }
let resultsFile = resultsDir.appendingPathComponent("\(runID).tsv")
fileManager.createFile(atPath: resultsFile.path, contents: "step\treward\n".data(using: .utf8))

// Run the experiment.
let experiment = Experiment(
  reward: reward,
  agent: agent,
  observation: observation,
  network: network,
  batchSize: batchSize,
  stepCount: stepCount,
  stepCountPerUpdate: stepCountPerUpdate)
try! experiment.run(resultsFile: resultsFile, logFrequency: 10)
