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
  case invalidCommand, invalidArgument
}

public enum Command: String {
  case run, makePlots
}

extension Command: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Command.run.rawValue, "Runs a Jelly Bean World experiment."),
      (Command.makePlots.rawValue,
        "Creates plots for the results of previously run Jelly Bean World experiments.")
    ])
  }
}

public enum Reward: String, CaseIterable, CustomStringConvertible {
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

public enum Agent: String, CaseIterable, CustomStringConvertible {
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

public enum Observation: String, CaseIterable, CustomStringConvertible {
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

public enum Network: String, CaseIterable, CustomStringConvertible {
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
let commandArg: PositionalArgument<Command> = parser.add(
  positional: "command",
  kind: Command.self,
  usage: "Experiment command to invoke. Can be one of: `run` and `makePlots`.")
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
let agentFieldOfViewArg: OptionArgument<Int> = parser.add(
  option: "--agent-field-of-view",
  kind: Int.self,
  usage: "Agents' field of view. Defaults to 360.")
let batchSizeArg: OptionArgument<Int> = parser.add(
  option: "--batch-size",
  kind: Int.self,
  usage: "Batch size to use for the experiments. Defaults to 32.")
let stepCountArg: OptionArgument<Int> = parser.add(
  option: "--step-count",
  kind: Int.self,
  usage: "Total number of steps to run. Defaults to `10_000_000`.")
let stepCountPerUpdateArg: OptionArgument<Int> = parser.add(
  option: "--step-count-per-update",
  kind: Int.self,
  usage: "Number of steps between model updates. Defaults to 512.")
let rewardRatePeriodArg: OptionArgument<Int> = parser.add(
  option: "--reward-rate-period",
  kind: Int.self,
  usage: "Moving average period used when computing the reward rate. Defaults to `100_000`.")
let resultsDirArg: OptionArgument<PathArgument> = parser.add(
  option: "--results-dir",
  kind: PathArgument.self,
  usage: "Path to the results directory.")
let minimumRunIDArg: OptionArgument<Int> = parser.add(
  option: "--minimum-run-id",
  kind: Int.self,
  usage: "Minimum run ID to use.")
let serverPortsArg: OptionArgument<[Int]> = parser.add(
  option: "--server-ports",
  kind: [Int].self,
  usage: "Ports to use for launching simulation servers.")

// The first argument is always the executable, and so we drop it.
let arguments = Array(ProcessInfo.processInfo.arguments.dropFirst())
let parsedArguments = try parser.parse(arguments)
let currentDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let resultsDir: Foundation.URL = {
  if let argument = parsedArguments.get(resultsDirArg) {
    return URL(fileURLWithPath: argument.path.pathString)
  }
  return currentDir.appendingPathComponent("temp/results")
}()
guard let reward = parsedArguments.get(rewardArg) else { throw Error.invalidArgument }
guard let agent = parsedArguments.get(agentArg) else { throw Error.invalidArgument }
let agentFieldOfView = parsedArguments.get(agentFieldOfViewArg) ?? 360
let batchSize = parsedArguments.get(batchSizeArg) ?? 32
let stepCount = parsedArguments.get(stepCountArg) ?? 10_000_000
let stepCountPerUpdate = parsedArguments.get(stepCountPerUpdateArg) ?? 512
let rewardRatePeriod = parsedArguments.get(rewardRatePeriodArg) ?? 100_000
let minimumRunID = parsedArguments.get(minimumRunIDArg) ?? 0
let serverPorts = parsedArguments.get(serverPortsArg)

let experiment = try! Experiment(
  reward: reward,
  agent: agent,
  agentFieldOfView: agentFieldOfView,
  batchSize: batchSize,
  stepCount: stepCount,
  stepCountPerUpdate: stepCountPerUpdate,
  resultsDir: resultsDir,
  minimumRunID: minimumRunID,
  serverPorts: serverPorts)

switch parsedArguments.get(commandArg) {
case .run:
  guard let observation = parsedArguments.get(observationArg) else { throw Error.invalidArgument }
  guard let network = parsedArguments.get(networkArg) else { throw Error.invalidArgument }
  try! experiment.run(
    observation: observation,
    network: network,
    writeFrequency: 100,
    logFrequency: 1000)
case .makePlots:
  try! experiment.makePlots(
    observations: [Observation](Observation.allCases),
    networks: [Network](Network.allCases),
    rewardRatePeriod: rewardRatePeriod)
case _: throw Error.invalidCommand
}
