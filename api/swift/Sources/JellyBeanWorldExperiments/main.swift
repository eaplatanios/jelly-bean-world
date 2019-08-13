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
import Logging
import SPMUtility

let logger = Logger(label: "Jelly Bean World Experiment")

enum Error: Swift.Error {
  case invalidExperiment, invalidAgent
}

enum Agent: String {
  case dummy // This is only used for dummy tests.
  case reinforce
  case a2c
  case ppo
  case dqn
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

enum Experiment: String {
  case dummy
  case collectJellyBeans
}

extension Experiment: StringEnumArgument {
  public static var completion: ShellCompletion {
    return .values([
      (Experiment.dummy.rawValue, "Dummy experiment used for testing."),
      (Experiment.collectJellyBeans.rawValue, "'CollectJellyBeans' experiment.")
    ])
  }
}

let parser = ArgumentParser(
  usage: "<options>",
  overview: "This executable can be used to run Jelly Bean World experiments.")
let experimentArg: PositionalArgument<Experiment> = parser.add(
  positional: "experiment",
  kind: Experiment.self,
  usage: "Experiment to run. Can be one of: `dummy`, and `collectJellyBeans`.")
let agentArg: PositionalArgument<Agent> = parser.add(
  positional: "agent",
  kind: Agent.self,
  usage: "Agent to use in the experiments. Can be one of: `reinforce`, `a2c`, `ppo`, and `dqn`.")
let resultsDirArg: OptionArgument<PathArgument> = parser.add(
  option: "--results-dir",
  kind: PathArgument.self,
  usage: "Path to the results directory.")

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
guard let experiment = parsedArguments.get(experimentArg) else { throw Error.invalidExperiment }

switch experiment {
case .dummy: runDummyExperiment()
case .collectJellyBeans:
  guard let agent = parsedArguments.get(agentArg) else { throw Error.invalidAgent }
  // runPPOExperiment(batchSize: 32)
}
