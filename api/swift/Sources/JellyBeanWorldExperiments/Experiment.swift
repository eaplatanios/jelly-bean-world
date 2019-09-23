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
import ReinforcementLearning
import TensorFlow

public struct Experiment {
  public let reward: Reward
  public let agent: Agent
  public let agentFieldOfView: Int
  public let enableVisualOcclusion: Bool
  public let batchSize: Int
  public let stepCount: Int
  public let stepCountPerUpdate: Int
  public let resultsDir: URL
  public let minimumRunID: Int
  public let serverPorts: [Int]?

  public var description: String {
    "reward-\(reward.description)" +
      ":agent-fov-\(agentFieldOfView)" +
      "\(enableVisualOcclusion ? "" : ":no-visual-occlusion")"
  }

  public init(
    reward: Reward,
    agent: Agent,
    agentFieldOfView: Int,
    enableVisualOcclusion: Bool,
    batchSize: Int,
    stepCount: Int,
    stepCountPerUpdate: Int,
    resultsDir: URL,
    minimumRunID: Int = 0,
    serverPorts: [Int]? = nil
  ) throws {
    self.reward = reward
    self.agent = agent
    self.agentFieldOfView = agentFieldOfView
    self.enableVisualOcclusion = enableVisualOcclusion
    self.batchSize = batchSize
    self.stepCount = stepCount
    self.stepCountPerUpdate = stepCountPerUpdate
    self.resultsDir = resultsDir
    self.minimumRunID = minimumRunID
    self.serverPorts = serverPorts
  }

  public func run(
    observation: Observation,
    network: Network,
    writeFrequency: Int = 100,
    logFrequency: Int = 1000
   ) throws {
    // Create the results file.
    let resultsDir = self.resultsDir
      .appendingPathComponent(description)
      .appendingPathComponent(agent.description)
      .appendingPathComponent(observation.description)
      .appendingPathComponent(network.description)
    if !FileManager.default.fileExists(atPath: resultsDir.path) {
      try FileManager.default.createDirectory(
        at: resultsDir,
        withIntermediateDirectories: true)
    }
    let runIDs = try FileManager.default.contentsOfDirectory(
      at: resultsDir,
      includingPropertiesForKeys: nil
    ).filter { $0.pathExtension == "tsv" } .compactMap {
      $0.lastPathComponent.split(separator: "-").first.flatMap { Int($0) }
    }
    var runID = minimumRunID
    while runIDs.contains(runID) { runID += 1 }
    let resultsFile = resultsDir.appendingPathComponent("\(runID)-rewards.tsv")
    let rewardScheduleFile = resultsDir.appendingPathComponent("\(runID)-reward-schedule.tsv")
    FileManager.default.createFile(
      atPath: resultsFile.path,
      contents: "step\treward\n".data(using: .utf8))
    FileManager.default.createFile(
      atPath: rewardScheduleFile.path,
      contents: "step\treward\n".data(using: .utf8))

    // Create a Jelly Bean World environment.
    let configuration = simulatorConfiguration(
      randomSeed: UInt32(runID),
      agentFieldOfView: agentFieldOfView,
      enableVisualOcclusion: enableVisualOcclusion)
    let configurations = (0..<batchSize).map {
      batchIndex -> JellyBeanWorld.Environment.Configuration in
      if let ports = serverPorts, batchIndex < ports.count {
        return JellyBeanWorld.Environment.Configuration(
          simulatorConfiguration: configuration,
          rewardSchedule: reward.schedule,
          serverConfiguration: Simulator.ServerConfiguration(port: UInt32(ports[batchIndex])))
      }
      return JellyBeanWorld.Environment.Configuration(
        simulatorConfiguration: configuration,
        rewardSchedule: reward.schedule)
    }
    var environment = try JellyBeanWorld.Environment(
      configurations: configurations,
      parallelizedBatchProcessing: batchSize > 1)

    // Run the experiment.
    var totalReward = Float(0.0)
    var rewardWriteDeque = Deque<Float>(size: writeFrequency)
    var currentRewardFunction = environment.currentStep.observation.rewardFunction
    try withRandomSeedForTensorFlow((Int32(runID), Int32(runID))) {
      var agent = self.agent.create(
        in: environment,
        network: network,
        observation: observation,
        batchSize: batchSize)
      let resultsFileHandle = try? FileHandle(forWritingTo: resultsFile)
      defer { resultsFileHandle?.closeFile() }
      resultsFileHandle?.seekToEndOfFile()
      let rewardScheduleFileHandle = try? FileHandle(forWritingTo: rewardScheduleFile)
      defer { rewardScheduleFileHandle?.closeFile() }
      rewardScheduleFileHandle?.seekToEndOfFile()
      var environmentStep = 0
      for _ in 0..<(stepCount / stepCountPerUpdate) {
        try agent.update(
          using: &environment,
          maxSteps: stepCountPerUpdate * batchSize,
          callbacks: [{ (environment, trajectory) in
            let reward = trajectory.reward.mean().scalarized()
            totalReward += reward
            rewardWriteDeque.push(reward)
            if environmentStep % logFrequency == 0 {
              let rewardRate = totalReward / Float(environmentStep)
              logger.info("Step: \(environmentStep) | Reward Rate: \(rewardRate)/s")
            }
            if environmentStep % writeFrequency == 0 {
              let reward = rewardWriteDeque.sum()
              resultsFileHandle?.write("\(environmentStep)\t\(reward)\n".data(using: .utf8)!)
            }
            let rewardFunction = trajectory.observation.rewardFunction
            if environmentStep == 0 || rewardFunction != currentRewardFunction {
              currentRewardFunction = rewardFunction
              rewardScheduleFileHandle?.write(
                "\(environmentStep)\t\(currentRewardFunction!.description)\n".data(using: .utf8)!)
            }
            environmentStep += 1
          }])
      }
    }
  }
}

extension JellyBeanWorldExperiments.Reward {
  public var schedule: RewardSchedule {
    switch self {
    case .collectJellyBeans:
      return FixedReward(JellyBeanWorld.Reward.collect(item: jellyBean, value: 1.0))
    case .collectOnions:
      return FixedReward(JellyBeanWorld.Reward.collect(item: onion, value: 1.0))
    case .collectJellyBeansAvoidOnions:
      return FixedReward(JellyBeanWorld.Reward.combined(
        JellyBeanWorld.Reward.collect(item: jellyBean, value: 1.0),
        JellyBeanWorld.Reward.avoid(item: onion, value: 1.0)))
      // return FixedReward(JellyBeanWorld.Reward.combined(
      //   JellyBeanWorld.Reward.combined(
      //     JellyBeanWorld.Reward.collect(item: jellyBean, value: 1.0),
      //     JellyBeanWorld.Reward.avoid(item: onion, value: 1.0)),
      //   JellyBeanWorld.Reward.explore(value: 0.01)))
    }
  }
}

extension JellyBeanWorldExperiments.Agent {
  public func create(
    in environment: JellyBeanWorld.Environment,
    network: Network,
    observation: Observation,
    batchSize: Int
   ) -> AnyAgent<JellyBeanWorld.Environment, LSTMCell<Float>.State> {
    // let learningRate = ExponentiallyDecayedLearningRate(
    //   baseLearningRate: FixedLearningRate(Float(1e-4)),
    //   decayRate: 0.999,
    //   decayStepCount: 1)
    let learningRate = FixedLearningRate(Float(1e-4))
    let advantageFunction = GeneralizedAdvantageEstimation(
      discountFactor: 0.99,
      discountWeight: 0.9)
    let ppoClip = PPOClip(epsilon: 0.1)
    let ppoPenalty = PPOPenalty(klCutoffFactor: 0.5)
    let ppoValueEstimationLoss = PPOValueEstimationLoss(weight: 0.5, clipThreshold: 0.1)
    let ppoEntropyRegularization = PPOEntropyRegularization(weight: 0.01)
    switch (self, network, observation) {
    case (.reinforce, _, _): fatalError("Not supported yet.")
    case (.a2c, _, _): fatalError("Not supported yet.")
    case (.ppo, .plain, .vision):
      let network = VisionActorCritic()
      return AnyAgent(PPOAgent(
        for: environment,
        network: network,
        initialState: network.initialState(batchSize: batchSize),
        optimizer: { AMSGrad(for: $0) },
        learningRate: learningRate,
        maxGradientNorm: 0.5,
        advantageFunction: advantageFunction,
        advantagesNormalizer: nil,
        useTDLambdaReturn: true,
        clip: ppoClip,
        penalty: ppoPenalty,
        valueEstimationLoss: ppoValueEstimationLoss,
        entropyRegularization: ppoEntropyRegularization,
        iterationCountPerUpdate: 4))
    case (.ppo, .plain, .scent):
      let network = ScentActorCritic()
      return AnyAgent(PPOAgent(
        for: environment,
        network: network,
        initialState: network.initialState(batchSize: batchSize),
        optimizer: { AMSGrad(for: $0) },
        learningRate: learningRate,
        maxGradientNorm: 0.5,
        advantageFunction: advantageFunction,
        advantagesNormalizer: nil,
        useTDLambdaReturn: true,
        clip: ppoClip,
        penalty: ppoPenalty,
        valueEstimationLoss: ppoValueEstimationLoss,
        entropyRegularization: ppoEntropyRegularization,
        iterationCountPerUpdate: 4))
    case (.ppo, .plain, .visionAndScent):
      let network = VisionAndScentActorCritic()
      return AnyAgent(PPOAgent(
        for: environment,
        network: network,
        initialState: network.initialState(batchSize: batchSize),
        optimizer: { AMSGrad(for: $0) },
        learningRate: learningRate,
        maxGradientNorm: 0.5,
        advantageFunction: advantageFunction,
        advantagesNormalizer: nil,
        useTDLambdaReturn: true,
        clip: ppoClip,
        penalty: ppoPenalty,
        valueEstimationLoss: ppoValueEstimationLoss,
        entropyRegularization: ppoEntropyRegularization,
        iterationCountPerUpdate: 4))
    case (.ppo, .contextual, _): fatalError("Not supported yet.")
    case (.dqn, _, _): fatalError("Not supported yet.")
    }
  }
}
