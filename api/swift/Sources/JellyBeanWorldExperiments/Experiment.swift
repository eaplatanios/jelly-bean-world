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
  public let observation: Observation
  public let network: Network
  public let batchSize: Int
  public let stepCount: Int
  public let stepCountPerUpdate: Int
  public let runID: Int
  public let resultsFile: URL

  public init(
    reward: Reward,
    agent: Agent,
    observation: Observation,
    network: Network,
    batchSize: Int,
    stepCount: Int,
    stepCountPerUpdate: Int,
    resultsDir: URL
  ) throws {
    self.reward = reward
    self.agent = agent
    self.observation = observation
    self.network = network
    self.batchSize = batchSize
    self.stepCount = stepCount
    self.stepCountPerUpdate = stepCountPerUpdate
    
    // Create the results file.
    let resultsDir = resultsDir
      .appendingPathComponent(reward.description)
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
      Int($0.deletingPathExtension().lastPathComponent)
    }
    var runID = 0
    while runIDs.contains(runID) { runID += 1 }
    self.runID = runID
    self.resultsFile = resultsDir.appendingPathComponent("\(runID).tsv")
    FileManager.default.createFile(
      atPath: self.resultsFile.path,
      contents: "step\treward\n".data(using: .utf8))
  }

  public func run(writeFrequency: Int = 100, logFrequency: Int = 1000) throws {
    let configurations = (0..<batchSize).map { _ in
      JellyBeanWorld.Environment.Configuration(
        simulatorConfiguration: simulatorConfiguration(randomSeed: UInt32(runID)),
        rewardSchedule: reward.schedule)
    }
    var environment = try JellyBeanWorld.Environment(
      configurations: configurations,
      parallelizedBatchProcessing: true)
    var rewardWriteDeque = Deque<Float>(size: writeFrequency)
    var rewardLogDeque = Deque<Float>(size: logFrequency)
    try withRandomSeedForTensorFlow((Int32(runID), Int32(runID))) {
      var agent = self.agent.create(
        in: environment,
        network: network,
        observation: observation,
        batchSize: batchSize)
      let resultsFileHandle = try? FileHandle(forWritingTo: resultsFile)
      defer { resultsFileHandle?.closeFile() }
      resultsFileHandle?.seekToEndOfFile()
      var environmentStep = 0
      for _ in 0..<(stepCount / stepCountPerUpdate) {
        try agent.update(
          using: &environment,
          maxSteps: stepCountPerUpdate * batchSize,
          callbacks: [{ (environment, trajectory) in
            let reward = trajectory.reward.mean().scalarized()
            rewardWriteDeque.push(reward)
            rewardLogDeque.push(reward)
            if environmentStep % writeFrequency == 0 {
              let reward = rewardWriteDeque.sum()
              resultsFileHandle?.write("\(environmentStep)\t\(reward)\n".data(using: .utf8)!)
            }
            if environmentStep % logFrequency == 0 {
              let rewardRate = rewardLogDeque.mean()
              logger.info("Step: \(environmentStep) | Reward Rate: \(rewardRate)/s")
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
    }
  }
}

extension JellyBeanWorldExperiments.Agent {
  public func create(
    in environment: JellyBeanWorld.Environment,
    network: Network,
    observation: Observation,
    batchSize: Int
   ) -> AnyAgent<JellyBeanWorld.Environment, LSTMState<Float>> {
    let learningRate = ExponentiallyDecayedLearningRate(
      baseLearningRate: FixedLearningRate(Float(1e-4)),
      decayRate: 0.999,
      decayStepCount: 1,
      lowerBound: 1e-6)
    let advantageFunction = GeneralizedAdvantageEstimation(
      discountFactor: 0.99,
      discountWeight: 0.95)
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
        optimizer: AMSGrad(for: network),
        learningRate: learningRate,
        advantageFunction: advantageFunction,
        advantagesNormalizer: nil,
        useTDLambdaReturn: true,
        clip: ppoClip,
        penalty: ppoPenalty,
        valueEstimationLoss: ppoValueEstimationLoss,
        entropyRegularization: ppoEntropyRegularization,
        iterationCountPerUpdate: 1))
    case (.ppo, .plain, .scent):
      let network = ScentActorCritic()
      return AnyAgent(PPOAgent(
        for: environment,
        network: network,
        initialState: network.initialState(batchSize: batchSize),
        optimizer: AMSGrad(for: network),
        learningRate: learningRate,
        advantageFunction: advantageFunction,
        advantagesNormalizer: nil,
        useTDLambdaReturn: true,
        clip: ppoClip,
        penalty: ppoPenalty,
        valueEstimationLoss: ppoValueEstimationLoss,
        entropyRegularization: ppoEntropyRegularization,
        iterationCountPerUpdate: 1))
    case (.ppo, .plain, .visionAndScent):
      let network = VisionAndScentActorCritic()
      return AnyAgent(PPOAgent(
        for: environment,
        network: network,
        initialState: network.initialState(batchSize: batchSize),
        optimizer: AMSGrad(for: network),
        learningRate: learningRate,
        advantageFunction: advantageFunction,
        advantagesNormalizer: nil,
        useTDLambdaReturn: true,
        clip: ppoClip,
        penalty: ppoPenalty,
        valueEstimationLoss: ppoValueEstimationLoss,
        entropyRegularization: ppoEntropyRegularization,
        iterationCountPerUpdate: 1))
    case (.ppo, .contextual, _): fatalError("Not supported yet.")
    case (.dqn, _, _): fatalError("Not supported yet.")
    }
  }
}
