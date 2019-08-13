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

import JellyBeanWorld
import Logging
import ReinforcementLearning
import TensorFlow

public enum InputType {
  case vision
  case scent
  case visionAndScent
}

public enum NetworkType {
  case plain
  case contextual
}

public func runPPO(batchSize: Int = 32) throws {
  let logger = Logger(label: "Jelly Bean World - PPO Agent")

  // let reward = Reward(item: jellyBean, value: 1.0) + Reward(item: onion, value: -1.0)
  let reward = Reward.collect(item: jellyBean, value: 1.0)
  let rewardSchedule = FixedReward(reward)
  let configurations = (0..<batchSize).map { _ in
    JellyBeanWorld.Environment.Configuration(
      simulatorConfiguration: simulatorConfiguration,
      rewardSchedule: rewardSchedule)
  }
  var environment = try JellyBeanWorld.Environment(configurations: configurations)
  var totalCumulativeReward = TotalCumulativeReward(for: environment)

  let network = VisionActorCritic(hiddenSize: 64)
  var agent = PPOAgent(
    for: environment,
    network: network,
    optimizer: AMSGrad(for: network, learningRate: 1e-3),
    learningRateSchedule: LinearLearningRateDecay(slope: 1e-3 / 100000.0, lowerBound: 1e-6),
    advantageFunction: GeneralizedAdvantageEstimation(discountFactor: 0.99, discountWeight: 0.95),
    advantagesNormalizer: TensorNormalizer<Float>(streaming: false, alongAxes: 0, 1),
    useTDLambdaReturn: true,
    clip: PPOClip(epsilon: 0.1),
    penalty: PPOPenalty(klCutoffFactor: 0.5),
    valueEstimationLoss: PPOValueEstimationLoss(weight: 0.5, clipThreshold: 0.1),
    entropyRegularization: PPOEntropyRegularization(weight: 0.01),
    iterationCountPerUpdate: 1)

  for step in 0..<1000000 {
    let loss = try agent.update(
      using: &environment,
      maxSteps: 128 * batchSize,
      callbacks: [{ (environment, trajectory) in
        totalCumulativeReward.update(using: trajectory)
        // if step > 140 {
        //   painter.draw()
        // }
      }])
    if step % 10 == 0 {
      logger.info("Step \(step) | Loss: \(loss) | Reward Rate: \(totalCumulativeReward.value()[0] / (Float(step) * 1000))/s")
    }
  }
}
