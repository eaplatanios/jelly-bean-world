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

fileprivate struct JellyBeanWorldActorCritic: Module {
  public var conv1: Conv2D<Float> = Conv2D<Float>(filterShape: (3, 3, 3, 16), strides: (2, 2))
  public var conv2: Conv2D<Float> = Conv2D<Float>(filterShape: (2, 2, 16, 16), strides: (1, 1))
  public var denseHidden: Dense<Float> = Dense<Float>(inputSize: 256, outputSize: 64)
  // public var denseScent1: Dense<Float> = Dense<Float>(inputSize: 3, outputSize: 32)
  // public var denseScent2: Dense<Float> = Dense<Float>(inputSize: 32, outputSize: 4)
  public var denseAction: Dense<Float> = Dense<Float>(inputSize: 64, outputSize: 3)
  public var denseValue: Dense<Float> = Dense<Float>(inputSize: 64, outputSize: 1)

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: JellyBeanWorld.Environment.Observation
  ) -> ActorCriticOutput<Categorical<Int32>> {
    let outerDimCount = input.vision.rank - 3
    let outerDims = [Int](input.vision.shape.dimensions[0..<outerDimCount])
    // let flattenedBatchInput = input.flattenedBatch(outerDimCount: outerDimCount)
    // let vision = flattenedBatchInput.vision
    // let scent = flattenedBatchInput.scent
    let vision = input.vision.flattenedBatch(outerDimCount: outerDimCount)
    let conv1 = leakyRelu(self.conv1(vision))
    let conv2 = leakyRelu(self.conv2(conv1)).reshaped(to: [-1, 256])
    let visionHidden = leakyRelu(denseHidden(conv2))
    // let scent1 = leakyRelu(denseScent1(scent))
    // let scentHidden = leakyRelu(denseScent2(scent1))
    let hidden = visionHidden // + scentHidden
    let actionLogits = denseAction(hidden)
    let flattenedValue = denseValue(hidden)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
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
  let environment = try JellyBeanWorld.Environment(configurations: configurations)
  let totalCumulativeReward = TotalCumulativeReward(for: environment)

  let network = JellyBeanWorldActorCritic()
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
      using: environment,
      maxSteps: 128 * batchSize,
      stepCallbacks: [{ (environment, trajectory) in
        totalCumulativeReward.update()
        // if step > 140 {
        //   painter.draw()
        // }
      }])
    if step % 10 == 0 {
      logger.info("Step \(step) | Loss: \(loss) | Reward Rate: \(totalCumulativeReward.value()[0] / (Float(step) * 1000))/s")
    }
  }
}
