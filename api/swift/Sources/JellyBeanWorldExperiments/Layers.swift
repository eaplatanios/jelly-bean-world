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
import ReinforcementLearning
import TensorFlow

public struct VisionLayer: Layer {
  public var conv1: Conv2D<Float>
  public var conv2: Conv2D<Float>
  public var dense: Dense<Float>

  public init(outputSize: Int) {
    conv1 = Conv2D<Float>(filterShape: (3, 3, 3, 16), strides: (2, 2))
    conv2 = Conv2D<Float>(filterShape: (2, 2, 16, 16), strides: (1, 1))
    dense = Dense<Float>(inputSize: 256, outputSize: outputSize)
  }

  @inlinable
  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let c1 = selu(conv1(input))
    let c2 = selu(conv2(c1)).reshaped(to: [-1, 256])
    return dense(c2)
  }
}

public struct ScentLayer: Layer {
  public var dense1: Dense<Float>
  public var dense2: Dense<Float>

  public init(outputSize: Int) {
    dense1 = Dense<Float>(inputSize: 3, outputSize: 32)
    dense2 = Dense<Float>(inputSize: 32, outputSize: outputSize)
  }

  @inlinable
  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    dense2(selu(dense1(input)))
  }
}

public struct VisionActorCritic: Module {
  public var visionLayer: VisionLayer
  public var denseAction: Dense<Float>
  public var denseValue: Dense<Float>

  public init(hiddenSize: Int = 64) {
    visionLayer = VisionLayer(outputSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
    denseValue = Dense<Float>(inputSize: hiddenSize, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: JellyBeanWorld.Environment.Observation
  ) -> ActorCriticOutput<Categorical<Int32>> {
    let outerDimCount = input.vision.rank - 3
    let outerDims = [Int](input.vision.shape.dimensions[0..<outerDimCount])
    let vision = input.vision.flattenedBatch(outerDimCount: outerDimCount)
    let hidden = leakyRelu(visionLayer(vision))
    let actionLogits = denseAction(hidden)
    let flattenedValue = denseValue(hidden)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
}

public struct ScentActorCritic: Module {
  public var scentLayer: ScentLayer
  public var denseAction: Dense<Float>
  public var denseValue: Dense<Float>

  public init(hiddenSize: Int = 64) {
    scentLayer = ScentLayer(outputSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
    denseValue = Dense<Float>(inputSize: hiddenSize, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: JellyBeanWorld.Environment.Observation
  ) -> ActorCriticOutput<Categorical<Int32>> {
    let outerDimCount = input.scent.rank - 3
    let outerDims = [Int](input.scent.shape.dimensions[0..<outerDimCount])
    let scent = input.scent.flattenedBatch(outerDimCount: outerDimCount)
    let hidden = leakyRelu(scentLayer(scent))
    let actionLogits = denseAction(hidden)
    let flattenedValue = denseValue(hidden)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
}

public struct VisionAndScentActorCritic: Module {
  public var visionLayer: VisionLayer
  public var scentLayer: ScentLayer
  public var denseAction: Dense<Float>
  public var denseValue: Dense<Float>

  public init(hiddenSize: Int = 64) {
    visionLayer = VisionLayer(outputSize: hiddenSize)
    scentLayer = ScentLayer(outputSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
    denseValue = Dense<Float>(inputSize: hiddenSize, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: JellyBeanWorld.Environment.Observation
  ) -> ActorCriticOutput<Categorical<Int32>> {
    let outerDimCount = input.vision.rank - 3
    let outerDims = [Int](input.vision.shape.dimensions[0..<outerDimCount])
    let vision = input.vision.flattenedBatch(outerDimCount: outerDimCount)
    let scent = input.scent.flattenedBatch(outerDimCount: outerDimCount)
    let visionHidden = leakyRelu(visionLayer(vision))
    let scentHidden = leakyRelu(scentLayer(scent))
    let hidden = visionHidden + scentHidden
    let actionLogits = denseAction(hidden)
    let flattenedValue = denseValue(hidden)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1))
  }
}
