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

  @inlinable
  public init(outputSize: Int) {
    conv1 = Conv2D<Float>(filterShape: (3, 3, 3, 16), strides: (2, 2))
    conv2 = Conv2D<Float>(filterShape: (2, 2, 16, 16), strides: (1, 1))
    dense = Dense<Float>(inputSize: rebuttal ? 256 : 784, outputSize: outputSize)
  }

  @inlinable
  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    let c1 = gelu(conv1(input))
    let c2 = gelu(conv2(c1)).reshaped(to: [-1, rebuttal ? 256 : 784])
    return dense(c2)
  }
}

public struct ScentLayer: Layer {
  public var dense1: Dense<Float>
  public var dense2: Dense<Float>

  @inlinable
  public init(outputSize: Int) {
    dense1 = Dense<Float>(inputSize: 3, outputSize: 32)
    dense2 = Dense<Float>(inputSize: 32, outputSize: outputSize)
  }

  @inlinable
  @differentiable
  public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
    dense2(gelu(dense1(input)))
  }
}

public struct LSTMCell<Scalar: TensorFlowFloatingPoint>: Layer {
  public var fusedWeight: Tensor<Scalar>
  public var fusedBias: Tensor<Scalar>

  @inlinable
  public func stateShape(batchSize: Int) -> TensorShape {
    TensorShape([batchSize, fusedWeight.shape[1] / 4])
  }

  @inlinable
  public func zeroState(batchSize: Int) -> State {
    let stateShape = self.stateShape(batchSize: batchSize)
    return State(cell: Tensor(zeros: stateShape), hidden: Tensor(zeros: stateShape))
  }

  @inlinable
  public init(inputSize: Int, hiddenSize: Int) {
    self.fusedWeight = Tensor(glorotUniform: [inputSize + hiddenSize, 4 * hiddenSize])
    self.fusedBias = Tensor(zeros: [4 * hiddenSize])
  }

  @inlinable
  @differentiable
  public func callAsFunction(_ input: Value) -> Value {
    let gateInput = input.value.concatenated(with: input.state.hidden, alongAxis: 1)
    let fused = matmul(gateInput, fusedWeight) + fusedBias
    let batchSize = fused.shape[0]
    let hiddenSize = fused.shape[1] / 4
    let inputGate = sigmoid(fused.slice(
      lowerBounds: [0, 0],
      upperBounds: [batchSize, hiddenSize]))
    let updateGate = tanh(fused.slice(
      lowerBounds: [0, hiddenSize],
      upperBounds: [batchSize, 2 * hiddenSize]))
    let forgetGate = sigmoid(fused.slice(
      lowerBounds: [0, 2 * hiddenSize],
      upperBounds: [batchSize, 3 * hiddenSize]))
    let outputGate = sigmoid(fused.slice(
      lowerBounds: [0, 3 * hiddenSize],
      upperBounds: [batchSize,4 * hiddenSize]))
    // TODO(SR-10697/TF-507): Replace with the following once it does not crash the compiler.
    // let fusedParts = fused.split(count: 4, alongAxis: 1)
    // let inputGate = sigmoid(fusedParts[0])
    // let updateGate = tanh(fusedParts[1])
    // let forgetGate = sigmoid(fusedParts[2])
    // let outputGate = sigmoid(fusedParts[3])
    let newCellState = input.state.cell * forgetGate + inputGate * updateGate
    let newHiddenState = tanh(newCellState) * outputGate
    let newState = LSTMCell<Scalar>.State(cell: newCellState, hidden: newHiddenState)
    return Value(value: newCellState, state: newState)
  }

  public struct Value: Differentiable, KeyPathIterable {
    public var value: Tensor<Scalar>
    public var state: State

    @inlinable
    @differentiable
    public init(value: Tensor<Scalar>, state: State) {
      self.value = value
      self.state = state
    }
  }

  public struct State: Differentiable, KeyPathIterable {
    public var cell: Tensor<Scalar>
    public var hidden: Tensor<Scalar>

    @inlinable
    @differentiable
    public init(cell: Tensor<Scalar>, hidden: Tensor<Scalar>) {
      self.cell = cell
      self.hidden = hidden
    }
  }
}

public struct VisionActorCritic: Module {
  public var visionLayer: VisionLayer
  public var hiddenLSTMCell: LSTMCell<Float>
  public var denseAction: Dense<Float>
  public var denseValue: Dense<Float>

  @inlinable
  public func initialState(batchSize: Int) -> LSTMCell<Float>.State {
    hiddenLSTMCell.zeroState(batchSize: batchSize)
  }

  @inlinable
  public init(hiddenSize: Int = 512) {
    visionLayer = VisionLayer(outputSize: hiddenSize)
    hiddenLSTMCell = LSTMCell<Float>(inputSize: hiddenSize + 1, hiddenSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: 2 * hiddenSize + 1, outputSize: 3)
    denseValue = Dense<Float>(inputSize: 2 * hiddenSize + 1, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<JellyBeanWorld.Environment.Observation, LSTMCell<Float>.State>
  ) -> ActorCriticOutput<Categorical<Int32>, LSTMCell<Float>.State> {
    let observation = input.observation
    let outerDimCount = observation.vision.rank - 3
    let outerDims = [Int](observation.vision.shape.dimensions[0..<outerDimCount])
    let vision = observation.vision.flattenedBatch(outerDimCount: outerDimCount)
    let moved = 2 * observation.moved.flattenedBatch(
      outerDimCount: outerDimCount
    ).expandingShape(at: -1) - 1
    let hidden = gelu(visionLayer(vision)).concatenated(with: moved, alongAxis: -1)
    let state = withoutDerivative(at: input.state).flattenedBatch(outerDimCount: outerDimCount)
    let hiddenLSTMOutput = hiddenLSTMCell(LSTMCell<Float>.Value(value: hidden, state: state))
    let hiddenConcatenated = hidden.concatenated(with: hiddenLSTMOutput.value, alongAxis: -1)
    let actionLogits = denseAction(hiddenConcatenated)
    let flattenedValue = denseValue(hiddenConcatenated)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1),
      state: withoutDerivative(at: hiddenLSTMOutput.state).unflattenedBatch(outerDims: outerDims))
  }
}

public struct ScentActorCritic: Module {
  public var scentLayer: ScentLayer
  public var hiddenLSTMCell: LSTMCell<Float>
  public var denseAction: Dense<Float>
  public var denseValue: Dense<Float>

  @inlinable
  public func initialState(batchSize: Int) -> LSTMCell<Float>.State {
    hiddenLSTMCell.zeroState(batchSize: batchSize)
  }

  @inlinable
  public init(hiddenSize: Int = 512) {
    scentLayer = ScentLayer(outputSize: hiddenSize)
    hiddenLSTMCell = LSTMCell<Float>(inputSize: hiddenSize + 1, hiddenSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: 2 * hiddenSize + 1, outputSize: 3)
    denseValue = Dense<Float>(inputSize: 2 * hiddenSize + 1, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<JellyBeanWorld.Environment.Observation, LSTMCell<Float>.State>
  ) -> ActorCriticOutput<Categorical<Int32>, LSTMCell<Float>.State> {
    let observation = input.observation
    let outerDimCount = observation.vision.rank - 3
    let outerDims = [Int](observation.vision.shape.dimensions[0..<outerDimCount])
    let scent = observation.scent.flattenedBatch(outerDimCount: outerDimCount)
    let moved = 2 * observation.moved.flattenedBatch(
      outerDimCount: outerDimCount
    ).expandingShape(at: -1) - 1
    let hidden = gelu(scentLayer(scent)).concatenated(with: moved, alongAxis: -1)
    let state = withoutDerivative(at: input.state).flattenedBatch(outerDimCount: outerDimCount)
    let hiddenLSTMOutput = hiddenLSTMCell(LSTMCell<Float>.Value(value: hidden, state: state))
    let hiddenConcatenated = hidden.concatenated(with: hiddenLSTMOutput.value, alongAxis: -1)
    let actionLogits = denseAction(hiddenConcatenated)
    let flattenedValue = denseValue(hiddenConcatenated)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1),
      state: withoutDerivative(at: hiddenLSTMOutput.state).unflattenedBatch(outerDims: outerDims))
  }
}

public struct VisionAndScentActorCritic: Module {
  public var visionLayer: VisionLayer
  public var scentLayer: ScentLayer
  public var hiddenLSTMCell: LSTMCell<Float>
  public var denseAction: Dense<Float>
  public var denseValue: Dense<Float>

  @inlinable
  public func initialState(batchSize: Int) -> LSTMCell<Float>.State {
    hiddenLSTMCell.zeroState(batchSize: batchSize)
  }

  @inlinable
  public init(hiddenSize: Int = 512) {
    visionLayer = VisionLayer(outputSize: hiddenSize)
    scentLayer = ScentLayer(outputSize: hiddenSize)
    hiddenLSTMCell = LSTMCell<Float>(inputSize: 2 * hiddenSize + 1, hiddenSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: 3 * hiddenSize + 1, outputSize: 3)
    denseValue = Dense<Float>(inputSize: 3 * hiddenSize + 1, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<JellyBeanWorld.Environment.Observation, LSTMCell<Float>.State>
  ) -> ActorCriticOutput<Categorical<Int32>, LSTMCell<Float>.State> {
    let observation = input.observation
    let outerDimCount = observation.vision.rank - 3
    let outerDims = [Int](observation.vision.shape.dimensions[0..<outerDimCount])
    let vision = observation.vision.flattenedBatch(outerDimCount: outerDimCount)
    let scent = observation.scent.flattenedBatch(outerDimCount: outerDimCount)
    let visionHidden = gelu(visionLayer(vision))
    let scentHidden = gelu(scentLayer(scent))
    let moved = gelu(2 * observation.moved.flattenedBatch(
      outerDimCount: outerDimCount
    ).expandingShape(at: -1) - 1)
    let hidden = visionHidden
      .concatenated(with: scentHidden, alongAxis: -1)
      .concatenated(with: moved, alongAxis: -1)
    let state = withoutDerivative(at: input.state).flattenedBatch(outerDimCount: outerDimCount)
    let hiddenLSTMOutput = hiddenLSTMCell(LSTMCell<Float>.Value(value: hidden, state: state))
    let hiddenConcatenated = hidden.concatenated(with: hiddenLSTMOutput.value, alongAxis: -1)
    let actionLogits = denseAction(hiddenConcatenated)
    let flattenedValue = denseValue(hiddenConcatenated)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1),
      state: withoutDerivative(at: hiddenLSTMOutput.state).unflattenedBatch(outerDims: outerDims))
  }
}

public struct RewardCompiler: Module {
  @noDerivative public let simulatorConfiguration: Simulator.Configuration
  @noDerivative public let embeddingSize: Int

  public var zeroEmbedding: Tensor<Float>
  public var actionEmbedding: Tensor<Float>
  public var collectLayer: Dense<Float>
  public var exploreEmbedding: Tensor<Float>
  public var itemEmbeddings: Tensor<Float>

  @inlinable
  public init(simulatorConfiguration: Simulator.Configuration, embeddingSize: Int = 8) {
    self.simulatorConfiguration = simulatorConfiguration
    self.embeddingSize = embeddingSize
    self.zeroEmbedding = Tensor<Float>(randomNormal: [embeddingSize])
    self.actionEmbedding = Tensor<Float>(randomNormal: [embeddingSize])
    self.collectLayer = Dense<Float>(inputSize: embeddingSize, outputSize: embeddingSize)
    self.exploreEmbedding = Tensor<Float>(randomNormal: [embeddingSize])
    self.itemEmbeddings = Tensor<Float>(
      randomNormal: [simulatorConfiguration.items.count, embeddingSize])
  }

  @inlinable
  @differentiable
  public func callAsFunction(_ input: JellyBeanWorld.Reward) -> Tensor<Float> {
    switch input {
    case .zero:
      return zeroEmbedding
    case let .action(value):
      return gelu(actionEmbedding * value)
    case let .collect(item, value):
      let itemEmbedding = itemEmbeddings[simulatorConfiguration.items.firstIndex(of: item)!]
      let embedding = gelu(collectLayer(itemEmbedding.expandingShape(at: 0)) * value)
      return embedding.squeezingShape(at: 0)
    case let .avoid(item, value):
      let itemEmbedding = itemEmbeddings[simulatorConfiguration.items.firstIndex(of: item)!]
      let embedding = gelu(collectLayer(itemEmbedding.expandingShape(at: 0)) * -value)
      return embedding.squeezingShape(at: 0)
    case let .explore(value):
      return gelu(exploreEmbedding * value)
    case let .combined(reward1, reward2):
      return gelu(self(reward1) + self(reward2))
    }
  }
}

public struct RewardAwareVisionAndScentActorCritic: Module {
  public var rewardCompiler: RewardCompiler
  public var visionLayer: VisionLayer
  public var scentLayer: ScentLayer
  public var hiddenLSTMCell: LSTMCell<Float>
  public var denseAction: Dense<Float>
  public var denseValue: Dense<Float>

  @inlinable
  public func initialState(batchSize: Int) -> LSTMCell<Float>.State {
    hiddenLSTMCell.zeroState(batchSize: batchSize)
  }

  @inlinable
  public init(
    simulatorConfiguration: Simulator.Configuration,
    hiddenSize: Int = 512,
    rewardEmbeddingSize: Int = 8
  ) {
    rewardCompiler = RewardCompiler(
      simulatorConfiguration: simulatorConfiguration,
      embeddingSize: rewardEmbeddingSize)
    visionLayer = VisionLayer(outputSize: hiddenSize)
    scentLayer = ScentLayer(outputSize: hiddenSize)
    hiddenLSTMCell = LSTMCell<Float>(
      inputSize: 2 * hiddenSize + 1 + rewardEmbeddingSize,
      hiddenSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: 3 * hiddenSize + 1 + rewardEmbeddingSize, outputSize: 3)
    denseValue = Dense<Float>(inputSize: 3 * hiddenSize + 1 + rewardEmbeddingSize, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<JellyBeanWorld.Environment.Observation, LSTMCell<Float>.State>
  ) -> ActorCriticOutput<Categorical<Int32>, LSTMCell<Float>.State> {
    let observation = input.observation
    let outerDimCount = observation.vision.rank - 3
    let outerDims = [Int](observation.vision.shape.dimensions[0..<outerDimCount])
    let reward = rewardCompiler(observation.rewardFunction)
    let vision = observation.vision.flattenedBatch(outerDimCount: outerDimCount)
    let scent = observation.scent.flattenedBatch(outerDimCount: outerDimCount)
    let rewardHidden = reward.expandingShape(at: 0).tiled(
      multiples: Tensor<Int32>([Int32(vision.shape[0]), 1]))
    let visionHidden = gelu(visionLayer(vision))
    let scentHidden = gelu(scentLayer(scent))
    let moved = gelu(2 * observation.moved.flattenedBatch(
      outerDimCount: outerDimCount
    ).expandingShape(at: -1) - 1)
    let hidden = rewardHidden
      .concatenated(with: visionHidden, alongAxis: -1)
      .concatenated(with: scentHidden, alongAxis: -1)
      .concatenated(with: moved, alongAxis: -1)
    let state = withoutDerivative(at: input.state).flattenedBatch(outerDimCount: outerDimCount)
    let hiddenLSTMOutput = hiddenLSTMCell(LSTMCell<Float>.Value(value: hidden, state: state))
    let hiddenConcatenated = hidden.concatenated(with: hiddenLSTMOutput.value, alongAxis: -1)
    let actionLogits = denseAction(hiddenConcatenated)
    let flattenedValue = denseValue(hiddenConcatenated)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1),
      state: withoutDerivative(at: hiddenLSTMOutput.state).unflattenedBatch(outerDims: outerDims))
  }
}
