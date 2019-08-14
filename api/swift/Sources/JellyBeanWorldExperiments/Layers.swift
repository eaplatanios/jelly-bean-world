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

public struct LSTMCell<Scalar: TensorFlowFloatingPoint>: Layer {
  public var inputWeight, updateWeight, forgetWeight, outputWeight: Tensor<Scalar>
  public var inputBias, updateBias, forgetBias, outputBias: Tensor<Scalar>

  public func stateShape(batchSize: Int) -> TensorShape {
    TensorShape([batchSize, inputWeight.shape[1]])
  }

  public func zeroState(batchSize: Int) -> LSTMState<Scalar> {
    let stateShape = self.stateShape(batchSize: batchSize)
    return LSTMState(cell: Tensor(zeros: stateShape), hidden: Tensor(zeros: stateShape))
  }

  public init(
    inputSize: Int,
    hiddenSize: Int,
    seed: TensorFlowSeed = Context.local.randomSeed
  ) {
    let concatenatedInputSize = inputSize + hiddenSize
    let gateWeightShape = TensorShape([concatenatedInputSize, hiddenSize])
    let gateBiasShape = TensorShape([hiddenSize])
    self.inputWeight = Tensor(glorotUniform: gateWeightShape, seed: seed)
    self.inputBias = Tensor(zeros: gateBiasShape)
    self.updateWeight = Tensor(glorotUniform: gateWeightShape, seed: seed)
    self.updateBias = Tensor(zeros: gateBiasShape)
    self.forgetWeight = Tensor(glorotUniform: gateWeightShape, seed: seed)
    self.forgetBias = Tensor(ones: gateBiasShape)
    self.outputWeight = Tensor(glorotUniform: gateWeightShape, seed: seed)
    self.outputBias = Tensor(zeros: gateBiasShape)
  }

  @differentiable
  public func callAsFunction(_ input: LSTMValue<Scalar>) -> LSTMValue<Scalar> {
    let gateInput = input.value.concatenated(with: input.state.hidden, alongAxis: 1)
    let inputGate = sigmoid(matmul(gateInput, inputWeight) + inputBias)
    let updateGate = tanh(matmul(gateInput, updateWeight) + updateBias)
    let forgetGate = sigmoid(matmul(gateInput, forgetWeight) + forgetBias)
    let outputGate = sigmoid(matmul(gateInput, outputWeight) + outputBias)
    let newCellState = input.state.cell * forgetGate + inputGate * updateGate
    let newHiddenState = tanh(newCellState) * outputGate
    let newState = LSTMState(cell: newCellState, hidden: newHiddenState)
    return LSTMValue(value: newCellState, state: newState)
  }
}

public struct LSTMValue<Scalar: TensorFlowFloatingPoint>: Differentiable, KeyPathIterable {
  public var value: Tensor<Scalar>
  public var state: LSTMState<Scalar>

  @inlinable
  @differentiable
  public init(value: Tensor<Scalar>, state: LSTMState<Scalar>) {
    self.value = value
    self.state = state
  }
}

public struct LSTMState<Scalar: TensorFlowFloatingPoint>: Differentiable, KeyPathIterable {
  public var cell: Tensor<Scalar>
  public var hidden: Tensor<Scalar>

  @inlinable
  @differentiable
  public init(cell: Tensor<Scalar>, hidden: Tensor<Scalar>) {
    self.cell = cell
    self.hidden = hidden
  }
}

public struct VisionActorCritic: Module {
  public var visionLayer: VisionLayer
  public var hiddenLSTMCell: LSTMCell<Float>
  public var denseAction: Dense<Float>
  public var denseValue: Dense<Float>

  public func initialState(batchSize: Int) -> LSTMState<Float> {
    hiddenLSTMCell.zeroState(batchSize: batchSize)
  }

  public init(hiddenSize: Int = 64) {
    visionLayer = VisionLayer(outputSize: hiddenSize)
    hiddenLSTMCell = LSTMCell<Float>(inputSize: hiddenSize, hiddenSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
    denseValue = Dense<Float>(inputSize: hiddenSize, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<JellyBeanWorld.Environment.Observation, LSTMState<Float>>
  ) -> ActorCriticOutput<Categorical<Int32>, LSTMState<Float>> {
    let observation = input.observation
    let outerDimCount = observation.vision.rank - 3
    let outerDims = [Int](observation.vision.shape.dimensions[0..<outerDimCount])
    let vision = observation.vision.flattenedBatch(outerDimCount: outerDimCount)
    let hidden = selu(visionLayer(vision))
    let state = withoutDerivative(at: input.state).flattenedBatch(outerDimCount: outerDimCount)
    let hiddenLSTMOutput = hiddenLSTMCell(LSTMValue<Float>(value: hidden, state: state))
    let actionLogits = denseAction(hiddenLSTMOutput.value)
    let flattenedValue = denseValue(hiddenLSTMOutput.value)
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

  public func initialState(batchSize: Int) -> LSTMState<Float> {
    hiddenLSTMCell.zeroState(batchSize: batchSize)
  }

  public init(hiddenSize: Int = 64) {
    scentLayer = ScentLayer(outputSize: hiddenSize)
    hiddenLSTMCell = LSTMCell<Float>(inputSize: hiddenSize, hiddenSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
    denseValue = Dense<Float>(inputSize: hiddenSize, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<JellyBeanWorld.Environment.Observation, LSTMState<Float>>
  ) -> ActorCriticOutput<Categorical<Int32>, LSTMState<Float>> {
    let observation = input.observation
    let outerDimCount = observation.vision.rank - 3
    let outerDims = [Int](observation.vision.shape.dimensions[0..<outerDimCount])
    let scent = observation.scent.flattenedBatch(outerDimCount: outerDimCount)
    let hidden = selu(scentLayer(scent))
    let state = withoutDerivative(at: input.state).flattenedBatch(outerDimCount: outerDimCount)
    let hiddenLSTMOutput = hiddenLSTMCell(LSTMValue<Float>(value: hidden, state: state))
    let actionLogits = denseAction(hiddenLSTMOutput.value)
    let flattenedValue = denseValue(hiddenLSTMOutput.value)
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

  public func initialState(batchSize: Int) -> LSTMState<Float> {
    hiddenLSTMCell.zeroState(batchSize: batchSize)
  }

  public init(hiddenSize: Int = 64) {
    visionLayer = VisionLayer(outputSize: hiddenSize)
    scentLayer = ScentLayer(outputSize: hiddenSize)
    hiddenLSTMCell = LSTMCell<Float>(inputSize: hiddenSize, hiddenSize: hiddenSize)
    denseAction = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
    denseValue = Dense<Float>(inputSize: hiddenSize, outputSize: 1)
  }

  @inlinable
  @differentiable
  public func callAsFunction(
    _ input: AgentInput<JellyBeanWorld.Environment.Observation, LSTMState<Float>>
  ) -> ActorCriticOutput<Categorical<Int32>, LSTMState<Float>> {
    let observation = input.observation
    let outerDimCount = observation.vision.rank - 3
    let outerDims = [Int](observation.vision.shape.dimensions[0..<outerDimCount])
    let vision = observation.vision.flattenedBatch(outerDimCount: outerDimCount)
    let scent = observation.scent.flattenedBatch(outerDimCount: outerDimCount)
    let visionHidden = selu(visionLayer(vision))
    let scentHidden = selu(scentLayer(scent))
    let hidden = visionHidden + scentHidden
    let state = withoutDerivative(at: input.state).flattenedBatch(outerDimCount: outerDimCount)
    let hiddenLSTMOutput = hiddenLSTMCell(LSTMValue<Float>(value: hidden, state: state))
    let actionLogits = denseAction(hiddenLSTMOutput.value)
    let flattenedValue = denseValue(hiddenLSTMOutput.value)
    let flattenedActionDistribution = Categorical<Int32>(logits: actionLogits)
    return ActorCriticOutput(
      actionDistribution: flattenedActionDistribution.unflattenedBatch(outerDims: outerDims),
      value: flattenedValue.unflattenedBatch(outerDims: outerDims).squeezingShape(at: -1),
      state: withoutDerivative(at: hiddenLSTMOutput.state).unflattenedBatch(outerDims: outerDims))
  }
}
