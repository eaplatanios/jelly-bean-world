// Copyright 2019-20, The Jelly Bean World Authors. All Rights Reserved.
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

import TensorFlow

/// Contextualized input to use for contextualized layers.
public struct ContextualizedInput<Input: Differentiable, Context: Differentiable>: Differentiable {
  public var input: Input
  public var context: Context

  /// Creates a new contextualized input.
  ///
  /// - Parameters:
  ///   - input: Input to a layer.
  ///   - context: Context that can be used to generate the parameters of the layer that will
  ///     be used to process `input`.
  @differentiable
  public init(input: Input, context: Context) {
    self.input = input
    self.context = context
  }
}

// TODO: [DOC] Add documentation.
public struct SelfContextualizedLayer<Base: Layer, Generator: Layer>: Layer
  where Generator.Input == Base.Input, Generator.Output == Tensor<Float> {
  @noDerivative public let base: Base
  public var generator: Generator

  public init(base: Base, generator: Generator) {
    self.base = base
    self.generator = generator
  }

  @differentiable
  public func callAsFunction(_ input: Base.Input) -> Base.Output {
    let parameters = generator(input)
    let layer = Base(unflattening: parameters, like: base)
    return layer(input)
  }
}

/// Contextualized layer. This layer consists of a generator layer and a base layer. The generator
/// layer is used to generate the parameters of the base layer which is then used to process some
/// input. The contextualized layer input consists of the base layer input and a context which is
/// fed to the generator layer.
///
/// - Source: [Contextual Parameter Generation for Universal Neural Machine Translation](
///             http://platanios.org/assets/pdf/platanios_2018_cpg_nmt/paper.pdf).
public struct ContextualizedLayer<Base: Layer, Generator: Layer>: Layer
where Generator.Output == Tensor<Float> {
  @noDerivative public let base: Base
  public var generator: Generator

  public init(base: Base, generator: Generator) {
    self.base = base
    self.generator = generator
  }

  @differentiable
  public func callAsFunction(
    _ input: ContextualizedInput<Base.Input, Generator.Input>
  ) -> Base.Output {
    let parameters = generator(input.context)
    let layer = Base(unflattening: parameters, like: base)
    return layer(input.input)
  }
}

extension KeyPathIterable {
  /// The number of parameters of this module. Note that only `Tensor<Float>`-valued parameters are
  /// being accounted for at this point.
  public var parameterCount: Int {
    var count = 0
    for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      count += self[keyPath: kp].shape.contiguousSize
    }
    return count
  }
}

// TODO: Support more than just Float tensors for contextualization.
extension Module {
  /// Creates a new module by initializing all its parameters to the values contained in the
  /// provided flattened tensor. The rest of its properties are initialized based on `other`.
  ///
  /// - Parameters:
  ///   - flattened: Flattened tensor containing all parameters of this module. Note that this
  ///     tensor has a dimension of size 1 that corresponds to a data batch. The same parameters
  ///     will be used for all data examples in each batch.
  ///   - other: Another module to use as template when initializing the new one, for the
  ///     properties which are not included in `flattened`.
  public init(unflattening flattened: Tensor<Float>, like other: Self) {
    let batchSize = flattened.shape[0]
    var newLayer = other
    var index = 0
    for kp in other.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      let shape = other[keyPath: kp].shape
      newLayer[keyPath: kp] = flattened[0..., index..<index + shape.contiguousSize]
      newLayer[keyPath: kp] = newLayer[keyPath: kp]
        .reshaped(to: batchSize > 1 ?
      TensorShape([batchSize] + shape.dimensions) :
      shape)
      index += shape.contiguousSize
    }
    self = newLayer
  }

  @inlinable
  @derivative(of: init(unflattening:like:), wrt: flattened)
  internal static func _vjpInit(unflattening flattened: Tensor<Float>, like other: Self) -> (
    value: Self,
    pullback: (TangentVector) -> Tensor<Float>
  ) {
    let batchSize = flattened.shape[0]
    return (.init(unflattening: flattened, like: other), { v in
      var values = [Tensor<Float>]()
      for kp in v.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
        values.append(v[keyPath: kp].reshaped(to: [batchSize, -1]))
      }
      if values.count > 1 {
        return Tensor(concatenating: values, alongAxis: -1)
      } else {
        return values[0]
      }
    })
  }

  /// Returns a flattened view of this layer. This corresponds to taking all the parameters of this
  /// layer, flattening them, and finally stacking them to form a single vector.
  public func flattened() -> Tensor<Float> {
    var parameters = [Tensor<Float>]()
    for kp in recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
      parameters.append(self[keyPath: kp].flattened())
    }
    return Tensor<Float>(concatenating: parameters, alongAxis: 0).rankLifted()
  }

  @inlinable
  @derivative(of: flattened(), wrt: self)
  internal func _vjpFlattened() -> (
    value: Tensor<Float>,
    pullback: (Tensor<Float>) -> TangentVector
  ) {
    (flattened(), { v in
      let batchSize = v.shape[0]
      var newLayer = self
      var index = 0
      for kp in self.recursivelyAllWritableKeyPaths(to: Tensor<Float>.self) {
        let shape = self[keyPath: kp].shape
        newLayer[keyPath: kp] = v[0..., index..<index + shape.contiguousSize]
        newLayer[keyPath: kp] = newLayer[keyPath: kp]
          .reshaped(to: batchSize > 1 ?
        TensorShape([batchSize] + shape.dimensions) :
        shape)
        index += shape.contiguousSize
      }
      return self.differentiableVectorView
    })
  }
}
