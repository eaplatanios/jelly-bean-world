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
import ReinforcementLearning
import TensorFlow

public final class Environment: ReinforcementLearning.Environment {
  public let batchSize: Int
  public let configurations: [Configuration]
  public let actionSpace: Discrete
  public let observationSpace: ObservationSpace

  @usableFromInline internal var states: [State]
  @usableFromInline internal var step: Step<Observation, Tensor<Float>>

  @inlinable public var currentStep: Step<Observation, Tensor<Float>> { step }

  @inlinable
  public init(configurations: [Configuration]) {
    let batchSize = configurations.count
    self.batchSize = batchSize
    self.configurations = configurations
    self.actionSpace = Discrete(withSize: 3, batchSize: batchSize)
    self.observationSpace = ObservationSpace(batchSize: batchSize)
    self.states = configurations.map { configuration -> State in
      let simulator = Simulator(using: configuration.simulatorConfiguration)
      let agent = Agent()
      simulator.add(agent: agent)
      return State(simulator: simulator, agent: agent)
    }
    let observation = Observation.stack(states.map { state in
      let agentState = state.simulator.agentStates.values.first!
      return Observation(
        vision: Tensor<Float>(agentState.vision),
        scent: Tensor<Float>(agentState.scent),
        moved: Tensor<Bool>(repeating: false, shape: [batchSize]))
    })
    self.step =  Step(
      kind: StepKind.first(batchSize: batchSize),
      observation: observation,
      reward: Tensor<Float>(zeros: [batchSize]))
  }

  /// Updates the environment according to the provided action.
  @inlinable
  @discardableResult
  public func step(taking action: Tensor<Int32>) -> Step<Observation, Tensor<Float>> {
    let actions = action.unstacked()
    step = Step<Observation, Tensor<Float>>.stack(states.indices.map { i in
      let previousAgentState = states[i].simulator.agentStates.values.first!
      states[i].agent.nextAction = Int(actions[i].scalarized())
      states[i].simulator.step()
      let agentState = states[i].simulator.agentStates.values.first!
      let observation = Observation(
        vision: Tensor<Float>(agentState.vision),
        scent: Tensor<Float>(agentState.scent),
        moved: Tensor<Bool>(agentState.position != previousAgentState.position))
      let reward = Tensor<Float>(configurations[i].reward(for: AgentTransition(
        previousState: previousAgentState,
        currentState: agentState)))
      return Step(kind: StepKind.transition(), observation: observation, reward: reward)
    })
    return step
  }

  /// Resets the environment.
  @inlinable
  @discardableResult
  public func reset() -> Step<Observation, Tensor<Float>> {
    states = configurations.map { configuration -> State in
      let simulator = Simulator(using: configuration.simulatorConfiguration)
      let agent = Agent()
      simulator.add(agent: agent)
      return State(simulator: simulator, agent: agent)
    }
    let observation = Observation.stack(states.map { state in
      let agentState = state.simulator.agentStates.values.first!
      return Observation(
        vision: Tensor<Float>(agentState.vision),
        scent: Tensor<Float>(agentState.scent),
        moved: Tensor<Bool>(repeating: false, shape: [batchSize]))
    })
    step =  Step(
      kind: StepKind.first(batchSize: batchSize),
      observation: observation,
      reward: Tensor<Float>(zeros: [batchSize]))
    return step
  }

  /// Returns a copy of this environment that is reset before being returned.
  @inlinable
  public func copy() -> Environment {
    Environment(configurations: configurations)
  }
}

extension Environment {
  public struct Configuration {
    public let simulatorConfiguration: Simulator.Configuration
    public let reward: Reward

    @inlinable
    public init(simulatorConfiguration: Simulator.Configuration, reward: Reward) {
      self.simulatorConfiguration = simulatorConfiguration
      self.reward = reward
    }
  }
}

extension Environment {
  public struct Observation: Differentiable, KeyPathIterable {
    public var vision: Tensor<Float>
    public var scent: Tensor<Float>
    @noDerivative public var moved: Tensor<Bool>

    @inlinable
    public init(vision: Tensor<Float>, scent: Tensor<Float>, moved: Tensor<Bool>) {
      self.vision = vision
      self.scent = scent
      self.moved = moved
    }
  }
}

extension Environment {
  @usableFromInline internal struct State {
    @usableFromInline internal let simulator: Simulator
    @usableFromInline internal let agent: Agent

    @inlinable
    internal init(simulator: Simulator, agent: Agent) {
      self.simulator = simulator
      self.agent = agent
    }
  }
}

extension Environment {
  public class Agent: JellyBeanWorld.Agent {
    @usableFromInline internal var nextAction: Int? = nil
    @usableFromInline internal var state: AgentState? = nil

    @inlinable
    internal init() {}

    @inlinable
    public func act(using state: AgentState) -> Action {
      switch nextAction {
        case 0: return .move(direction: .up, stepCount: 1)
        case 1: return .turn(direction: .left)
        case 2: return .turn(direction: .right)
        case _: return .none
      }
    }
  }
}

extension Environment {
  public struct ObservationSpace: Space {
    public let distribution: ValueDistribution

    @inlinable
    public init(batchSize: Int) {
      self.distribution = ValueDistribution()
    }

    @inlinable
    public var description: String {
      "JellyBeanWorldObservationSpace"
    }

    @inlinable
    public func contains(_ value: Observation) -> Bool {
      true // TODO: Check for the range of values.
    }

    public struct ValueDistribution: DifferentiableDistribution, KeyPathIterable {
      @usableFromInline internal var visionDistribution: Uniform<Float> = Uniform<Float>(
        lowerBound: Tensor<Float>(0),
        upperBound: Tensor<Float>(1))
      // TODO: Should we limit the range of the following values?
      @usableFromInline internal var scentDistribution: Uniform<Float> = Uniform<Float>(
        lowerBound: Tensor<Float>(-Float.greatestFiniteMagnitude),
        upperBound: Tensor<Float>(Float.greatestFiniteMagnitude))
      @usableFromInline internal var movedDistribution: Bernoulli<Int32> = Bernoulli<Int32>(
        probabilities: Tensor<Float>(0.5))
      
      @inlinable
      public init() {}

      @inlinable
      @differentiable(wrt: self)
      public func logProbability(of value: Observation) -> Tensor<Float> {
        visionDistribution.logProbability(of: value.vision) +
          scentDistribution.logProbability(of: value.scent) +
          movedDistribution.logProbability(of: Tensor<Int32>(value.moved))
      }

      @inlinable
      @differentiable(wrt: self)
      public func entropy() -> Tensor<Float> {
        visionDistribution.entropy() + scentDistribution.entropy() + movedDistribution.entropy()
      }

      @inlinable
      public func mode() -> Observation {
        Observation(
          vision: visionDistribution.mode(),
          scent: scentDistribution.mode(),
          moved: movedDistribution.mode() .> 0)
      }

      @inlinable
      public func sample() -> Observation {
        Observation(
          vision: visionDistribution.sample(),
          scent: scentDistribution.sample(),
          moved: movedDistribution.sample() .> 0)
      }
    }
  }
}
