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

/// Transition of an agent during a single simulation step.
public struct AgentTransition {
  /// State of the agent before the simulation step was performed.
  public let previousState: AgentState

  /// State of the agent after the simulation  step was performed.
  public let currentState: AgentState

  @inlinable
  public init(previousState: AgentState, currentState: AgentState) {
    self.previousState = previousState
    self.currentState = currentState
  }
}

/// Reward function that scores agent transitions.
public struct Reward {
  public let scoringFunction: (AgentTransition) -> Float

  @inlinable
  public init(using scoringFunction: @escaping (AgentTransition) -> Float) {
    self.scoringFunction = scoringFunction
  }

  /// Returns a reward value for the provided transition.
  ///
  /// - Parameter transition: Agent transition for which to compute a reward.
  /// - Returns: Reward value for the provided transition.
  @inlinable
  public func callAsFunction(for transition: AgentTransition) -> Float {
    scoringFunction(transition)
  }
}

extension Reward {
  public init(summing rewards: [Reward]) {
    self.init(using: { transition in rewards.map {
      $0(for: transition)
    }.reduce(0, +) })
  }

  public init(summing rewards: Reward...) {
    self.init(summing: rewards)
  }
}

extension Reward {
  public init(@RewardBuilder rewardBuilder: () -> Reward) {
    self = rewardBuilder()
  }
}

@_functionBuilder
public struct RewardBuilder {
  public static func buildBlock(_ rewards: Reward...) -> Reward {
    Reward(summing: rewards)
  }
}

extension Reward {
  public init(forCollecting item: Item, withValue value: Float) {
    self.init(using: { transition in
      let currentItemCount = transition.currentState.items[item] ?? 0
      let previousItemCount = transition.previousState.items[item] ?? 0
      return Float(currentItemCount - previousItemCount) * value
    })
  }
}
