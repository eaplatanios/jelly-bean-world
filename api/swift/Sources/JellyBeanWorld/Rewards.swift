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

///  Reward function that scores agent transitions.
public enum Reward: Equatable {
  case zero
  case action(value: Float)
  case collect(item: Item, value: Float)
  case avoid(item: Item, value: Float)
  case explore(value: Float)
  indirect case combined(Reward, Reward)

  /// Adds two reward functions. The resulting reward will be equal to the sum of the rewards
  /// computed by the two functions.
  @inlinable
  public static func +(lhs: Reward, rhs: Reward) -> Reward {
    .combined(lhs, rhs)
  }

  /// Returns a reward value for the provided transition.
  ///
  /// - Parameter transition: Agent transition for which to compute a reward.
  /// - Returns: Reward value for the provided transition.
  @inlinable
  public func callAsFunction(for transition: AgentTransition) -> Float {
    switch self {
    case .zero:
      return 0
    case let .action(value):
      return value
    case let .collect(item, value):
      let currentItemCount = transition.currentState.items[item] ?? 0
      let previousItemCount = transition.previousState.items[item] ?? 0
      return Float(currentItemCount - previousItemCount) * value
    case let .avoid(item, value):
      let currentItemCount = transition.currentState.items[item] ?? 0
      let previousItemCount = transition.previousState.items[item] ?? 0
      return Float(previousItemCount - currentItemCount) * value
    case let .explore(value):
      let x = Float(transition.currentState.position.x)
      let y = Float(transition.currentState.position.y)
      let previousX = Float(transition.previousState.position.x)
      let previousY = Float(transition.previousState.position.y)
      let distance = x * x + y * y
      let previousDistance = previousX * previousX + previousY * previousY
      return distance > previousDistance ? value : 0.0
    case let .combined(reward1, reward2):
      return reward1(for: transition) + reward2(for: transition)
    }
  }
}

extension Reward: CustomStringConvertible {
  public var description: String {
    switch self {
    case .zero:
      return "Zero"
    case let .action(value):
      return "Action[\(String(format: "%.2f", value))]"
    case let .collect(item, value):
      return "Collect[\(item.description), \(String(format: "%.2f", value))]"
    case let .avoid(item, value):
      return "Avoid[\(item.description), \(String(format: "%.2f", value))]"
    case let .explore(value):
      return "Explore[\(String(format: "%.2f", value))]"
    case let .combined(reward1, reward2):
      return "\(reward1.description) ∧ \(reward2.description)"
    }
  }
}

/// Reward function schedule which specifies which reward function is used at each time step.
/// This is useful for representing never-ending learning settings that require adaptation.
public protocol RewardSchedule {
  /// Returns the reward function to use for the specified time step.
  func reward(forStep step: UInt64) -> Reward
}

/// Fixed reward function schedule that uses the same reward function for all time steps.
public struct FixedReward: RewardSchedule {
  public let reward: Reward

  public init(_ reward: Reward) {
    self.reward = reward
  }

  public func reward(forStep step: UInt64) -> Reward {
    reward
  }
}

public struct CyclicalSchedule: RewardSchedule {
  public let rewards: [(Reward, UInt64)]
  public let cycleDuration: UInt64

  public init(_ rewards: [(Reward, UInt64)]) {
    precondition(!rewards.isEmpty)
    self.rewards = rewards
    self.cycleDuration = rewards.map { $0.1 }.reduce(0, +)
  }

  public func reward(forStep step: UInt64) -> Reward {
    let step = step % cycleDuration
    var cumulativeDuration: UInt64 = 0
    for reward in rewards {
      if step < cumulativeDuration + reward.1 { return reward.0 }
      cumulativeDuration += reward.1
    }
    return rewards.last!.0
  }
}
