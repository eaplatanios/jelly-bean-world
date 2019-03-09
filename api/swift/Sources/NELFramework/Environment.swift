import Foundation
import TensorFlow

public typealias NELEnvironmentRewardFunction = (
  _ previousItems: [Item : UInt32]?,
  _ currentItems: [Item : UInt32]?) -> Float

public struct NELEnvironmentState {
  public let scent: ShapedArray<Float>
  public let vision: ShapedArray<Float>
  public let moved: Bool
}

public enum NELEnvironmentAction: Int {
  case moveForward = 0
  case turnLeft = 1
  case turnRight = 2

  fileprivate func act(_ agent: Agent) {
    switch self {
      case .moveForward: agent.move(towards: .up, by: 1)
      case .turnLeft: agent.turn(towards: .left)
      case .turnRight: agent.turn(towards: .right)
    }
  }
}

public struct NELEnvironment {
  public let simulatorConfig: SimulatorConfig
  public let rewardFunction: NELEnvironmentRewardFunction

  private var simulator: Simulator? = nil
  private var agentDelegate: NELEnvironmentAgentDelegate? = nil
  private var agent: Agent? = nil
  private var state: NELEnvironmentState? = nil
  private var visualizer: MapVisualizer? = nil

  /// Creates a new NEL environment with an interface similar to that of OpenAI gym environments.
  public init(
    using simulatorConfig: SimulatorConfig,
    rewardFunction: @escaping NELEnvironmentRewardFunction
  ) {
    self.simulatorConfig = simulatorConfig
    self.rewardFunction = rewardFunction
    reset()
  }

  /// Runs a simulation step.
  ///
  /// - Parameters:
  ///   - action: The action to take in this step.
  ///
  /// - Returns: Reward received for the action taken.
  public mutating func step(using action: NELEnvironmentAction?) -> Float {
    let previousPosition = agent!.position
    let previousItems = agent!.items

    agentDelegate!.nextAction = action
    agent!.act()

    state = NELEnvironmentState(
      scent: agent!.scent!,
      vision: agent!.vision!,
      moved: agent!.position != previousPosition)

    return rewardFunction(previousItems, agent!.items)
  }

  /// Resets this environment to its initial state.
  public mutating func reset() {
    simulator = Simulator(using: simulatorConfig)
    agentDelegate = NELEnvironmentAgentDelegate()
    agent = Agent(in: simulator!, with: agentDelegate!)
    state = NELEnvironmentState(
      scent: agent!.scent!,
      vision: agent!.vision!,
      moved: false)
  }

  /// Renders this environment in its current state.
  public mutating func render() {
    if visualizer == nil {
      visualizer = MapVisualizer(
        for: simulator!, 
        bottomLeft: Position(x: -70, y: -70), 
        topRight: Position(x: 70, y: 70))
    }
    visualizer?.draw()
  }
}

fileprivate final class NELEnvironmentAgentDelegate : AgentDelegate {
  fileprivate var nextAction: NELEnvironmentAction? = nil

  fileprivate func act(_ agent: Agent) {
    nextAction?.act(agent)
  }

  fileprivate func save(_ agent: Agent, to file: URL) throws { }
  fileprivate func load(_ agent: Agent, from file: URL) throws { }
}
