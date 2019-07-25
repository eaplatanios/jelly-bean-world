import CNELFramework
import Foundation
import TensorFlow

public protocol Agent {
  /// Returns the agent's desired next action.
  ///
  /// - Parameter state: Current state of the agent.
  /// - Returns: Action requested by the agent.
  ///
  /// - Note: The action is not actually performed until the simulator advances by a time step and
  ///   issues a notification about that event. The simulator only advances by a time step only
  ///   once all agents have requested some action.
  mutating func act(using state: AgentState) -> Action
  func save(to file: URL) throws
  mutating func load(from file: URL) throws
}

/// State of an agent in the jelly bean world.
public struct AgentState {
  public let position: Position
  public let direction: Direction
  public let scent: Tensor<Float>
  public let vision: Tensor<Float>
  public let items: [Item: Int]

  @inlinable
  internal init(fromC value: AgentSimulationState, for simulator: Simulator) {
    self.position = Position(fromC: value.position)
    self.direction = Direction(fromC: value.direction)

    // Update scent.
    let scentShape = [Int(simulator.configuration.scentDimSize)]
    let scentBuffer = UnsafeBufferPointer(start: value.scent!, count: scentShape[0])
    self.scent = Tensor(shape: TensorShape(scentShape), scalars: [Float](scentBuffer))

    // Update vision.
    let visionShape = [
      2 * Int(simulator.configuration.visionRange) + 1,
      2 * Int(simulator.configuration.visionRange) + 1,
      Int(simulator.configuration.colorDimSize)]
    let visionSize = Int(
      (2 * simulator.configuration.visionRange + 1) *
      (2 * simulator.configuration.visionRange + 1) *
      simulator.configuration.colorDimSize)
    let visionBuffer = UnsafeBufferPointer(start: value.vision!, count: visionSize)
    self.vision = Tensor(shape: TensorShape(visionShape), scalars: [Float](visionBuffer))

    // Update items.
    let simulatorItems = simulator.configuration.items
    self.items = [Item: Int](uniqueKeysWithValues: zip(
      simulatorItems,
      UnsafeBufferPointer(
        start: value.collectedItems!,
        count: simulatorItems.count).map(Int.init)))
  }
}

/// Action that can be taken by agents in the jelly bean world.
public enum Action {
  /// No action.
  case none

  /// Move action, along the specified direction and for the provided number of steps.
  case move(direction: Direction, stepCount: Int = 1)

  /// Turn action (without any movement to a different cell).
  case turn(direction: TurnDirection)

  @inlinable
  internal func invoke(
    simulatorHandle: UnsafeMutableRawPointer?,
    clientHandle: UnsafeMutableRawPointer?,
    agentID: UInt64
  ) {
    switch self {
    case .none:
      simulatorNoOpAgent(simulatorHandle, clientHandle, agentID)
    case let .move(direction, stepCount):
      simulatorMoveAgent(
        simulatorHandle,
        clientHandle,
        agentID,
        direction.toC(),
        UInt32(stepCount))
    case let .turn(direction):
      simulatorTurnAgent(simulatorHandle, clientHandle, agentID, direction.toC())
    }
  }
}
