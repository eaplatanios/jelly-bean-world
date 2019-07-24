import CNELFramework
import Foundation
import TensorFlow

open class Agent {
  public final let simulator: Simulator

  public internal(set) final var id: UInt64? = nil
  public internal(set) final var position: Position? = nil
  public internal(set) final var direction: Direction? = nil
  
  public final var scent: ShapedArray<Float>?
  public final var vision: ShapedArray<Float>?
  public final var items: [Item: UInt32]?

  @inlinable
  public init(in simulator: Simulator) {
    self.simulator = simulator
    simulator.addAgent(self)
  }

  @inlinable
  open func act() {}

  @inlinable
  open func save(to file: URL) throws {}

  @inlinable
  open func load(from file: URL) throws {}

  /// Moves this agent in the simulated environment.
  ///
  /// - Note: The agent is not moved until the simulator advances by a time step and issues a
  ///   notification about that event. The simulator only advances the time step once all agents 
  ///   have requested to move.
  @inlinable
  @discardableResult
  public final func move(towards direction: Direction, by numSteps: UInt32 = 1) -> Bool {
    simulator.moveAgent(agent: self, towards: direction, by: numSteps)
  }
  
  /// Turns this agent in the simulated environment.
  ///
  /// - Note: The agent is not turned until the simulator advances by a time step and issues a
  ///   notification about that event. The simulator only advances the time step once all agents
  ///   have requested to move.
  @inlinable
  @discardableResult
  public final func turn(towards direction: TurnDirection) -> Bool {
    simulator.turnAgent(agent: self, towards: direction)
  }

  @usableFromInline
  internal final func updateSimulationState(_ state: AgentSimulationState) {
    id = state.id
    position = Position(fromC: state.position)
    direction = Direction(fromC: state.direction)

    // Update scent.
    let scentShape = [Int(simulator.configuration.scentDimSize)]
    let scentBuffer = UnsafeBufferPointer(start: state.scent!, count: scentShape[0])
    scent = ShapedArray(shape: scentShape, scalars: [Float](scentBuffer))

    // Update vision.
    let visionShape = [
      2 * Int(simulator.configuration.visionRange) + 1, 
      2 * Int(simulator.configuration.visionRange) + 1, 
      Int(simulator.configuration.colorDimSize)]
    let visionSize = Int(
      (2 * simulator.configuration.visionRange + 1) * 
      (2 * simulator.configuration.visionRange + 1) * 
      simulator.configuration.colorDimSize)
    let visionBuffer = UnsafeBufferPointer(start: state.vision!, count: visionSize)
    vision = ShapedArray(shape: visionShape, scalars: [Float](visionBuffer))

    // Update items.
    let simulatorItems = simulator.configuration.items
    items = [Item: UInt32](uniqueKeysWithValues: zip(
      simulatorItems,
      UnsafeBufferPointer(
        start: state.collectedItems!, 
        count: simulatorItems.count)))
  }
}
