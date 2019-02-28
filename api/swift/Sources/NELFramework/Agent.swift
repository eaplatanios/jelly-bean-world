import CNELFramework
import Foundation
import TensorFlow

public class Agent {
  let simulator: Simulator
  let delegate: AgentDelegate

  var id: UInt64? = nil
  var position: Position? = nil
  var direction: Direction? = nil
  var scent: ShapedArray<Float>? = nil
  var vision: ShapedArray<Float>? = nil
  var items: [Item : UInt32]? = nil
  var lastCollectedItems: [Item : UInt32]? = nil

  public init(in simulator: Simulator, with delegate: AgentDelegate) {
    self.simulator = simulator
    self.delegate = delegate
    self.items = [:]
    simulator.addAgent(self)
  }

  @inline(__always)
  public func act() {
    delegate.act(self)
  }

  /// Moves this agent in the simulated environment.
  ///
  /// Note that the agent is not moved until the simulator 
  /// advances by a time step and issues a notification 
  /// about that event. The simulator only advances the 
  /// time step once all agents have requested to move.
  @inline(__always) @discardableResult
  public func move(
    towards direction: Direction, 
    by numSteps: UInt32
  ) -> Bool {
    return self.simulator.moveAgent(
      agent: self,
      towards: direction,
      by: numSteps)
  }
  
  /// Turns this agent in the simulated environment.
  ///
  /// Note that the agent is not turned until the simulator 
  /// advances by a time step and issues a notification 
  /// about that event. The simulator only advances the 
  /// time step once all agents have requested to move.
  @inline(__always) @discardableResult
  public func turn(
    towards direction: TurnDirection
  ) -> Bool {
    return self.simulator.turnAgent(
      agent: self,
      towards: direction)
  }

  internal func updateSimulationState(
    _ state: AgentSimulationState
  ) {
    self.id = state.id
    self.position = state.position
    self.direction = Direction.fromCDirection(state.direction)
    self.scent = scentToShapedArray(
      for: simulator.config, 
      state.scent!)
    self.vision = visionToShapedArray(
      for: simulator.config, 
      state.vision!)
    let previousItems = self.items!
    self.items = itemCountsToDictionary(
      for: simulator.config, 
      state.collectedItems!)
    // TODO: Better way to do this.
    self.lastCollectedItems = self.items
    for (item, count) in previousItems {
      self.lastCollectedItems![item] = self.items![item]! - count
    }
  }
}

public protocol AgentDelegate {
  func act(_ agent: Agent)
  func save(_ agent: Agent, to file: URL) throws
  func load(_ agent: Agent, from file: URL) throws
}
