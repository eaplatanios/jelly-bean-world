import CNELFramework
import Foundation
import TensorFlow

public class Agent {
  let simulator: Simulator
  let delegate: AgentDelegate

  internal var rawScent: (shape: [Int], values: [Float])? = nil
  internal var rawVision: (shape: [Int], values: [Float])? = nil
  internal var rawItems: [UInt32]? = nil

  public internal(set) var id: UInt64? = nil
  public internal(set) var position: Position? = nil
  public internal(set) var direction: Direction? = nil
  
  public var scent: ShapedArray<Float>? {
    return rawScent.map { ShapedArray(shape: $0.shape, scalars: $0.values) }
  }

  public var vision: ShapedArray<Float>? {
    return rawVision.map { ShapedArray(shape: $0.shape, scalars: $0.values) }
  }

  public var items: [Item : UInt32]? {
    return rawItems.map {
      var counts = [Item: UInt32]()
      for (index, item) in simulator.config.items.enumerated() {
        counts[item] = $0[index]
      }
      return counts
    }
  }

  public init(in simulator: Simulator, with delegate: AgentDelegate) {
    self.simulator = simulator
    self.delegate = delegate
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
  public func move(towards direction: Direction, by numSteps: UInt32 = 1) -> Bool {
    return self.simulator.moveAgent(agent: self, towards: direction, by: numSteps)
  }
  
  /// Turns this agent in the simulated environment.
  ///
  /// Note that the agent is not turned until the simulator 
  /// advances by a time step and issues a notification 
  /// about that event. The simulator only advances the 
  /// time step once all agents have requested to move.
  @inline(__always) @discardableResult
  public func turn(towards direction: TurnDirection) -> Bool {
    return self.simulator.turnAgent(agent: self, towards: direction)
  }

  internal func updateSimulationState(_ state: AgentSimulationState) {
    self.id = state.id
    self.position = state.position
    self.direction = Direction.fromC(state.direction)
    self.rawScent = scentToArray(for: simulator.config, state.scent!)
    self.rawVision = visionToArray(for: simulator.config, state.vision!)
    self.rawItems = itemCountsToArray(for: simulator.config, state.collectedItems!)
  }
}

public protocol AgentDelegate {
  func act(_ agent: Agent)
  func save(_ agent: Agent, to file: URL) throws
  func load(_ agent: Agent, from file: URL) throws
}
