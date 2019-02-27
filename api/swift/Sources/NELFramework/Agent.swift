import CNELFramework
import Foundation
import TensorFlow

public protocol Agent : AnyObject {
  var simulator: Simulator { get }

  var id: UInt64 { get set }
  var position: Position { get set }
  var direction: Direction { get set }
  var scent: ShapedArray<Float> { get set }
  var vision: ShapedArray<Float> { get set }
  var items: [Item : UInt32] { get set }

  init(in simulator: Simulator)

  func act()
}

public extension Agent { 
  init(in simulator: Simulator) {
    self.init(in: simulator)
    simulator.addAgent(self)
  }

  /// Moves this agent in the simulated environment.
  ///
  /// Note that the agent is not moved until the simulator 
  /// advances by a time step and issues a notification 
  /// about that event. The simulator only advances the 
  /// time step once all agents have requested to move.
  @inline(__always)
  func move(
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
  @inline(__always)
  func turn(
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
    self.items = itemCountsToDictionary(
      for: simulator.config, 
      state.collectedItems!)
  }
}

public protocol StatefulAgent : Agent {
  associatedtype S

  var state: S { get set }
  
  static func load(from file: URL) -> S
  func save(state: S, to file: URL) -> Void
}

extension StatefulAgent {
  public init(in simulator: Simulator, from file: URL) {
    self.init(in: simulator)
    self.state = Self.load(from: file)
  }
}
