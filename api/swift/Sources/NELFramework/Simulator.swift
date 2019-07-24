import CNELFramework
import Foundation
import TensorFlow

public struct Position: Equatable {
  public let x: Int64
  public let y: Int64

  public init(x: Int64, y: Int64) {
    self.x = x
    self.y = y
  }

  @inlinable
  internal init(fromC value: CNELFramework.Position) {
    self.init(x: value.x, y: value.y)
  }

  @inlinable
  internal func toC() -> CNELFramework.Position {
    CNELFramework.Position(x: x, y: y)
  }
}

public enum Direction: UInt32 {
  case up = 0, down, left, right

  @inlinable
  internal init(fromC value: CNELFramework.Direction) {
    self.init(rawValue: value.rawValue)!
  }

  @inlinable
  internal func toC() -> CNELFramework.Direction {
    CNELFramework.Direction(rawValue: self.rawValue)
  }
}

public enum TurnDirection: UInt32 {
  case front = 0, back, left, right

  @inlinable
  internal init(fromC value: CNELFramework.TurnDirection) {
    self.init(rawValue: value.rawValue)!
  }

  @inlinable
  internal func toC() -> CNELFramework.TurnDirection {
    CNELFramework.TurnDirection(rawValue: self.rawValue)
  }
}

public enum MoveConflictPolicy: UInt32 {
  case noCollisions = 0, firstComeFirstServe, random

  @inlinable
  internal init(fromC value: CNELFramework.MovementConflictPolicy) {
    self.init(rawValue: value.rawValue)!
  }

  @inlinable
  internal func toC() -> CNELFramework.MovementConflictPolicy {
    CNELFramework.MovementConflictPolicy(rawValue: self.rawValue)
  }
}

public final class Simulator {
  public let configuration: Simulator.Configuration

  /// Pointer to the underlying C API simulator instance.
  @usableFromInline internal var handle: UnsafeMutableRawPointer?

  /// Agents interacting with this simulator (keyed by their unique identifiers).
  @usableFromInline internal var agents: [UInt64: Agent] = [:]

  /// Represents the number of simulation steps that have been executed so far.
  public private(set) var time: UInt64 = 0

  @usableFromInline
  internal let dispatchSemaphore = DispatchSemaphore(value: 1)

  @usableFromInline
  internal let dispatchQueue = DispatchQueue(
    label: "SimulatorDispatchQueue", 
    qos: .default, 
    attributes: .concurrent)

  @usableFromInline
  internal var usingDispatchQueue = false

  @inlinable
  public init(
    using configuration: Simulator.Configuration,
    saveFrequency: UInt32 = 1000, 
    savePath: String? = nil
  ) {
    self.configuration = configuration
    var cConfiguration = configuration.toC()
    let pointer = Unmanaged.passUnretained(self).toOpaque()
    self.handle = CNELFramework.simulatorCreate(
      &cConfiguration.simulatorConfig,
      nativeOnStepCallback,
      pointer,
      saveFrequency, 
      savePath)
    cConfiguration.deallocate()
  }

  // @inlinable
  // public init(
  //   using configuration: Simulator.Configuration,
  //   from file: URL,
  //   saveFrequency: UInt32,
  //   savePath: String
  // ) {
  //   self.configuration = configuration
  //   let opaque = Unmanaged.passUnretained(self).toOpaque()
  //   let pointer = UnsafeMutableRawPointer(opaque)
  //   let info = CNELFramework.simulatorLoad(
  //     file.absoluteString, 
  //     pointer,
  //     nativeOnStepCallback, 
  //     saveFrequency,
  //     savePath)
  //   self.handle = info.handle
  //   self.time = info.time
  //   let agentInfo = Array(UnsafeBufferPointer(
  //     start: info.agents!,
  //     count: Int(info.numAgents)))
  // 
  // }

  deinit {
    CNELFramework.simulatorDelete(self.handle)
  }

  @usableFromInline
  internal let nativeOnStepCallback: @convention(c) (
      UnsafeRawPointer?, 
      UnsafePointer<AgentSimulationState>?,
      UInt32, 
      Bool
  ) -> Void = { (simulatorPointer, states, numStates, saved) in
    let unmanagedSimulator = Unmanaged<Simulator>.fromOpaque(simulatorPointer!)
    let simulator = unmanagedSimulator.takeUnretainedValue()
    simulator.time += 1
    let buffer = UnsafeBufferPointer(start: states!, count: Int(numStates))
    for state in buffer { simulator.agents[state.id]!.updateSimulationState(state) }
    if saved { simulator.saveAgents() }
    if simulator.usingDispatchQueue { simulator.dispatchSemaphore.signal() }
  }

  @inlinable
  public func step() {
    if agents.count == 1 {
      usingDispatchQueue = false
      agents.first!.value.act()
    } else {
      usingDispatchQueue = true
      for agent in agents.values {
        dispatchQueue.async { agent.act() }
      }
      dispatchSemaphore.wait()
    }
  }

  @inlinable
  internal func saveAgents() {
    // TODO
  }

  @inlinable
  internal func loadAgents() {
    // TODO
  }

  /// Adds a new agent to this simulator, and updates
  /// its simulation state.
  /// 
  /// - Parameters:
  ///   - agent: The agent to be added to this simulator.
  @inlinable
  internal func addAgent<A: Agent>(_ agent: A) {
    let state = CNELFramework.simulatorAddAgent(handle, nil)
    agent.updateSimulationState(state)
    agents[state.id] = agent
    CNELFramework.simulatorDeleteAgentSimulationState(state)
  }

  @inlinable
  internal func moveAgent(
    agent: Agent,
    towards direction: Direction,
    by numSteps: UInt32
  ) -> Bool {
    CNELFramework.simulatorMoveAgent(handle, nil, agent.id!, direction.toC(), numSteps)
  }

  @inlinable
  internal func turnAgent(agent: Agent, towards direction: TurnDirection) -> Bool {
    CNELFramework.simulatorTurnAgent(handle, nil, agent.id!, direction.toC())
  }

  @inlinable
  internal func map(bottomLeft: Position, topRight: Position) -> SimulationMap {
    let cSimulationMap = CNELFramework.simulatorMap(handle, nil, bottomLeft.toC(), topRight.toC())
    defer { CNELFramework.simulatorDeleteSimulationMap(cSimulationMap) }
    return SimulationMap(fromC: cSimulationMap, for: self)
  }
}
