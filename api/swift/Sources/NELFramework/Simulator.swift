import CNELFramework
import Foundation
import TensorFlow

/// Jelly Bean World (JBW) simulator.
public final class Simulator {
  /// Simulator configuration.
  public let configuration: Simulator.Configuration

  /// Number of simulation steps that have been executed so far.
  public private(set) var time: UInt64 = 0

  /// Pointer to the underlying C API simulator instance.
  @usableFromInline internal var handle: UnsafeMutableRawPointer?

  /// Agents interacting with this simulator (keyed by their unique identifiers).
  @usableFromInline internal var agents: [UInt64: Agent] = [:]

  /// States of the agents managed by this simulator (keyed by the agents' unique identifiers).
  @usableFromInline internal var agentStates: [UInt64: AgentState] = [:]

  /// Semaphore used for synchronization when multiple agents are added to the simulator (as
  /// opposed to having just a single agent).
  @usableFromInline internal var dispatchSemaphore: DispatchSemaphore? = nil

  /// Dispatch queue used for agents taking actions asynchronously. This is only used when multiple
  /// agents are added to the simulator (as opposed to having just a single agent).
  @usableFromInline internal var dispatchQueue: DispatchQueue? = nil

  /// Creates a new simulator.
  ///
  /// - Parameter configuration: Configuration for the new simulator.
  @inlinable
  public init(using configuration: Simulator.Configuration) {
    self.configuration = configuration
    var cConfiguration = configuration.toC()
    let swiftSimulator = Unmanaged.passUnretained(self).toOpaque()
    self.handle = simulatorCreate(
      &cConfiguration.simulatorConfig,
      nativeOnStepCallback,
      swiftSimulator,
      /* saveFrequency */0,
      /* savePath */ nil)
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
  //   let info = simulatorLoad(
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
  // }

  deinit {
    simulatorDelete(handle)
  }

  /// Adds a new agent to this simulator, and updates
  /// its simulation state.
  /// 
  /// - Parameters:
  ///   - agent: The agent to be added to this simulator.
  @inlinable
  public func add(agent: Agent) {
    let state = simulatorAddAgent(handle, nil)
    agents[state.id] = agent
    agentStates[state.id] = AgentState(fromC: state, for: self)
    simulatorDeleteAgentSimulationState(state)
  }

  @inlinable
  public func step() {
    if agents.count == 1 {
      var (id, agent) = agents.first!
      switch agent.act(using: agentStates[id]!) {
      case .none: ()
      case let .move(direction, stepCount):
        simulatorMoveAgent(handle, nil, id, direction.toC(), UInt32(stepCount))
      case let .turn(direction):
        simulatorTurnAgent(handle, nil, id, direction.toC())
      }
    } else {
      if dispatchQueue == nil {
        dispatchSemaphore = DispatchSemaphore(value: 1)
        dispatchQueue = DispatchQueue(
          label: "Jelly Bean World Simulator",
          qos: .default,
          attributes: .concurrent)
      }
      for var (id, agent) in agents {
        let state = agentStates[id]!
        dispatchQueue!.async {
          switch agent.act(using: state) {
          case .none: ()
          case let .move(direction, stepCount):
            simulatorMoveAgent(self.handle, nil, id, direction.toC(), UInt32(stepCount))
          case let .turn(direction):
            simulatorTurnAgent(self.handle, nil, id, direction.toC())
          }
        }
      }
      dispatchSemaphore!.wait()
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

  @inlinable
  internal func map(bottomLeft: Position, topRight: Position) -> SimulationMap {
    let cSimulationMap = simulatorMap(handle, nil, bottomLeft.toC(), topRight.toC())
    defer { simulatorDeleteSimulationMap(cSimulationMap) }
    return SimulationMap(fromC: cSimulationMap, for: self)
  }

  @usableFromInline internal let nativeOnStepCallback: @convention(c) (
      UnsafeRawPointer?, 
      UnsafePointer<AgentSimulationState>?,
      UInt32, 
      Bool
  ) -> Void = { (simulatorPointer, states, numStates, saved) in
    let unmanagedSimulator = Unmanaged<Simulator>.fromOpaque(simulatorPointer!)
    let simulator = unmanagedSimulator.takeUnretainedValue()
    simulator.time += 1
    let buffer = UnsafeBufferPointer(start: states!, count: Int(numStates))
    for state in buffer {
      simulator.agentStates[state.id] = AgentState(fromC: state, for: simulator)
    }
    if saved { simulator.saveAgents() }
    simulator.dispatchSemaphore?.signal()
  }
}

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
