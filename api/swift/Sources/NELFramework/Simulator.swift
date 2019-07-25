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

  /// States of the agents managed by this simulator (keyed by the agents' unique identifiers).
  @usableFromInline internal var agentStates: [UInt64: AgentState] = [:]

  /// Agents interacting with this simulator (keyed by their unique identifiers).
  @usableFromInline internal var agents: [UInt64: Agent] = [:]

  /// Semaphore used for synchronization when multiple agents are added to the simulator (as
  /// opposed to having just a single agent).
  @usableFromInline internal var dispatchSemaphore: DispatchSemaphore? = nil

  /// Dispatch queue used for agents taking actions asynchronously. This is only used when multiple
  /// agents are added to the simulator (as opposed to having just a single agent).
  @usableFromInline internal var dispatchQueue: DispatchQueue? = nil

  /// Creates a new simulator.
  ///
  /// - Parameter configuration: Configuration for the new simulator.
  public init(using configuration: Simulator.Configuration) {
    self.configuration = configuration
    var cConfig = configuration.toC()
    defer { cConfig.deallocate() }
    let swiftSimulator = Unmanaged.passUnretained(self).toOpaque()
    self.handle = simulatorCreate(&cConfig.configuration, nativeOnStepCallback, swiftSimulator)
  }

  /// Loads a simulator from the provided file.
  ///
  /// - Parameters:
  ///   - file: File in which the simulator is saved.
  ///   - agents: Agents that this simulator manages.
  /// - Precondition: The number of agents provided must match the number of agents the simulator
  ///   managed before its state was saved.
  public init(fromFile file: URL, agents: [Agent]) {
    let swiftSimulator = Unmanaged.passUnretained(self).toOpaque()
    let info = simulatorLoad(file.absoluteString, nativeOnStepCallback, swiftSimulator)
    self.configuration = info.config
    self.handle = info.handle
    self.time = info.time
    self.agentStates = [UInt64: AgentState](
      uniqueKeysWithValues: UnsafeBufferPointer(
        start: info.agents!,
        count: Int(info.numAgents)
      ).map { ($0.id, AgentState(fromC: $0, for: self)) })
    precondition(
      agents.count == agentStates.count,
      """
      The number of agent states stored in the provided simulator file does not match
      the number of agents provided.
      """)
    self.agents = [UInt64: Agent](uniqueKeysWithValues: zip(agentStates.keys, agents))
  }

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

  /// Performs a simulation step.
  ///
  /// - Note: This function will block until all the agents managed by this simulator has acted.
  @inlinable
  public func step() {
    if agents.count == 1 {
      let id = agents.first!.key
      agents[id]!.act(using: agentStates[id]!).invoke(
        simulatorHandle: handle,
        clientHandle: nil,
        agentID: id)
    } else {
      if dispatchQueue == nil {
        dispatchSemaphore = DispatchSemaphore(value: 1)
        dispatchQueue = DispatchQueue(
          label: "Jelly Bean World Simulator",
          qos: .default,
          attributes: .concurrent)
      }
      for id in agents.keys {
        let state = agentStates[id]!
        dispatchQueue!.async {
          self.agents[id]!.act(using: state).invoke(
            simulatorHandle: self.handle,
            clientHandle: nil,
            agentID: id)
        }
      }
      dispatchSemaphore!.wait()
    }
  }

  /// Returns the portion of the simulator map that lies within the rectangle formed by the
  /// `bottomLeft` and `topRight` corners.
  ///
  /// - Parameters:
  ///   - bottomLeft: Bottom left corner of the requested map portion.
  ///   - topRight: Top right corner of the requested map portion.
  @inlinable
  internal func map(
    bottomLeft: Position = Position(x: Int64.min, y: Int64.min),
    topRight: Position = Position(x: Int64.max, y: Int64.max)
  ) -> SimulationMap {
    let cSimulationMap = simulatorMap(handle, nil, bottomLeft.toC(), topRight.toC())
    defer { simulatorDeleteSimulationMap(cSimulationMap) }
    return SimulationMap(fromC: cSimulationMap, for: self)
  }

  /// Callback function that is invoked by the C API side simulator whenever a step is completed.
  @usableFromInline internal let nativeOnStepCallback: @convention(c) (
    UnsafeRawPointer?,
    UnsafePointer<AgentSimulationState>?,
    UInt32
  ) -> Void = { (simulatorPointer, states, numStates) in
    let unmanagedSimulator = Unmanaged<Simulator>.fromOpaque(simulatorPointer!)
    let simulator = unmanagedSimulator.takeUnretainedValue()
    simulator.time += 1
    let buffer = UnsafeBufferPointer(start: states!, count: Int(numStates))
    for state in buffer {
      simulator.agentStates[state.id] = AgentState(fromC: state, for: simulator)
    }
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
    CNELFramework.Direction(rawValue: rawValue)
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
    CNELFramework.TurnDirection(rawValue: rawValue)
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
    CNELFramework.MovementConflictPolicy(rawValue: rawValue)
  }
}
