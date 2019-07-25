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

/// Simulation agent. Agents can interact with the Jelly Bean World by being added to simulators
/// that manage them.
public protocol Agent {
  /// Returns the agent's desired next action. This function is called automatically by all
  /// simulators managing this agent, whenever the `Simulator.step()` function is invoked.
  ///
  /// - Parameter state: Current state of the agent.
  /// - Returns: Action requested by the agent.
  ///
  /// - Note: The action is not actually performed until the simulator advances by a time step and
  ///   issues a notification about that event. The simulator only advances by a time step only
  ///   once all agents have requested some action.
  mutating func act(using state: AgentState) -> Action

  /// Saves this agent's state to the specified file.
  ///
  /// - Parameter file: URL of the file in which to save the agent's state.
  /// - Note: If the provided file exists, it may be overwritten by this function.
  func save(to file: URL) throws

  /// Loads this agent's state from the specified file.
  ///
  /// - Parameter file: URL of the file from which to load the agent's state.
  mutating func load(from file: URL) throws
}

/// State of an agent in the Jelly Bean World.
public struct AgentState {
  /// Position of the agent.
  public let position: Position

  /// Direction in which the agent is facing.
  public let direction: Direction

  /// Scent that the agent smells. This is a vector with size `S`, where `S` is the scent vector
  /// size (i.e., the scent dimensionality).
  public let scent: Tensor<Float>

  /// Visual field of the agent. This is a matrix with shape `[V + 1, V + 1, C]`, where `V` is the
  /// visual range of the agent and `C` is the color vector size (i.e., the color dimensionality).
  public let vision: Tensor<Float>

  /// Items that collected by the agent so far represented as a dictionary mapping items to counts.
  public let items: [Item: Int]

  /// Creates a new agent state on the Swift API side, corresponding to an exising agent state on
  /// the C API side.
  @inlinable
  internal init(fromC value: AgentSimulationState, for simulator: Simulator) {
    self.position = Position(fromC: value.position)
    self.direction = Direction(fromC: value.direction)

    // Construct the scent vector.
    let scentShape = [Int(simulator.configuration.scentDimSize)]
    let scentBuffer = UnsafeBufferPointer(start: value.scent!, count: scentShape[0])
    self.scent = Tensor(shape: TensorShape(scentShape), scalars: [Float](scentBuffer))

    // Construct the visual field.
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

    // Construcct the collected items dictionary.
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
