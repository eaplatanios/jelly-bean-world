// Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import CJellyBeanWorld
import Foundation
import TensorFlow

public enum JellyBeanWorldError: Error {
  case OutOfMemory
  case InvalidAgentID
  case ViolatedPermissions
  case AgentAlreadyActed
  case AgentAlreadyExists
  case ServerMessageParsingFailure
  case ClientMessageParsingFailure
  case ServerOutOfMemory
  case ClientOutOfMemory
  case IOError
  case LostConnection
  case InvalidSimulatorConfiguration
  case MPIError
  case InvalidSemaphoreID
  case SemaphoreAlreadySignaled
  case UnknownNativeError
}

@usableFromInline
internal func checkStatus(_ status: UnsafeMutablePointer<JBW_Status>?) throws {
  let status = status!.pointee
  switch status.code {
  case JBW_OK: ()
  case JBW_OUT_OF_MEMORY: throw JellyBeanWorldError.OutOfMemory
  case JBW_INVALID_AGENT_ID: throw JellyBeanWorldError.InvalidAgentID
  case JBW_VIOLATED_PERMISSIONS: throw JellyBeanWorldError.ViolatedPermissions
  case JBW_AGENT_ALREADY_ACTED: throw JellyBeanWorldError.AgentAlreadyActed
  case JBW_AGENT_ALREADY_EXISTS: throw JellyBeanWorldError.AgentAlreadyExists
  case JBW_SERVER_PARSE_MESSAGE_ERROR: throw JellyBeanWorldError.ServerMessageParsingFailure
  case JBW_CLIENT_PARSE_MESSAGE_ERROR: throw JellyBeanWorldError.ClientMessageParsingFailure
  case JBW_SERVER_OUT_OF_MEMORY: throw JellyBeanWorldError.ServerOutOfMemory
  case JBW_CLIENT_OUT_OF_MEMORY: throw JellyBeanWorldError.ClientOutOfMemory
  case JBW_IO_ERROR: throw JellyBeanWorldError.IOError
  case JBW_LOST_CONNECTION: throw JellyBeanWorldError.LostConnection
  case JBW_INVALID_SIMULATOR_CONFIGURATION: throw JellyBeanWorldError.InvalidSimulatorConfiguration
  case JBW_MPI_ERROR: throw JellyBeanWorldError.MPIError
  case JBW_INVALID_SEMAPHORE_ID: throw JellyBeanWorldError.InvalidSemaphoreID
  case JBW_SEMAPHORE_ALREADY_SIGNALED: throw JellyBeanWorldError.SemaphoreAlreadySignaled
  case _: throw JellyBeanWorldError.UnknownNativeError
  }
}

/// Position in the simulation map.
public struct Position: Equatable {
  /// Horizontal coordinate of the position.
  public let x: Int64

  /// Vertical coordination of the position.
  public let y: Int64

  public init(x: Int64, y: Int64) {
    self.x = x
    self.y = y
  }
}

/// Direction in the simulation map. This can either be the direction in which an agent is facing,
/// or the direction in which an agent is moving.
public enum Direction: UInt32 {
  case up = 0, down, left, right
}

/// Direction for turning actions.
public enum TurnDirection: UInt32 {
  case front = 0, back, left, right
}

/// Conflict resolution policy for when two or more agents request to move to the same location.
public enum MoveConflictPolicy: UInt32 {
  case noCollisions = 0, firstComeFirstServe, random
}

/// Policy used to indicate whether an action is allowed, disallowed, or ignored. If an action is
/// disallowed, then attempting to perform it will immediately fail, preventing the simulator from
/// progressing if an agent has not performed an action during the current time step. If an action
/// is ignored, then the agent will perform a no-op for that time step (i.e., the simulator will
/// simply ignore that action).
public enum ActionPolicy: UInt32 {
  case allowed = 0, disallowed, ignored
}

/// Jelly Bean World (JBW) simulator.
public final class Simulator {
  /// Simulator configuration.
  public let configuration: Configuration

  /// Simulation server configuration. This is `nil` when this simulator instance is not a
  /// simulation server.
  public let serverConfiguration: ServerConfiguration?

  /// Simulation client configuration. This is `nil` when this simulator instance is not a
  /// simulation client.
  public let clientConfiguration: ClientConfiguration?

  /// Number of simulation steps that have been executed so far.
  public private(set) var time: UInt64 = 0

  /// Pointer to the underlying C API simulator instance.
  @usableFromInline internal var handle: UnsafeMutableRawPointer?

  /// Pointer to the underlying C API simulation server instance. This is `nil` when this simulator
  /// instance is not a simulation server.
  @usableFromInline internal var serverHandle: UnsafeMutableRawPointer?

  /// Pointer to the underlying C API simulation client instance. This is `nil` when this simulator
  /// instance is not a simulation client.
  @usableFromInline internal var clientHandle: UnsafeMutableRawPointer?

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
  /// - Parameters:
  ///   - configuration: Configuration for the new simulator.
  ///   - serverConfiguration: Server configuration. If `nil` the created simulator will not act
  ///     as a server, but rather as isolated simulator instance.
  public init(
    using configuration: Configuration,
    serverConfiguration: ServerConfiguration? = nil
  ) throws {
    self.configuration = configuration
    self.serverConfiguration = serverConfiguration
    self.clientConfiguration = nil
    var cConfig = configuration.toC()
    defer { cConfig.deallocate() }
    var status = JBW_Status()
    self.handle = simulatorCreate(&cConfig.configuration, nativeOnStepCallback, &status)
    try checkStatus(&status)
    simulatorSetStepCallbackData(handle, Unmanaged.passUnretained(self).toOpaque())
    if let config = serverConfiguration {
      self.serverHandle = simulationServerStart(
        handle,
        config.port,
        config.connectionQueueCapacity,
        config.workerCount,
        // TODO [PERMISSIONS].
        Permissions(
          addAgent: true,
          removeAgent: true,
          removeClient: true,
          setActive: true,
          getMap: true,
          getAgentIds: true,
          getAgentStates: true,
          semaphores: true),
        &status)
      try checkStatus(&status)
    }
  }

  /// Loads a simulator from the provided file.
  ///
  /// - Parameters:
  ///   - file: File in which the simulator is saved.
  ///   - agents: Agents that this simulator manages.
  ///   - serverConfiguration: Server configuration. If `nil` the created simulator will not act
  ///     as a server, but rather as isolated simulator instance.
  /// - Precondition: The number of agents provided must match the number of agents the simulator
  ///   managed before its state was saved.
  public init(
    fromFile file: URL,
    agents: [Agent],
    serverConfiguration: ServerConfiguration? = nil
  ) throws {
    var status = JBW_Status()
    let cSimulatorInfo = simulatorLoad(file.absoluteString, nativeOnStepCallback, &status)
    try checkStatus(&status)
    defer { simulatorDeleteSimulatorInfo(cSimulatorInfo) }
    self.handle = cSimulatorInfo.handle
    self.configuration = Configuration(fromC: cSimulatorInfo.config)
    self.serverConfiguration = serverConfiguration
    self.clientConfiguration = nil
    self.time = cSimulatorInfo.time
    self.agentStates = [UInt64: AgentState](
      uniqueKeysWithValues: UnsafeBufferPointer(
        start: cSimulatorInfo.agents!,
        count: Int(cSimulatorInfo.numAgents)
      ).map { ($0.id, AgentState(fromC: $0, using: configuration)) })
    precondition(
      agents.count == agentStates.count,
      """
      The number of agent states stored in the provided simulator file does not match
      the number of agents provided.
      """)
    self.agents = [UInt64: Agent](uniqueKeysWithValues: zip(agentStates.keys, agents))
    simulatorSetStepCallbackData(handle, Unmanaged.passUnretained(self).toOpaque())
    if let config = serverConfiguration {
      self.serverHandle = simulationServerStart(
        handle,
        config.port,
        config.connectionQueueCapacity,
        config.workerCount,
        // TODO [PERMISSIONS].
        Permissions(
          addAgent: true,
          removeAgent: true,
          removeClient: true,
          setActive: true,
          getMap: true,
          getAgentIds: true,
          getAgentStates: true,
          semaphores: true),
        &status)
      try checkStatus(&status)
    }
  }

  // /// Creates a new simulation client.
  // ///
  // /// - Parameters:
  // ///   - configuration: Configuration for the new simulation client.
  // public init(
  //   using configuration: Configuration,
  //   clientConfiguration: ClientConfiguration,
  //   agents: [Agent]
  // ) {
  //   self.configuration = configuration
  //   self.serverConfiguration = nil
  //   self.clientConfiguration = clientConfiguration
  //   let clientInformation = simulationClientStart(
  //     configuration.serverAddress,
  //     configuration.serverPort,
  //     nativeOnStepCallback,
  //     nativeLostConnectionCallback)
  //   defer { simulatorDeleteSimulationClientInfo(clientInformation, ) }
  //   self.handle = clientInformation.handle
  //   self.time = clientInformation.simulationTime
  //   simulationClientSetStepCallbackData(handle, Unmanaged.passUnretained(self).toOpaque())
  // }

  deinit {
    if let h = serverHandle { simulationServerStop(h) }
    if let h = clientHandle { simulationClientStop(h) }
    if let h = handle { simulatorDelete(h) }
  }

  /// Adds a new agent to this simulator, and updates its simulation state.
  /// 
  /// - Parameter agent: The agent to be added to this simulator.
  /// - Returns: ID of the new agent.
  @inlinable
  @discardableResult
  public func add(agent: Agent) throws -> UInt64 {
    var status = JBW_Status()
    let state = simulatorAddAgent(handle, clientHandle, &status)
    try checkStatus(&status)
    defer { simulatorDeleteAgentSimulationState(state) }
    let id = state.id
    agents[id] = agent
    agentStates[id] = AgentState(fromC: state, using: configuration)
    return id
  }

  /// Removes the agent with ID `agentID` from this simulator.
  ///
  /// - Parameter id: ID of the agent to remove.
  @inlinable
  public func remove(agentWithID id: UInt64) throws {
    var status = JBW_Status()
    simulatorRemoveAgent(handle, clientHandle, id, &status)
    try checkStatus(&status)
  }

  /// Activates the agent with ID `agentID` managed by this simulator.
  ///
  /// - Parameter id: ID of the agent to activate.
  @inlinable
  public func activate(agentWithID id: UInt64) throws {
    var status = JBW_Status()
    simulatorSetActive(handle, clientHandle, id, true, &status)
    try checkStatus(&status)
  }

  /// Deactivates the agent with ID `agentID` managed by this simulator.
  ///
  /// - Parameter id: ID of the agent to deactivate.
  @inlinable
  public func deactivate(agentWithID id: UInt64) throws {
    var status = JBW_Status()
    simulatorSetActive(handle, clientHandle, id, false, &status)
    try checkStatus(&status)
  }

  /// Performs a simulation step.
  ///
  /// - Note: This function will block until all the agents managed by this simulator has acted.
  @inlinable
  public func step() throws {
    if agents.count == 1 {
      let id = agents.first!.key
      try agents[id]!.act(using: agentStates[id]!).invoke(
        simulatorHandle: handle,
        clientHandle: clientHandle,
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
          // TODO: !!!! Fix this by propagating the error upwards.
          try! self.agents[id]!.act(using: state).invoke(
            simulatorHandle: self.handle,
            clientHandle: self.clientHandle,
            agentID: id)
        }
      }
      dispatchSemaphore!.wait()
    }
  }

  /// Saves this simulator in the provided file.
  ///
  /// - Parameter file: File in which to save the state of this simulator.
  @inlinable
  public func save(to file: URL) throws {
    var status = JBW_Status()
    simulatorSave(handle, file.absoluteString, &status)
    try checkStatus(&status)
  }

  /// Returns the portion of the simulator map that lies within the rectangle formed by the
  /// `bottomLeft` and `topRight` corners.
  ///
  /// - Parameters:
  ///   - bottomLeft: Bottom left corner of the requested map portion.
  ///   - topRight: Top right corner of the requested map portion.
  ///   - includingScent: Indicates whether to include scent in the obtained map representation.
  @inlinable
  internal func map(
    bottomLeft: Position = Position(x: Int64.min, y: Int64.min),
    topRight: Position = Position(x: Int64.max, y: Int64.max),
    includingScent: Bool = true
   ) throws -> SimulationMap {
    var status = JBW_Status()
    let cSimulationMap = simulatorMap(
      handle, clientHandle, bottomLeft.toC(), topRight.toC(), includingScent, &status)
    try checkStatus(&status)
    defer { simulatorDeleteSimulationMap(cSimulationMap) }
    return SimulationMap(fromC: cSimulationMap, using: configuration)
  }

  /// Stops the simulation server, if one was started when creating this simulator.
  @inlinable
  public func stopServer() {
    if let h = serverHandle { simulationServerStop(h) }
    serverHandle = nil
  }

  /// Stops the simulation client, if one was started when creating this simulator.
  @inlinable
  public func stopClient() {
    if let h = clientHandle { simulationClientStop(h) }
    clientHandle = nil
  }

  /// Callback function that is invoked by the C API side simulator whenever a step is completed.
  @usableFromInline internal let nativeOnStepCallback: @convention(c) (
    UnsafeRawPointer?,
    UnsafePointer<AgentSimulationState>?,
    UInt32
  ) -> Void = { (simulatorPointer, states, stateCount) in
    let unmanagedSimulator = Unmanaged<Simulator>.fromOpaque(simulatorPointer!)
    let simulator = unmanagedSimulator.takeUnretainedValue()
    simulator.time += 1
    let buffer = UnsafeBufferPointer(start: states!, count: Int(stateCount))
    for state in buffer {
      simulator.agentStates[state.id] = AgentState(fromC: state, using: simulator.configuration)
    }
    simulator.dispatchSemaphore?.signal()
  }

  /// Callback function that is invoked by the C API side simulator client whenever the connection
  /// to the simulation server is lost.
  @usableFromInline internal let nativeLostConnectionCallback: @convention(c) (
    UnsafeRawPointer?
  ) -> Void = { (simulatorPointer) in
    let unmanaged = Unmanaged<Simulator>.fromOpaque(simulatorPointer!)
    let simulator = unmanaged.takeUnretainedValue()
    simulator.clientConfiguration!.lostConnectionCallback(simulator)
   }
}

extension Simulator {
  /// Simulator configuration.
  public struct Configuration: Equatable, Hashable {
    /// Seed used by the random number generator that is used while proceeduraly generating the
    /// Jelly Bean World map.
    public let randomSeed: UInt32

    /// Maximum movement steps allowed during each simulation step.
    public let maxStepsPerMove: UInt32

    /// Dimensionality (i.e., size) of the scent vector.
    public let scentDimensionality: UInt32

    /// Dimensionality (i.e., size) of the color vector.
    public let colorDimensionality: UInt32

    /// Vision range for the agents (i.e., how far they can see).
    public let visionRange: UInt32

    /// Policies for the available move actions. The default policy for move directions not
    /// included in this dictionary is that they are disallowed.
    public let movePolicies: [Direction: ActionPolicy]

    /// Policies for the available turn actions. The default policy for turn directions not
    /// included in this dictionary is that they are disallowed.
    public let turnPolicies: [TurnDirection: ActionPolicy]

    /// Boolean flag that indicates whether no-op actions are permitted.
    public let noOpAllowed: Bool

    /// Size of each map patch. All patches are square and this quantity denotes their length.
    public let patchSize: UInt32

    /// Number of Markov Chain Monte Carlo (MCMC) iterations used when sampling map patches.
    public let mcmcIterations: UInt32

    /// All possible items that can exist in this simulation.
    public let items: [Item]

    /// Color of each agent. This is a vector of size `colorDimensionality`.
    public let agentColor: ShapedArray<Float>

    /// Conflict resolution policy for when multiple agents request to move to the same location.
    public let moveConflictPolicy: MoveConflictPolicy

    /// Scent decay parameter (used by the scent simulation algorithm).
    public let scentDecay: Float

    /// Scent diffusion parameter (used by the scent simulation algorithm).
    public let scentDiffusion: Float

    /// Lifetime of removed items (used by the scent simulation algorithm).
    public let removedItemLifetime: UInt32

    public init(
      randomSeed: UInt32,
      maxStepsPerMove: UInt32,
      scentDimensionality: UInt32,
      colorDimensionality: UInt32,
      visionRange: UInt32,
      movePolicies: [Direction: ActionPolicy],
      turnPolicies: [TurnDirection: ActionPolicy],
      noOpAllowed: Bool,
      patchSize: UInt32,
      mcmcIterations: UInt32,
      items: [Item],
      agentColor: ShapedArray<Float>,
      moveConflictPolicy: MoveConflictPolicy,
      scentDecay: Float,
      scentDiffusion: Float,
      removedItemLifetime: UInt32
    ) {
      self.randomSeed = randomSeed
      self.maxStepsPerMove = maxStepsPerMove
      self.scentDimensionality = scentDimensionality
      self.colorDimensionality = colorDimensionality
      self.visionRange = visionRange
      self.movePolicies = movePolicies
      self.turnPolicies = turnPolicies
      self.noOpAllowed = noOpAllowed
      self.patchSize = patchSize
      self.mcmcIterations = mcmcIterations
      self.items = items
      self.agentColor = agentColor
      self.moveConflictPolicy = moveConflictPolicy
      self.scentDecay = scentDecay
      self.scentDiffusion = scentDiffusion
      self.removedItemLifetime = removedItemLifetime
    }
  }
}

extension Simulator {
  /// Simulation server configuration.
  public struct ServerConfiguration {
    /// Port in which the simulation server will be listening.
    public let port: UInt32

    /// Maximum number of simultaneous new connections that can be handled by the server.
    public let connectionQueueCapacity: UInt32

    /// Number of worker threads to dispatch. They are tasked with processing incoming messages
    ///  from clients.
    public let workerCount: UInt32

    public init(port: UInt32, connectionQueueCapacity: UInt32 = 256, workerCount: UInt32 = 8) {
      self.port = port
      self.connectionQueueCapacity = connectionQueueCapacity
      self.workerCount = workerCount
    }
  }
}

extension Simulator {
  /// Simulation client configuration.
  public struct ClientConfiguration {
    /// Address in which the simulation server is listening.
    public let serverAddress: String

    /// Port in which the simulation server is listening.
    public let serverPort: UInt32

    /// Callback function that is invoked whenever the connection to the server is lost.
    public let lostConnectionCallback: (Simulator) -> Void

    public init(
      serverAddress: String,
      serverPort: UInt32,
      lostConnectionCallback: @escaping (Simulator) -> Void = { _ in
        fatalError("Lost connection to the Jelly Bean World simulation server.")
      }
    ) {
      self.serverAddress = serverAddress
      self.serverPort = serverPort
      self.lostConnectionCallback = lostConnectionCallback
    }
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
}

/// State of an agent in the Jelly Bean World.
public struct AgentState {
  /// Position of the agent.
  public let position: Position

  /// Direction in which the agent is facing.
  public let direction: Direction

  /// Scent that the agent smells. This is a vector with size `S`, where `S` is the scent vector
  /// size (i.e., the scent dimensionality).
  public let scent: ShapedArray<Float>

  /// Visual field of the agent. This is a matrix with shape `[V + 1, V + 1, C]`, where `V` is the
  /// visual range of the agent and `C` is the color vector size (i.e., the color dimensionality).
  public let vision: ShapedArray<Float>

  /// Items that collected by the agent so far represented as a dictionary mapping items to counts.
  public let items: [Item: Int]
}

/// Action that can be taken by agents in the jelly bean world.
public enum Action {
  /// No action.
  case none

  /// Move action, along the specified direction and for the provided number of steps.
  case move(direction: Direction, stepCount: Int = 1)

  /// Turn action (without any movement to a different cell).
  case turn(direction: TurnDirection)

  /// Invokes this action in the simulator.
  ///
  /// - Parameters:
  ///   - simulatorHandle: Pointer to the C simulator instance.
  ///   - clientHandle: Pointer to the C simulator client instance (if a client is being used).
  ///   - agentID: Identifier of the agent for which to invoke this action.
  @inlinable
  internal func invoke(
    simulatorHandle: UnsafeMutableRawPointer?,
    clientHandle: UnsafeMutableRawPointer?,
    agentID: UInt64
  ) throws {
    var status = JBW_Status()
    switch self {
    case .none:
      simulatorNoOpAgent(simulatorHandle, clientHandle, agentID, &status)
    case let .move(direction, stepCount):
      simulatorMoveAgent(
        simulatorHandle,
        clientHandle,
        agentID,
        direction.toC(),
        UInt32(stepCount),
        &status)
    case let .turn(direction):
      simulatorTurnAgent(simulatorHandle, clientHandle, agentID, direction.toC(), &status)
    }
    try checkStatus(&status)
  }
}

/// Item type.
public struct Item: Equatable, Hashable {
  /// Name of this item type.
  public let name: String

  /// Scent of this item type. This is a vector with size `S`, where `S` is the scent vector
  /// size (i.e., the scent dimensionality).
  public let scent: ShapedArray<Float>

  /// Color of this item type. This is a vector with size `C`, where `C` is the color vector
  /// size (i.e., the color dimensionality).
  public let color: ShapedArray<Float>

  /// Map from item ID to counts that represents how many of each item are required to be collected
  /// first, before being able to collect this item.
  public let requiredItemCounts: [Int: UInt32] // TODO: Convert to [Item: Int].

  /// Map from item ID to counts that represents how many of each item are required to be exchanged
  /// for collecting this item.
  public let requiredItemCosts: [Int: UInt32] // TODO: Convert to [Item: Int].

  /// Indicates whether this item blocks the movement of agents (e.g., used for walls).
  public let blocksMovement: Bool

  /// Energy functions that represent how instances of this item type are distributed in the world.
  public let energyFunctions: EnergyFunctions

  @inlinable
  public init(
    name: String,
    scent: ShapedArray<Float>,
    color: ShapedArray<Float>,
    requiredItemCounts: [Int: UInt32],
    requiredItemCosts: [Int: UInt32],
    blocksMovement: Bool,
    energyFunctions: EnergyFunctions
  ) {
    self.name = name
    self.scent = scent
    self.color = color
    self.requiredItemCounts = requiredItemCounts
    self.requiredItemCosts = requiredItemCosts
    self.blocksMovement = blocksMovement
    self.energyFunctions = energyFunctions
  }
}

/// Energy functions for an item that define how instances of that item are distributed in the
/// Jelly Bean World. The instances distribution is defined as a Gibbs random field which is
/// described in terms of an intensity function and a set of interaction functions with other
/// items. More specifically, the probability of an item `m` appearing in position `(i, j)` is:
/// ```
/// P[item m at position (i, j)] ∝ \exp{
///   Intensity[m] *
///   \sum_{(k, l)}\sum_{n} Exists[item n at position (k, l)] * Interaction[m, n, (i, j), (k, l)] }
/// ```
/// where `(k, l)` is a position in the map and `n` is another item type.
public struct EnergyFunctions: Hashable {
  /// Intensity function of an item type.
  @usableFromInline internal let intensityFn: IntensityFunction

  /// Interaction functions of an item type with other other types.
  @usableFromInline internal let interactionFns: [InteractionFunction]

  @inlinable
  public init(intensityFn: IntensityFunction, interactionFns: [InteractionFunction]) {
    self.intensityFn = intensityFn
    self.interactionFns = interactionFns
  }
}

/// Intensity function of an item type.
public struct IntensityFunction: Hashable {
  /// ID of this function.
  @usableFromInline internal let id: UInt32

  /// Extra arguments that are passed to the intensity function whenever it is invoked.
  @usableFromInline internal let arguments: [Float]

  @inlinable
  public init(id: UInt32, arguments: [Float] = []) {
    self.id = id
    self.arguments = arguments
  }
}

extension IntensityFunction {
  /// Returns an intensity function that always returns `value`, irrespective of its inputs.
  @inlinable
  public static func constant(_ value: Float) -> IntensityFunction {
    IntensityFunction(id: 1, arguments: [value])
  }
}

/// Interaction function of an item type with another item type.
public struct InteractionFunction: Hashable {
  /// ID of this function.
  @usableFromInline internal let id: UInt32

  /// ID of the item for which this interaction is defined.
  @usableFromInline internal let itemId: UInt32

  /// Extra arguments that are passed to the interaction function whenever it is invoked.
  @usableFromInline internal let arguments: [Float]

  @inlinable
  public init(id: UInt32, itemId: UInt32, arguments: [Float] = []) {
    self.id = id
    self.itemId = itemId
    self.arguments = arguments
  }
}

extension InteractionFunction {
  // TODO: Document.
  @inlinable
  public static func piecewiseBox(
    itemId: UInt32, 
    _ firstCutoff: Float, 
    _ secondCutoff: Float,
    _ firstValue: Float,
    _ secondValue: Float
  ) -> InteractionFunction {
    InteractionFunction(
      id: 1,
      itemId: itemId,
      arguments: [
        firstCutoff, secondCutoff, 
        firstValue, secondValue])
  }

  // TODO: Document.
  @inlinable
  public static func cross(
    itemId: UInt32,
    _ nearCutoff: Float,
    _ farCutoff: Float,
    _ nearAxisAlignedValue: Float,
    _ nearMisalignedValue: Float,
    _ farAxisAlignedValue: Float,
    _ farMisalignedValue: Float
  ) -> InteractionFunction {
    InteractionFunction(
      id: 2,
      itemId: itemId,
      arguments: [
        nearCutoff, farCutoff,
        nearAxisAlignedValue, nearMisalignedValue, 
        farAxisAlignedValue, farMisalignedValue])
  }
}

/// A simulation map.
public struct SimulationMap {
  public let patches: [Patch]

  /// Creates a new simulation map which consists of the provided patches.
  /// - Parameter patches: Patches comprising the new simulation map.
  @inlinable
  public init(patches: [Patch]) {
    self.patches = patches
  }

  /// Creates a new simulation map on the Swift API side, corresponding to an exising simulation
  /// map on the C API side.
  @inlinable
  internal init(
    fromC value: CJellyBeanWorld.SimulationMap,
    using configuration: Simulator.Configuration
  ) {
    let cPatches = UnsafeBufferPointer(start: value.patches!, count: Int(value.numPatches))
    self.patches = cPatches.map { Patch(fromC: $0, using: configuration) }
  }
}

extension SimulationMap {
  /// A patch of the simulation map.
  public struct Patch {
    /// Position of the patch.
    @usableFromInline let position: Position

    /// Flag indicating whether the patch has been sampled and fixed or whether it was sampled as a 
    /// patch in the map boundaries that may later be resampled.
    @usableFromInline let fixed: Bool

    /// Tensor containing the scent at each cell in this patch. The shape of the tensor is
    /// `[N, N, S]`, where `N` is the patch width and height (all patches are square) and `S` is
    /// the scent vector size (i.e., the scent dimensionality).
    @usableFromInline let scent: ShapedArray<Float>

    /// Tensor containing the color of each cell in this patch. The shape of the tensor is
    /// `[N, N, C]`, where `N` is the patch width and height (all patches are square) and `C` is
    /// the color vector size (i.e., the color dimensionality).
    @usableFromInline let vision: ShapedArray<Float>

    /// Array containing all items in this patch (their types and positions).
    @usableFromInline let items: [ItemInformation]

    /// Array containing all agents in this patch (their positions and directions in which
    /// they are facing).
    @usableFromInline let agents: [AgentInformation]

    /// Creates a new simulation map patch.
    @inlinable
    public init(
      position: Position,
      fixed: Bool,
      scent: ShapedArray<Float>,
      vision: ShapedArray<Float>,
      items: [ItemInformation],
      agents: [AgentInformation]
    ) {
      self.position = position
      self.fixed = fixed
      self.scent = scent
      self.vision = vision
      self.items = items
      self.agents = agents
    }

    /// Creates a new simulation map patch on the Swift API side, corresponding to an exising
    /// simulation map patch on the C side.
    @inlinable
    internal init(
      fromC value: CJellyBeanWorld.SimulationMapPatch,
      using configuration: Simulator.Configuration
    ) {
      let n = Int(configuration.patchSize)
      let s = Int(configuration.scentDimensionality)
      let c = Int(configuration.colorDimensionality)
      let scentBuffer = UnsafeBufferPointer(start: value.scent, count: n * n * s)
      let scent = ShapedArray(shape: [n, n, s], scalars: Array(scentBuffer))
      let visionBuffer = UnsafeBufferPointer(start: value.vision, count: n * n * c)
      let vision = ShapedArray(shape: [n, n, c], scalars: Array(visionBuffer))
      let items = [CJellyBeanWorld.ItemInfo](
        UnsafeBufferPointer(start: value.items!, count: Int(value.numItems))
      ).map { ItemInformation(fromC: $0) }
      let agents = [CJellyBeanWorld.AgentInfo](
        UnsafeBufferPointer(start: value.agents!, count: Int(value.numAgents))
      ).map { AgentInformation(fromC: $0) }
      self.init(
        position: Position(fromC: value.position),
        fixed: value.fixed,
        scent: scent,
        vision: vision,
        items: items,
        agents: agents)
    }
  }

  /// Information about an item located in the simulation map.
  public struct ItemInformation {
    /// Item type.
    public let itemType: Int

    /// Item position in the map.
    public let position: Position

    /// Creates a new item information object.
    @inlinable
    public init(itemType: Int, position: Position) {
      self.itemType = itemType
      self.position = position
    }
  }

  /// Information about an agent located in the simulation map.
  public struct AgentInformation {
    /// Agent position.
    public let position: Position

    /// Direction in which the agent is facing.
    public let direction: Direction

    /// Creates a new agent information object.
    @inlinable
    public init(position: Position, direction: Direction) {
      self.position = position
      self.direction = direction
    }
  }
}
