import CNELFramework
import Foundation
import TensorFlow

public enum JellyBeanWorldError: Error {
  /// Failure while trying to save the simulator state at the specified path.
  case SimulatorSaveFailure(URL)
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

/// Conflict resolution policy for when multiple agents request to move to the same location.
public enum MoveConflictPolicy: UInt32 {
  case noCollisions = 0, firstComeFirstServe, random
}

/// Policy that determines whether specific actions are allowed, disallowed, or simply allowed but
/// ignored by the simulator (i.e., nothing happens if an ignored action is requested).
public enum ActionPolicy: UInt32 {
  case allowed = 0, disallowed, ignored
}

/// Jelly Bean World (JBW) simulator.
public final class Simulator {
  /// Simulator configuration.
  public let configuration: Configuration

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
  public init(using configuration: Configuration) {
    self.configuration = configuration
    var cConfig = configuration.toC()
    defer { cConfig.deallocate() }
    self.handle = simulatorCreate(&cConfig.configuration, nativeOnStepCallback)
    simulatorSetStepCallbackData(handle, Unmanaged.passUnretained(self).toOpaque())
  }

  /// Loads a simulator from the provided file.
  ///
  /// - Parameters:
  ///   - file: File in which the simulator is saved.
  ///   - agents: Agents that this simulator manages.
  /// - Precondition: The number of agents provided must match the number of agents the simulator
  ///   managed before its state was saved.
  public init(fromFile file: URL, agents: [Agent]) {
    let cSimulatorInfo = simulatorLoad(file.absoluteString, nativeOnStepCallback)
    defer { simulatorDeleteSimulatorInfo(cSimulatorInfo) }
    self.handle = cSimulatorInfo.handle
    self.configuration = Configuration(fromC: cSimulatorInfo.config)
    self.time = cSimulatorInfo.time
    self.agentStates = [UInt64: AgentState](
      uniqueKeysWithValues: UnsafeBufferPointer(
        start: cSimulatorInfo.agents!,
        count: Int(cSimulatorInfo.numAgents)
      ).map { ($0.id, AgentState(fromC: $0, for: self)) })
    precondition(
      agents.count == agentStates.count,
      """
      The number of agent states stored in the provided simulator file does not match
      the number of agents provided.
      """)
    self.agents = [UInt64: Agent](uniqueKeysWithValues: zip(agentStates.keys, agents))
    simulatorSetStepCallbackData(handle, Unmanaged.passUnretained(self).toOpaque())
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

  /// Saves this simulator in the provided file.
  ///
  /// - Parameter file: File in which to save the state of this simulator.
  @inlinable
  public func save(to file: URL) throws {
    if !simulatorSave(handle, file.absoluteString) {
      throw JellyBeanWorldError.SimulatorSaveFailure(file)
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

    @inlinable
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

public struct EnergyFunctions: Hashable {
  @usableFromInline internal let intensityFn: IntensityFunction
  @usableFromInline internal let interactionFns: [InteractionFunction]

  @inlinable
  public init(intensityFn: IntensityFunction, interactionFns: [InteractionFunction]) {
    self.intensityFn = intensityFn
    self.interactionFns = interactionFns
  }
}

public struct IntensityFunction: Hashable {
  @usableFromInline internal let id: UInt32
  @usableFromInline internal let arguments: [Float]

  @inlinable
  public init(id: UInt32, arguments: [Float] = []) {
    self.id = id
    self.arguments = arguments
  }
}

extension IntensityFunction {
  @inlinable
  public static func constant(_ value: Float) -> IntensityFunction {
    IntensityFunction(id: 1, arguments: [value])
  }
}

public struct InteractionFunction: Hashable {
  @usableFromInline internal let id: UInt32
  @usableFromInline internal let itemId: UInt32 // TODO: Convert to Item.
  @usableFromInline internal let arguments: [Float]

  @inlinable
  public init(id: UInt32, itemId: UInt32, arguments: [Float] = []) {
    self.id = id
    self.itemId = itemId
    self.arguments = arguments
  }
}

extension InteractionFunction {
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
