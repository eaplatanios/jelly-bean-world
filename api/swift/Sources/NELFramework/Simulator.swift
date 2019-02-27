import CNELFramework
import Foundation
import TensorFlow

extension Dictionary {
  public init(keys: [Key], values: [Value]) {
    precondition(keys.count == values.count)
    self.init()
    for (index, key) in keys.enumerated() {
      self[key] = values[index]
    }
  }
}

public enum Direction: UInt32 {
  case up = 0, down, left, right

  internal static func fromCDirection(
    _ value: CNELFramework.Direction
  ) -> Direction {
    return Direction(rawValue: value.rawValue)!
  }

  internal func toCDirection() -> CNELFramework.Direction {
    return CNELFramework.Direction(rawValue: self.rawValue)
  }
}

public enum TurnDirection: UInt32 {
  case front = 0, back, left, right

  internal static func fromCTurnDirection(
    _ value: CNELFramework.TurnDirection
  ) -> TurnDirection {
    return TurnDirection(rawValue: value.rawValue)!
  }

  internal func toCTurnDirection() -> CNELFramework.TurnDirection {
    return CNELFramework.TurnDirection(rawValue: self.rawValue)
  }
}

public enum MoveConflictPolicy: UInt32 {
  case noCollisions = 0, firstComeFirstServe, random

  internal static func fromCMoveConflictPolicy(
    _ value: CNELFramework.MovementConflictPolicy
  ) -> MoveConflictPolicy {
    return MoveConflictPolicy(rawValue: value.rawValue)!
  }

  internal func toCMoveConflictPolicy() -> CNELFramework.MovementConflictPolicy {
    return CNELFramework.MovementConflictPolicy(rawValue: self.rawValue)
  }
}

public struct Item : Equatable, Hashable {
  let name: String
  let scent: ShapedArray<Float>
  let color: ShapedArray<Float>
  let requiredItemCounts: [Item: UInt32]
  let requiredItemCosts: [Item: UInt32]
  let blocksMovement: Bool
  let intensityFn: (Position) -> Float
  let interactionFns: [(Position, Position) -> Float]

  public static func == (lhs: Item, rhs: Item) -> Bool {
    return lhs.name == rhs.name &&
      lhs.scent == rhs.scent &&
      lhs.color == rhs.color &&
      lhs.requiredItemCounts == rhs.requiredItemCounts &&
      lhs.requiredItemCosts == rhs.requiredItemCosts &&
      lhs.blocksMovement == rhs.blocksMovement
  }

  public func hash(into hasher: inout Hasher) {
    hasher.combine(name)
    hasher.combine(scent.scalars)
    hasher.combine(color.scalars)
    hasher.combine(requiredItemCounts)
    hasher.combine(requiredItemCosts)
    hasher.combine(blocksMovement)
  }

  internal func toItemProperties(
    in environment: Environment
  ) -> ItemProperties {
    return ItemProperties(
      name: name, 
      scent: scent.scalars, 
      color: color.scalars, 
      requiredItemCounts: environment.items.map {
        requiredItemCounts[$0] ?? 0
      }, 
      requiredItemCosts: environment.items.map {
        requiredItemCosts[$0] ?? 0
      }, 
      blocksMovement: blocksMovement, 
      intensityFn: nil, 
      interactionFns: nil, 
      intensityFnArgs: nil, 
      interactionFnArgs: nil, 
      intensityFnArgCount: 0, 
      interactionFnArgCounts: [UInt32](
        repeating: 0, 
        count: interactionFns.count))
  }
}

public struct Environment : Equatable, Hashable {
  // Simulation parameters
  let randomSeed: UInt32

  // Agent capabilities
  let maxStepsPerMove: UInt32
  let scentDimSize: UInt32
  let colorDimSize: UInt32
  let visionRange: UInt32
  let allowedMoves: Set<Direction>
  let allowedTurns: Set<TurnDirection>

  // World properties
  let patchSize: UInt32
  let gibbsIterations: UInt32
  let numItems: UInt32
  let items: [Item]
  let agentColor: ShapedArray<Float>
  let moveConflictPolicy: MoveConflictPolicy
  
  // Scent diffusion parameters
  let scentDecay: Float
  let scentDiffusion: Float
  let removedItemLifetime: UInt32

  public func hash(into hasher: inout Hasher) {
    hasher.combine(randomSeed)
    hasher.combine(maxStepsPerMove)
    hasher.combine(scentDimSize)
    hasher.combine(colorDimSize)
    hasher.combine(visionRange)
    hasher.combine(allowedMoves)
    hasher.combine(allowedTurns)
    hasher.combine(patchSize)
    hasher.combine(gibbsIterations)
    hasher.combine(numItems)
    hasher.combine(items)
    hasher.combine(agentColor.scalars)
    hasher.combine(moveConflictPolicy)
    hasher.combine(scentDecay)
    hasher.combine(scentDiffusion)
    hasher.combine(removedItemLifetime)
  }

  internal func toSimulatorConfig() -> SimulatorConfig {
    return SimulatorConfig(
      randomSeed: randomSeed, 
      maxStepsPerMove: maxStepsPerMove, 
      scentDimSize: scentDimSize, 
      colorDimSize: colorDimSize, 
      visionRange: visionRange, 
      allowedMoveDirections: (
        allowedMoves.contains(.up),
        allowedMoves.contains(.down),
        allowedMoves.contains(.left),
        allowedMoves.contains(.right)), 
      allowedRotations: (
        allowedTurns.contains(.front),
        allowedTurns.contains(.back),
        allowedTurns.contains(.left),
        allowedTurns.contains(.right)),
      patchSize: patchSize, 
      gibbsIterations: gibbsIterations, 
      itemTypes: items.map { $0.toItemProperties(in: self) },
      numItemTypes: UInt32(items.count),
      agentColor: agentColor.scalars, 
      movementConflictPolicy: moveConflictPolicy.toCMoveConflictPolicy(), 
      scentDecay: scentDecay, 
      scentDiffusion: scentDiffusion, 
      removedItemLifetime: removedItemLifetime)
  }
}

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
    self.scent = simulator.scentToArray(state.scent!)
    self.vision = simulator.visionToArray(state.vision!)
    self.items = simulator.itemCountsToDictionary(state.collectedItems!)
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

public class Simulator {
  let environment: Environment

  internal var handle: UnsafeMutableRawPointer?
  
  private var onStepCallback: () -> Void = {}

  public private(set) var agents: [UInt64: Agent] = [:]

  /// Represents the number of simulation steps that have 
  /// been executed so far.
  public private(set) var time: UInt64 = 0

  public init(
    using environment: Environment,
    onStep callback: @escaping () -> Void,
    saveFrequency: UInt32, 
    savePath: String
  ) {
    self.environment = environment
    self.onStepCallback = callback
    var config = environment.toSimulatorConfig()
    let opaque = Unmanaged.passUnretained(self).toOpaque()
    let pointer = UnsafeMutableRawPointer(opaque)
    self.handle = CNELFramework.simulatorCreate(
      &config,
      nativeOnStepCallback,
      pointer,
      saveFrequency, 
      savePath)
  }

  // public init(
  //   from file: URL, 
  //   onStep callback: @escaping () -> Void,
  //   saveFrequency: UInt32, 
  //   savePath: String
  // ) {
  //   self.handle = CNELFramework.simulatorLoad(
  //     file.absoluteString, 
  //     self.toRawPointer(),
  //     nativeOnStepCallback, 
  //     saveFrequency,
  //     savePath)
  //   self.onStepCallback = callback
  // }

  deinit {
    CNELFramework.simulatorDelete(&self.handle)
  }

  private static func fromRawPointer(
    _ pointer: UnsafeMutableRawPointer
  ) -> Simulator {
    let unmanaged = Unmanaged<Simulator>.fromOpaque(pointer)
    return unmanaged.takeUnretainedValue()
  }

  private let nativeOnStepCallback: @convention(c) (
      UnsafeMutableRawPointer?, 
      UnsafePointer<AgentSimulationState>?,
      UInt32, 
      Bool) -> Void = { (simulatorPointer, states, numStates, saved) in 
    let simulator = Simulator.fromRawPointer(simulatorPointer!)
    let buffer = UnsafeBufferPointer(
      start: states!, 
      count: Int(numStates))
    simulator.time += 1
    for state in buffer {
      simulator.agents[state.id]!.updateSimulationState(state)
    }
    if saved {
      simulator.saveAgents()
    }
    simulator.onStepCallback()
  }

  internal func scentToArray(
    _ buffer: UnsafeMutablePointer<Float>
  ) -> ShapedArray<Float> {
    let scentShape = [Int(environment.scentDimSize)]
    let scentBuffer = UnsafeBufferPointer(
        start: buffer,
        count: Int(environment.scentDimSize))
    return ShapedArray(
      shape: scentShape,
      scalars: scentBuffer)
  }

  internal func visionToArray(
    _ buffer: UnsafeMutablePointer<Float>
  ) -> ShapedArray<Float> {
    let visionShape = [
      2 * Int(environment.visionRange) + 1, 
      2 * Int(environment.visionRange) + 1, 
      Int(environment.colorDimSize)]
    let visionBuffer = UnsafeBufferPointer(
        start: buffer,
        count: Int(
          4 * environment.visionRange + 2 + 
          environment.colorDimSize))
    return ShapedArray(
      shape: visionShape,
      scalars: visionBuffer)
  }

  internal func itemCountsToDictionary(
    _ countsPointer: UnsafeMutablePointer<UInt32>
  ) -> [Item: UInt32] {
    let counts = Array(UnsafeBufferPointer(
      start: countsPointer, 
      count: Int(environment.items.count)))
    return Dictionary(
      keys: environment.items, 
      values: counts)
  }

  private func saveAgents() -> Void {

  }

  /// Adds a new agent to this simulator, and updates
  /// its simulation state.
  /// 
  /// - Parameters:
  ///   - agent: The agent to be added to this simulator.
  @inline(__always)
  internal func addAgent<A: Agent>(_ agent: A) {
    let state = CNELFramework.simulatorAddAgent(&self.handle, nil)
    agent.updateSimulationState(state)
    CNELFramework.simulatorDeleteAgentSimulationState(state)
    self.agents[state.id] = agent
  }

  @inline(__always)
  internal func moveAgent(
    agent: Agent, 
    towards direction: Direction, 
    by numSteps: UInt32
  ) -> Bool {
    return CNELFramework.simulatorMoveAgent(
      &self.handle, nil, agent.id, 
      direction.toCDirection(), numSteps)
  }

  @inline(__always)
  internal func turnAgent(
    agent: Agent, 
    towards direction: TurnDirection
  ) -> Bool {
    return CNELFramework.simulatorTurnAgent(
      &self.handle, nil, agent.id, 
      direction.toCTurnDirection())
  }
}

public class SimulationServer {
  internal var handle: UnsafeMutableRawPointer

  public init(
    using simulator: Simulator, 
    port: UInt32,
    connectionQueueCapacity: UInt32 = 256, 
    numWorkers: UInt32 = 8
  ) {
    self.handle = CNELFramework.simulationServerStart(
      &simulator.handle, 
      port, 
      connectionQueueCapacity,
      numWorkers)
  }

  deinit {
    // Deletes the underlying native simulator and 
    // deallocates all associated memory.
    CNELFramework.simulationServerStop(&self.handle)
  }
}

// public class SimulationClient {
//   internal let client: CNELFramework.SimulationClient

//   public init(
//     serverAddress: String, 
//     serverPort: UInt,
//     onStep onStepCallback: @escaping @convention(c) () -> (),
//     onLostConnection: onLostConnectionCallback: @escaping @convention(c) () -> (),
//     // TODO: agents and numAgents
//   ) {
//     self.client = CNELFramework.simulationClientStart(
//       serverAddress, 
//       serverPort, 
//       onStepCallback, 
//       onLostConnectionCallback, 
//       ???, 
//       ???)
//   }

//   deinit {
//     CNELFramework.simulationClientStop(self.client)
//   }
// }
