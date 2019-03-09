import CNELFramework
import Foundation
import TensorFlow

@usableFromInline
internal typealias CPosition = CNELFramework.Position

@usableFromInline
internal typealias CDirection = CNELFramework.Direction

@usableFromInline
internal typealias CTurnDirection = CNELFramework.TurnDirection

@usableFromInline
internal typealias CMovementConflictPolicy = CNELFramework.MovementConflictPolicy

public struct Position: Equatable {
  public let x: Int64
  public let y: Int64

  public init(x: Int64, y: Int64) {
    self.x = x
    self.y = y
  }

  @inlinable
  internal static func fromC(_ value: CPosition) -> Position {
    return Position(x: value.x, y: value.y)
  }

  @inlinable
  internal func toC() -> CPosition {
    return CPosition(x: x, y: y)
  }
}

public enum Direction: UInt32 {
  case up = 0, down, left, right

  @inlinable
  internal static func fromC(_ value: CDirection) -> Direction {
    return Direction(rawValue: value.rawValue)!
  }

  @inlinable
  internal func toC() -> CDirection {
    return CDirection(rawValue: self.rawValue)
  }
}

public enum TurnDirection: UInt32 {
  case front = 0, back, left, right

  @inlinable
  internal static func fromC(_ value: CTurnDirection) -> TurnDirection {
    return TurnDirection(rawValue: value.rawValue)!
  }

  @inlinable
  internal func toC() -> CTurnDirection {
    return CTurnDirection(rawValue: self.rawValue)
  }
}

public enum MoveConflictPolicy: UInt32 {
  case noCollisions = 0, firstComeFirstServe, random

  @inlinable
  internal static func fromC(_ value: CMovementConflictPolicy) -> MoveConflictPolicy {
    return MoveConflictPolicy(rawValue: value.rawValue)!
  }

  @inlinable
  internal func toC() -> CMovementConflictPolicy {
    return CMovementConflictPolicy(rawValue: self.rawValue)
  }
}

public final class Simulator {
  public let config: SimulatorConfig

  @usableFromInline
  internal var handle: UnsafeMutableRawPointer?

  public private(set) var agents: [UInt64: Agent] = [:]

  /// Represents the number of simulation steps that have 
  /// been executed so far.
  public private(set) var time: UInt64 = 0

  private let dispatchSemaphore = DispatchSemaphore(value: 1)
  private let dispatchQueue = DispatchQueue(
    label: "SimulatorDispatchQueue", 
    qos: .default, 
    attributes: .concurrent)
  private var usingDispatchQueue = false

  public init(
    using config: SimulatorConfig,
    saveFrequency: UInt32 = 1000, 
    savePath: String? = nil
  ) {
    self.config = config
    var cConfig = config.toC()
    let pointer = Unmanaged.passUnretained(self).toOpaque()
    self.handle = CNELFramework.simulatorCreate(
      &cConfig.simulatorConfig,
      nativeOnStepCallback,
      pointer,
      saveFrequency, 
      savePath)
    cConfig.deallocate()
  }

  // public init(
  //   using config: SimulatorConfig,
  //   from file: URL,
  //   saveFrequency: UInt32,
  //   savePath: String
  // ) {
  //   self.config = config
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

  private let nativeOnStepCallback: @convention(c) (
      UnsafeRawPointer?, 
      UnsafePointer<AgentSimulationState>?,
      UInt32, 
      Bool) -> Void = { (simulatorPointer, states, numStates, saved) in 
    let unmanagedSimulator = Unmanaged<Simulator>.fromOpaque(simulatorPointer!)
    let simulator = unmanagedSimulator.takeUnretainedValue()
    simulator.time += 1
    let buffer = UnsafeBufferPointer(start: states!, count: Int(numStates))
    for state in buffer {
      simulator.agents[state.id]!.updateSimulationState(state)
    }
    if saved { simulator.saveAgents() }
    if simulator.usingDispatchQueue {
      simulator.dispatchSemaphore.signal()
    }
  }

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

  internal func saveAgents() {
    // TODO
  }

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
    let state = CNELFramework.simulatorAddAgent(self.handle, nil)
    agent.updateSimulationState(state)
    self.agents[state.id] = agent
    CNELFramework.simulatorDeleteAgentSimulationState(state)
  }

  @inlinable
  internal func moveAgent(agent: Agent, towards direction: Direction, by numSteps: UInt32) -> Bool {
    return CNELFramework.simulatorMoveAgent(self.handle, nil, agent.id!, direction.toC(), numSteps)
  }

  @inlinable
  internal func turnAgent(agent: Agent, towards direction: TurnDirection) -> Bool {
    return CNELFramework.simulatorTurnAgent(self.handle, nil, agent.id!, direction.toC())
  }

  // TODO: Map deallocator.
  @inlinable
  internal func map(bottomLeft: Position, topRight: Position) -> SimulationMap {
    let cSimulationMap = CNELFramework.simulatorMap(
      self.handle, nil, bottomLeft.toC(), topRight.toC())
    return SimulationMap.fromC(cSimulationMap, for: self)
  }
}
