import CNELFramework
import Foundation
import TensorFlow

public typealias Position = CNELFramework.Position

public enum Direction: UInt32 {
  case up = 0, down, left, right

  @inline(__always)
  internal static func fromCDirection(
    _ value: CNELFramework.Direction
  ) -> Direction {
    return Direction(rawValue: value.rawValue)!
  }

  @inline(__always)
  internal func toCDirection() -> CNELFramework.Direction {
    return CNELFramework.Direction(rawValue: self.rawValue)
  }
}

public enum TurnDirection: UInt32 {
  case front = 0, back, left, right

  @inline(__always)
  internal static func fromCTurnDirection(
    _ value: CNELFramework.TurnDirection
  ) -> TurnDirection {
    return TurnDirection(rawValue: value.rawValue)!
  }

  @inline(__always)
  internal func toCTurnDirection() -> CNELFramework.TurnDirection {
    return CNELFramework.TurnDirection(rawValue: self.rawValue)
  }
}

public enum MoveConflictPolicy: UInt32 {
  case noCollisions = 0, firstComeFirstServe, random

  @inline(__always)
  internal static func fromCMoveConflictPolicy(
    _ value: CNELFramework.MovementConflictPolicy
  ) -> MoveConflictPolicy {
    return MoveConflictPolicy(rawValue: value.rawValue)!
  }

  @inline(__always)
  internal func toCMoveConflictPolicy() -> CNELFramework.MovementConflictPolicy {
    return CNELFramework.MovementConflictPolicy(rawValue: self.rawValue)
  }
}

public final class Simulator {
  public let config: SimulatorConfig

  internal var handle: UnsafeMutableRawPointer?

  public private(set) var agents: [UInt64: Agent] = [:]

  /// Represents the number of simulation steps that have 
  /// been executed so far.
  public private(set) var time: UInt64 = 0

  private let dispatchGroup = DispatchGroup()
  private let dispatchQueue = DispatchQueue(
    label: "SimulatorDispatchQueue", 
    qos: .default, 
    attributes: .concurrent)

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
    let buffer = UnsafeBufferPointer(
      start: states!,
      count: Int(numStates))
    for state in buffer {
      simulator.agents[state.id]!.updateSimulationState(state)
    }
    if saved {
      simulator.saveAgents()
    }
    simulator.dispatchGroup.leave()
  }

  public func step() {
    self.dispatchGroup.enter()
    for agent in self.agents.values {
      self.dispatchQueue.async {
        agent.act()
      }
    }
    self.dispatchGroup.wait()
  }

  private func saveAgents() {
    
  }

  /// Adds a new agent to this simulator, and updates
  /// its simulation state.
  /// 
  /// - Parameters:
  ///   - agent: The agent to be added to this simulator.
  @inline(__always)
  internal func addAgent<A: Agent>(_ agent: A) {
    let state = CNELFramework.simulatorAddAgent(self.handle, nil)
    agent.updateSimulationState(state)
    self.agents[state.id] = agent
    CNELFramework.simulatorDeleteAgentSimulationState(state)
  }

  @inline(__always)
  internal func moveAgent(
    agent: Agent, 
    towards direction: Direction, 
    by numSteps: UInt32
  ) -> Bool {
    return CNELFramework.simulatorMoveAgent(
      self.handle, nil, agent.id!, 
      direction.toCDirection(), numSteps)
  }

  @inline(__always)
  internal func turnAgent(
    agent: Agent, 
    towards direction: TurnDirection
  ) -> Bool {
    return CNELFramework.simulatorTurnAgent(
      self.handle, nil, agent.id!, 
      direction.toCTurnDirection())
  }
}
