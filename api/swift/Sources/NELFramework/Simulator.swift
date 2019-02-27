import CNELFramework
import Foundation
import TensorFlow

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

public class Simulator {
  let config: SimulatorConfig

  internal var handle: UnsafeMutableRawPointer?
  
  private var onStepCallback: () -> Void = {}

  public private(set) var agents: [UInt64: Agent] = [:]

  /// Represents the number of simulation steps that have 
  /// been executed so far.
  public private(set) var time: UInt64 = 0

  public init(
    using config: SimulatorConfig,
    onStep callback: @escaping () -> Void,
    saveFrequency: UInt32, 
    savePath: String
  ) {
    self.config = config
    self.onStepCallback = callback
    var cConfig = config.toCSimulatorConfig()
    let opaque = Unmanaged.passUnretained(self).toOpaque()
    let pointer = UnsafeMutableRawPointer(opaque)
    self.handle = CNELFramework.simulatorCreate(
      &cConfig,
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

  private let nativeOnStepCallback: @convention(c) (
      UnsafeMutableRawPointer?, 
      UnsafePointer<AgentSimulationState>?,
      UInt32, 
      Bool) -> Void = { (simulatorPointer, states, numStates, saved) in 
    let unmanagedSimulator = Unmanaged<Simulator>.fromOpaque(simulatorPointer!)
    let simulator = unmanagedSimulator.takeUnretainedValue()
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
