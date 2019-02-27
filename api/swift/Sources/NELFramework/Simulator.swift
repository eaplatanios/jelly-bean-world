import CNELFramework
import Foundation

open class Agent {
  public let simulator: Simulator

  public private(set) var simulationState: AgentSimulationState

  public init(in simulator: Simulator) {
    self.simulator = simulator
    self.simulationState = simulator.addAgent(self)
  }

  deinit {
    CNELFramework.simulatorDeleteAgentState(
      &self.simulationState)
  }

  internal func setSimulationState(
    _ state: AgentSimulationState
  ) -> Void {
    CNELFramework.simulatorDeleteAgentState(
      &self.simulationState)
    self.simulationState = state
  }
  
  @inline(__always)
  func id() -> UInt64 {
    return self.simulationState.id
  }

  @inline(__always)
  func position() -> Position {
    return self.simulationState.position
  }

  @inline(__always)
  func direction() -> Direction {
    return self.simulationState.direction
  }

  // TODO: scent, vision, collectedItems
  
  /// Moves this agent in the simulated environment.
  ///
  /// Note that the agent is not moved until the simulator 
  /// advances by a time step and issues a notification 
  /// about that event. The simulator only advances the 
  /// time step once all agents have requested to move.
  @inline(__always)
  internal func move(
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
  internal func turn(
    towards direction: TurnDirection
  ) -> Bool {
    return self.simulator.turnAgent(
      agent: self,
      towards: direction)
  }
}

protocol StatefulAgent : Agent {
  associatedtype State

  private(set) var state: State { get set }

  static func load(from file: URL) -> State
  func save(state: State, to file: URL) -> Void
}

extension StatefulAgent {
  public init(in simulator: Simulator, from file: URL) {
    super.init(in: simulator)
    self.state = Self.load(from: file)
  }
}

public class Simulator {
  internal var handle: UnsafeMutableRawPointer
  
  private var onStepCallback: () -> Void

  public private(set) var agents: [UInt64: Agent] = [:]

  /// Represents the number of simulation steps that have 
  /// been executed so far.
  public private(set) var time: UInt64

  public init(
    using config: inout SimulatorConfig,
    onStep callback: @escaping () -> Void,
    saveFrequency: UInt32, 
    savePath: String
  ) {
    let pointer = self.toRawPointer()
    self.handle = CNELFramework.simulatorCreate(
      &config,
      pointer,
      nativeOnStepCallback,
      saveFrequency, 
      savePath)
    self.onStepCallback = callback
    self.time = 0
  }

  public init(
    from file: URL, 
    onStep callback: @escaping () -> Void,
    saveFrequency: UInt32, 
    savePath: String
  ) {
    let pointer = self.toRawPointer()
    self.handle = CNELFramework.simulatorLoad(
      file.absoluteString, 
      pointer,
      nativeOnStepCallback, 
      saveFrequency,
      savePath)
    self.onStepCallback = callback
  }

  deinit {
    CNELFramework.simulatorDelete(&self.handle)
  }

  private static func fromRawPointer(
    _ pointer: UnsafeRawPointer
  ) -> Simulator {
    let unmanaged = Unmanaged<Simulator>.fromOpaque(pointer)
    return unmanaged.takeUnretainedValue()
  }

  private func toRawPointer() -> UnsafeRawPointer {
    let pointer = Unmanaged.passUnretained(self).toOpaque()
    return UnsafeRawPointer(pointer)
  }

  private let nativeOnStepCallback: @convention(c) (
      UnsafeRawPointer, 
      UnsafePointer<AgentSimulationState>?, 
      UInt32, 
      Bool) -> Void = {
    (simulatorPointer, states, numStates, saved) in {
      let simulator = Simulator.fromRawPointer(simulatorPointer)
      let buffer = UnsafeBufferPointer(
        start: states!, 
        count: Int(numStates))
      simulator.time += 1
      for state in Array(buffer) {
        simulator.agents[state.id]!.state = state
      }
      if saved {
        simulator.saveAgents()
      }
      simulator.onStepCallback()
    }
  }

  private func saveAgents() -> Void {

  }

  /// Adds a new agent to this simulator.
  /// 
  /// - Parameters:
  ///   - agent: The agent to be added to this simulator.
  /// 
  /// - Returns: The agent's simulation state.
  @inline(__always)
  internal func addAgent(_ agent: Agent) -> AgentSimulationState {
    let state = CNELFramework.simulatorAddAgent(
      &self.handle, nil)!.pointee
    self.agents[state.id] = agent
    return state
  }

  @inline(__always)
  internal func moveAgent(
    agent: Agent, 
    towards direction: Direction, 
    by numSteps: UInt32
  ) -> Bool {
    return CNELFramework.simulatorMoveAgent(
      &self.handle, nil, agent.simulationState.id, direction, numSteps)
  }

  @inline(__always)
  internal func turnAgent(
    agent: Agent, 
    towards direction: TurnDirection
  ) -> Bool {
    return CNELFramework.simulatorTurnAgent(
      &self.handle, nil, agent.simulationState.id, direction)
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
