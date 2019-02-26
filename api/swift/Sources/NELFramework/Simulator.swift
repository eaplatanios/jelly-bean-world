import CNELFramework
import Foundation

protocol Agent {
  var simulator: Simulator { get }
  var simulationState: AgentState { get set }

  init(in simulator: Simulator)
}

extension Agent {
  public init(in simulator: Simulator) {
    self.init(in: simulator)
    self.simulationState = simulator.addAgent()
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
  
  func move(towards direction: Direction, by numSteps: UInt32) -> Bool {
    return self.simulator.moveAgent(
      agent: self,
      towards: direction,
      by: numSteps)
  }
}

protocol StatefulAgent : Agent {
  associatedtype State

  var state: State { get set }

  static func load(from file: URL) -> State
  func save(state: State, to file: URL) -> Void
}

extension StatefulAgent {
  public init(in simulator: Simulator, from file: URL) {
    self.init(in: simulator)
    self.state = Self.load(from: file)
  }
}

public class Simulator {
  internal var handle: UnsafeMutableRawPointer

  public init(
    using config: inout SimulatorConfig,
    onStep callback: @escaping @convention(c) () -> (),
    saveFrequency: UInt32, 
    savePath: String
  ) {
    self.handle = CNELFramework.simulatorCreate(
      &config,
      callback, 
      saveFrequency, 
      savePath)
  }

  public init(
    from file: URL, 
    onStep callback: @escaping @convention(c) () -> (),
    saveFrequency: UInt32, 
    savePath: String
  ) {
    self.handle = CNELFramework.simulatorLoad(
      file.absoluteString, 
      callback, 
      saveFrequency,
      savePath)
  }

  deinit {
    CNELFramework.simulatorDelete(&self.handle)
  }

  fileprivate func addAgent() -> AgentState {
    return CNELFramework.simulatorAddAgent(&self.handle, nil)
  }

  fileprivate func moveAgent(
    agent: Agent, 
    towards direction: Direction, 
    by numSteps: UInt32
  ) -> Bool {
    return CNELFramework.simulatorMoveAgent(
      &self.handle, nil, agent.simulationState.id, direction, numSteps)
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
