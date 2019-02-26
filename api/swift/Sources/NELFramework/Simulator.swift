import CNELFramework
import Foundation

protocol Agent {
  var state: CNELFramework.AgentState { get }
}

public class Simulator {
  internal var simulator: CNELFramework.Simulator

  public init(
    using config: inout SimulatorConfig,
    onStep callback: @escaping @convention(c) () -> (),
    saveFrequency: Int32, 
    savePath: String
  ) {
    self.simulator = CNELFramework.simulatorCreate(
      &config, 
      callback, 
      saveFrequency, 
      savePath)
  }

  public init(
    from file: URL, 
    onStep callback: @escaping @convention(c) () -> (),
    saveFrequency: Int32, 
    savePath: String
  ) {
    self.simulator = CNELFramework.simulatorLoad(
      file.absoluteString, 
      callback, 
      saveFrequency,
      savePath)
  }

  deinit {
    CNELFramework.simulatorDelete(&self.simulator)
  }

  // func addAgent() -> Agent {

  // }
}

public class SimulationServer {
  internal var server: CNELFramework.SimulationServer

  public init(
    using simulator: Simulator, 
    port: UInt32,
    connectionQueueCapacity: UInt32 = 256, 
    numWorkers: UInt32 = 8
  ) {
    self.server = CNELFramework.simulationServerStart(
      &simulator.simulator, 
      port, 
      connectionQueueCapacity, 
      numWorkers)
  }

  deinit {
    CNELFramework.simulationServerStop(&self.server)
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
