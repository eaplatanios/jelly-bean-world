import CNELFramework

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
