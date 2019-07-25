import CJellyBeanWorld

public class SimulationServer {
  internal var handle: UnsafeMutableRawPointer

  public init(
    using simulator: Simulator, 
    port: UInt32,
    connectionQueueCapacity: UInt32 = 256, 
    numWorkers: UInt32 = 8
  ) {
    self.handle = CJellyBeanWorld.simulationServerStart(
      &simulator.handle, 
      port, 
      connectionQueueCapacity,
      numWorkers)
  }

  deinit {
    // Deletes the underlying native simulator and 
    // deallocates all associated memory.
    CJellyBeanWorld.simulationServerStop(&self.handle)
  }
}

// public class SimulationClient {
//   internal let client: CJellyBeanWorld.SimulationClient

//   public init(
//     serverAddress: String, 
//     serverPort: UInt,
//     onStep onStepCallback: @escaping @convention(c) () -> (),
//     onLostConnection: onLostConnectionCallback: @escaping @convention(c) () -> (),
//     // TODO: agents and numAgents
//   ) {
//     self.client = CJellyBeanWorld.simulationClientStart(
//       serverAddress, 
//       serverPort, 
//       onStepCallback, 
//       onLostConnectionCallback, 
//       ???, 
//       ???)
//   }

//   deinit {
//     CJellyBeanWorld.simulationClientStop(self.client)
//   }
// }
