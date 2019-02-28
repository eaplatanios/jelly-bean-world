import CNELFramework
import TensorFlow

internal typealias CSimulationMap = CNELFramework.SimulationMap
internal typealias CSimulationMapPatch = CNELFramework.SimulationMapPatch
internal typealias CSimulationMapItemInfo = CNELFramework.ItemInfo
internal typealias CSimulationMapAgentInfo = CNELFramework.AgentInfo

public typealias SimulationMap = [SimulationMapPatch]

extension Array where Element == SimulationMapPatch {
  @inline(__always)
  internal static func fromC(
    _ value: CSimulationMap,
    for simulator: Simulator
  ) -> SimulationMap {
    let cPatches = UnsafeBufferPointer(start: value.patches!, count: Int(value.numPatches))
    return cPatches.map { SimulationMapPatch.fromC($0, for: simulator) }
  }
}

public struct SimulationMapPatch {
  let position: Position
  let fixed: Bool
  let scent: ShapedArray<Float>
  let vision: ShapedArray<Float>
  let items: [CSimulationMapItemInfo]
  let agents: [CSimulationMapAgentInfo]

  internal init(
    position: Position,
    fixed: Bool,
    scent: ShapedArray<Float>,
    vision: ShapedArray<Float>,
    items: [CSimulationMapItemInfo],
    agents: [CSimulationMapAgentInfo]
  ) {
    self.position = position
    self.fixed = fixed
    self.scent = scent
    self.vision = vision
    self.items = items
    self.agents = agents
  }

  @inline(__always)
  internal static func fromC(
    _ value: CSimulationMapPatch,
    for simulator: Simulator
  ) -> SimulationMapPatch {
    let n = Int(simulator.config.patchSize)
    let s = Int(simulator.config.scentDimSize)
    let c = Int(simulator.config.colorDimSize)
    let scentBuffer = UnsafeBufferPointer(start: value.scent, count: n * n * s)
    let scent = ShapedArray(shape: [n, n, s], scalars: Array(scentBuffer))
    let visionBuffer = UnsafeBufferPointer(start: value.vision, count: n * n * c)
    let vision = ShapedArray(shape: [n, n, c], scalars: Array(visionBuffer))
    return SimulationMapPatch(
      position: value.position,
      fixed: value.fixed,
      scent: scent,
      vision: vision,
      items: Array(UnsafeBufferPointer(start: value.items!, count: Int(value.numItems))),
      agents: Array(UnsafeBufferPointer(start: value.agents!, count: Int(value.numAgents))))
  }
}
