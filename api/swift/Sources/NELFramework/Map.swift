import CNELFramework
import TensorFlow

/// Pointer to a simulation map on the C API side.
@usableFromInline internal typealias CSimulationMap = CNELFramework.SimulationMap

/// Pointer to a simulation map patch on the C API side.
@usableFromInline internal typealias CSimulationMapPatch = CNELFramework.SimulationMapPatch

/// Pointer to an item information struct on the C API side. The item information consists of the
/// item's type and position in the map.
@usableFromInline internal typealias CSimulationMapItemInfo = CNELFramework.ItemInfo

/// Pointer to an agent information struct on the C API side. The agent information consists of the
/// agent's position in the map and the direction in which it is facing.
@usableFromInline internal typealias CSimulationMapAgentInfo = CNELFramework.AgentInfo

/// A simulation map is simply a collection of all the patches that make it up.
public typealias SimulationMap = [SimulationMapPatch]

/// Helper extension for loading simulation map patches from the C API of the simulator.
extension Array where Element == SimulationMapPatch {
  /// Creates a new simulation map on the Swift API side, corresponding to an exising simulation
  /// map on the C API side. Note that a simulation map is simply an array containing all patches
  /// in the simulation map.
  @inlinable
  internal init(fromC value: CSimulationMap, for simulator: Simulator) {
    let cPatches = UnsafeBufferPointer(start: value.patches!, count: Int(value.numPatches))
    self.init(cPatches.map { SimulationMapPatch(fromC: $0, for: simulator) })
  }
}

/// A patch of the simulation map.
public struct SimulationMapPatch {
  /// Position of the patch.
  @usableFromInline let position: Position

  /// Flag indicating whether the patch has been sampled and fixed or whether it was sampled as a 
  /// patch in the map boundaries that may later be resampled.
  @usableFromInline let fixed: Bool

  /// Tensor containing the scent at each cell in this patch. The shape of the tensor is
  /// `[N, N, S]`, where `N` is the patch width and height (all patches are square) and `S` is the
  /// scent vector size (i.e., the scent dimensionality).
  @usableFromInline let scent: ShapedArray<Float>

  /// Tensor containing the color of each cell in this patch. The shape of the tensor is
  /// `[N, N, C]`, where `N` is the patch width and height (all patches are square) and `C` is the
  /// color vector size (i.e., the color dimensionality).
  @usableFromInline let vision: ShapedArray<Float>

  /// Array containing all items in this patch (their types and positions).
  @usableFromInline let items: [CSimulationMapItemInfo]

  /// Array containing all agents in this patch (their positions and directions in which
  /// they are facing).
  @usableFromInline let agents: [CSimulationMapAgentInfo]

  /// Creates a new simulation map patch.
  @inlinable
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

  /// Creates a new simulation map patch on the Swift API side, corresponding to an exising
  /// simulation map patch on the C side.
  @inlinable
  internal init(fromC value: CSimulationMapPatch, for simulator: Simulator) {
    let n = Int(simulator.config.patchSize)
    let s = Int(simulator.config.scentDimSize)
    let c = Int(simulator.config.colorDimSize)
    let scentBuffer = UnsafeBufferPointer(start: value.scent, count: n * n * s)
    let scent = ShapedArray(shape: [n, n, s], scalars: Array(scentBuffer))
    let visionBuffer = UnsafeBufferPointer(start: value.vision, count: n * n * c)
    let vision = ShapedArray(shape: [n, n, c], scalars: Array(visionBuffer))
    self.init(
      position: Position.fromC(value.position),
      fixed: value.fixed,
      scent: scent,
      vision: vision,
      items: Array(UnsafeBufferPointer(start: value.items!, count: Int(value.numItems))),
      agents: Array(UnsafeBufferPointer(start: value.agents!, count: Int(value.numAgents))))
  }
}
