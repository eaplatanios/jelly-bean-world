import CNELFramework
import TensorFlow

/// A simulation map.
public struct SimulationMap {
  public let patches: [Patch]

  /// Creates a new simulation map which consists of the provided patches.
  /// - Parameter patches: Patches comprising the new simulation map.
  @inlinable
  public init(patches: [Patch]) {
    self.patches = patches
  }

  /// Creates a new simulation map on the Swift API side, corresponding to an exising simulation
  /// map on the C API side.
  @inlinable
  internal init(fromC value: CNELFramework.SimulationMap, for simulator: Simulator) {
    let cPatches = UnsafeBufferPointer(start: value.patches!, count: Int(value.numPatches))
    self.patches = cPatches.map { Patch(fromC: $0, for: simulator) }
  }
}

extension SimulationMap {
  /// A patch of the simulation map.
  public struct Patch {
    /// Position of the patch.
    @usableFromInline let position: Position

    /// Flag indicating whether the patch has been sampled and fixed or whether it was sampled as a 
    /// patch in the map boundaries that may later be resampled.
    @usableFromInline let fixed: Bool

    /// Tensor containing the scent at each cell in this patch. The shape of the tensor is
    /// `[N, N, S]`, where `N` is the patch width and height (all patches are square) and `S` is
    /// the scent vector size (i.e., the scent dimensionality).
    @usableFromInline let scent: ShapedArray<Float>

    /// Tensor containing the color of each cell in this patch. The shape of the tensor is
    /// `[N, N, C]`, where `N` is the patch width and height (all patches are square) and `C` is
    /// the color vector size (i.e., the color dimensionality).
    @usableFromInline let vision: ShapedArray<Float>

    /// Array containing all items in this patch (their types and positions).
    @usableFromInline let items: [ItemInformation]

    /// Array containing all agents in this patch (their positions and directions in which
    /// they are facing).
    @usableFromInline let agents: [AgentInformation]

    /// Creates a new simulation map patch.
    @inlinable
    public init(
      position: Position,
      fixed: Bool,
      scent: ShapedArray<Float>,
      vision: ShapedArray<Float>,
      items: [ItemInformation],
      agents: [AgentInformation]
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
    internal init(fromC value: CNELFramework.SimulationMapPatch, for simulator: Simulator) {
      let n = Int(simulator.configuration.patchSize)
      let s = Int(simulator.configuration.scentDimSize)
      let c = Int(simulator.configuration.colorDimSize)
      let scentBuffer = UnsafeBufferPointer(start: value.scent, count: n * n * s)
      let scent = ShapedArray(shape: [n, n, s], scalars: Array(scentBuffer))
      let visionBuffer = UnsafeBufferPointer(start: value.vision, count: n * n * c)
      let vision = ShapedArray(shape: [n, n, c], scalars: Array(visionBuffer))
      let items = [CNELFramework.ItemInfo](
        UnsafeBufferPointer(start: value.items!, count: Int(value.numItems))
      ).map { ItemInformation(fromC: $0) }
      let agents = [CNELFramework.AgentInfo](
        UnsafeBufferPointer(start: value.agents!, count: Int(value.numAgents))
      ).map { AgentInformation(fromC: $0) }
      self.init(
        position: Position(fromC: value.position),
        fixed: value.fixed,
        scent: scent,
        vision: vision,
        items: items,
        agents: agents)
    }
  }

  /// Information about an item located in the simulation map.
  public struct ItemInformation {
    /// Item type.
    public let itemType: Int

    /// Item position in the map.
    public let position: Position

    /// Creates a new item information object.
    @inlinable
    public init(itemType: Int, position: Position) {
      self.itemType = itemType
      self.position = position
    }

    /// Creates a new item information object on the Swift API side, corresponding to an exising
    /// item information object on the C API side.
    @inlinable
    internal init(fromC value: CNELFramework.ItemInfo) {
      self.itemType = Int(value.type)
      self.position = Position(fromC: value.position)
    }
  }

  /// Information about an agent located in the simulation map.
  public struct AgentInformation {
    /// Agent position.
    public let position: Position

    /// Direction in which the agent is facing.
    public let direction: Direction

    /// Creates a new agent information object.
    @inlinable
    public init(position: Position, direction: Direction) {
      self.position = position
      self.direction = direction
    }

    /// Creates a new agent information object on the Swift API side, corresponding to an exising
    /// agent information object on the C API side.
    @inlinable
    internal init(fromC value: CNELFramework.AgentInfo) {
      self.position = Position(fromC: value.position)
      self.direction = Direction(fromC: value.direction)
    }
  }
}
