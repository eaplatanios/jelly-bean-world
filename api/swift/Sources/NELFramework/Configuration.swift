import CNELFramework
import TensorFlow

extension Simulator {
  public struct Configuration: Equatable, Hashable {
    // Simulation parameters
    public let randomSeed: UInt32

    // Agent capabilities
    public let maxStepsPerMove: UInt32
    public let scentDimSize: UInt32
    public let colorDimSize: UInt32
    public let visionRange: UInt32
    public let allowedMoves: [Direction: ActionPolicy]
    public let allowedTurns: [TurnDirection: ActionPolicy]
    public let noOpAllowed: Bool

    // World properties
    public let patchSize: UInt32
    public let mcmcIterations: UInt32
    public let items: [Item]
    public let agentColor: ShapedArray<Float>
    public let moveConflictPolicy: MoveConflictPolicy
    
    // Scent diffusion parameters
    public let scentDecay: Float
    public let scentDiffusion: Float
    public let removedItemLifetime: UInt32

    @inlinable
    public init(
      randomSeed: UInt32,
      maxStepsPerMove: UInt32,
      scentDimSize: UInt32,
      colorDimSize: UInt32,
      visionRange: UInt32,
      allowedMoves: [Direction: ActionPolicy],
      allowedTurns: [TurnDirection: ActionPolicy],
      noOpAllowed: Bool,
      patchSize: UInt32,
      mcmcIterations: UInt32,
      items: [Item],
      agentColor: ShapedArray<Float>,
      moveConflictPolicy: MoveConflictPolicy,
      scentDecay: Float,
      scentDiffusion: Float,
      removedItemLifetime: UInt32
    ) {
      self.randomSeed = randomSeed
      self.maxStepsPerMove = maxStepsPerMove
      self.scentDimSize = scentDimSize
      self.colorDimSize = colorDimSize
      self.visionRange = visionRange
      self.allowedMoves = allowedMoves
      self.allowedTurns = allowedTurns
      self.noOpAllowed = noOpAllowed
      self.patchSize = patchSize
      self.mcmcIterations = mcmcIterations
      self.items = items
      self.agentColor = agentColor
      self.moveConflictPolicy = moveConflictPolicy
      self.scentDecay = scentDecay
      self.scentDiffusion = scentDiffusion
      self.removedItemLifetime = removedItemLifetime
    }
  }
}

public struct Item: Equatable, Hashable {
  public let name: String
  public let scent: ShapedArray<Float>
  public let color: ShapedArray<Float>
  public let requiredItemCounts: [Int: UInt32]
  public let requiredItemCosts: [Int: UInt32]
  public let blocksMovement: Bool
  public let energyFunctions: EnergyFunctions

  @inlinable
  public init(
    name: String,
    scent: ShapedArray<Float>,
    color: ShapedArray<Float>,
    requiredItemCounts: [Int: UInt32],
    requiredItemCosts: [Int: UInt32],
    blocksMovement: Bool,
    energyFunctions: EnergyFunctions
  ) {
    self.name = name
    self.scent = scent
    self.color = color
    self.requiredItemCounts = requiredItemCounts
    self.requiredItemCosts = requiredItemCosts
    self.blocksMovement = blocksMovement
    self.energyFunctions = energyFunctions
  }
}

public struct EnergyFunctions: Hashable {
  @usableFromInline internal let intensityFn: IntensityFunction
  @usableFromInline internal let interactionFns: [InteractionFunction]

  @inlinable
  public init(intensityFn: IntensityFunction, interactionFns: [InteractionFunction]) {
    self.intensityFn = intensityFn
    self.interactionFns = interactionFns
  }
}

public struct IntensityFunction: Hashable {
  @usableFromInline internal let id: UInt32
  @usableFromInline internal let arguments: [Float]

  @inlinable
  public init(id: UInt32, arguments: [Float] = []) {
    self.id = id
    self.arguments = arguments
  }
}

extension IntensityFunction {
  @inlinable
  public static func constant(_ value: Float) -> IntensityFunction {
    IntensityFunction(id: 1, arguments: [value])
  }
}

public struct InteractionFunction: Hashable {
  @usableFromInline internal let id: UInt32
  @usableFromInline internal let itemId: UInt32
  @usableFromInline internal let arguments: [Float]

  @inlinable
  public init(id: UInt32, itemId: UInt32, arguments: [Float] = []) {
    self.id = id
    self.itemId = itemId
    self.arguments = arguments
  }
}

extension InteractionFunction {
  @inlinable
  public static func piecewiseBox(
    itemId: UInt32, 
    _ firstCutoff: Float, 
    _ secondCutoff: Float,
    _ firstValue: Float,
    _ secondValue: Float
  ) -> InteractionFunction {
    InteractionFunction(
      id: 1,
      itemId: itemId,
      arguments: [
        firstCutoff, secondCutoff, 
        firstValue, secondValue])
  }

  @inlinable
  public static func cross(
    itemId: UInt32,
    _ nearCutoff: Float,
    _ farCutoff: Float,
    _ nearAxisAlignedValue: Float,
    _ nearMisalignedValue: Float,
    _ farAxisAlignedValue: Float,
    _ farMisalignedValue: Float
  ) -> InteractionFunction {
    InteractionFunction(
      id: 2,
      itemId: itemId,
      arguments: [
        nearCutoff, farCutoff,
        nearAxisAlignedValue, nearMisalignedValue, 
        farAxisAlignedValue, farMisalignedValue])
  }
}
