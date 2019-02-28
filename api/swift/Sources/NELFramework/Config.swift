import CNELFramework
import TensorFlow

public struct Item : Equatable, Hashable {
  let name: String
  let scent: ShapedArray<Float>
  let color: ShapedArray<Float>
  let requiredItemCounts: [Int: UInt32]
  let requiredItemCosts: [Int: UInt32]
  let blocksMovement: Bool
  let energyFunctions: EnergyFunctions

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

  internal func toItemProperties(
    in config: SimulatorConfig
  ) -> ItemProperties {
    return ItemProperties(
      name: name, 
      scent: scent.scalars,
      color: color.scalars,
      requiredItemCounts: config.items.indices.map {
        requiredItemCounts[$0, default: 0]
      }, 
      requiredItemCosts: config.items.indices.map {
        requiredItemCosts[$0, default: 0]
      }, 
      blocksMovement: blocksMovement,
      energyFunctions: energyFunctions.toC())
  }
}

public struct SimulatorConfig : Equatable, Hashable {
  // Simulation parameters
  let randomSeed: UInt32

  // Agent capabilities
  let maxStepsPerMove: UInt32
  let scentDimSize: UInt32
  let colorDimSize: UInt32
  let visionRange: UInt32
  let allowedMoves: Set<Direction>
  let allowedTurns: Set<TurnDirection>

  // World properties
  let patchSize: UInt32
  let gibbsIterations: UInt32
  let items: [Item]
  let agentColor: ShapedArray<Float>
  let moveConflictPolicy: MoveConflictPolicy
  
  // Scent diffusion parameters
  let scentDecay: Float
  let scentDiffusion: Float
  let removedItemLifetime: UInt32

  public init(
    randomSeed: UInt32,
    maxStepsPerMove: UInt32,
    scentDimSize: UInt32,
    colorDimSize: UInt32,
    visionRange: UInt32,
    allowedMoves: Set<Direction>,
    allowedTurns: Set<TurnDirection>,
    patchSize: UInt32,
    gibbsIterations: UInt32,
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
    self.patchSize = patchSize
    self.gibbsIterations = gibbsIterations
    self.items = items
    self.agentColor = agentColor
    self.moveConflictPolicy = moveConflictPolicy
    self.scentDecay = scentDecay
    self.scentDiffusion = scentDiffusion
    self.removedItemLifetime = removedItemLifetime
  }

  internal func toCSimulatorConfig() -> CNELFramework.SimulatorConfig {
    return CNELFramework.SimulatorConfig(
      randomSeed: randomSeed, 
      maxStepsPerMove: maxStepsPerMove, 
      scentDimSize: scentDimSize, 
      colorDimSize: colorDimSize, 
      visionRange: visionRange, 
      allowedMoveDirections: (
        allowedMoves.contains(.up),
        allowedMoves.contains(.down),
        allowedMoves.contains(.left),
        allowedMoves.contains(.right)), 
      allowedRotations: (
        allowedTurns.contains(.front),
        allowedTurns.contains(.back),
        allowedTurns.contains(.left),
        allowedTurns.contains(.right)),
      patchSize: patchSize, 
      gibbsIterations: gibbsIterations, 
      itemTypes: items.map { $0.toItemProperties(in: self) },
      numItemTypes: UInt32(items.count),
      agentColor: agentColor.scalars, 
      movementConflictPolicy: moveConflictPolicy.toCMoveConflictPolicy(), 
      scentDecay: scentDecay, 
      scentDiffusion: scentDiffusion, 
      removedItemLifetime: removedItemLifetime)
  }
}
