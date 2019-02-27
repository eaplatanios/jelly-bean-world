import CNELFramework
import TensorFlow

public struct Item : Equatable, Hashable {
  let name: String
  let scent: ShapedArray<Float>
  let color: ShapedArray<Float>
  let requiredItemCounts: [Item: UInt32]
  let requiredItemCosts: [Item: UInt32]
  let blocksMovement: Bool
  let intensityFn: (Position) -> Float
  let interactionFns: [(Position, Position) -> Float]

  public static func == (lhs: Item, rhs: Item) -> Bool {
    return lhs.name == rhs.name &&
      lhs.scent == rhs.scent &&
      lhs.color == rhs.color &&
      lhs.requiredItemCounts == rhs.requiredItemCounts &&
      lhs.requiredItemCosts == rhs.requiredItemCosts &&
      lhs.blocksMovement == rhs.blocksMovement
  }

  public func hash(into hasher: inout Hasher) {
    hasher.combine(name)
    hasher.combine(scent.scalars)
    hasher.combine(color.scalars)
    hasher.combine(requiredItemCounts)
    hasher.combine(requiredItemCosts)
    hasher.combine(blocksMovement)
  }

  internal func toItemProperties(
    in config: SimulatorConfig
  ) -> ItemProperties {
    return ItemProperties(
      name: name, 
      scent: scent.scalars, 
      color: color.scalars, 
      requiredItemCounts: config.items.map {
        requiredItemCounts[$0, default: 0]
      }, 
      requiredItemCosts: config.items.map {
        requiredItemCosts[$0, default: 0]
      }, 
      blocksMovement: blocksMovement, 
      intensityFn: nil, 
      interactionFns: nil, 
      intensityFnArgs: nil, 
      interactionFnArgs: nil, 
      intensityFnArgCount: 0, 
      interactionFnArgCounts: [UInt32](
        repeating: 0, 
        count: interactionFns.count))
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
  let numItems: UInt32
  let items: [Item]
  let agentColor: ShapedArray<Float>
  let moveConflictPolicy: MoveConflictPolicy
  
  // Scent diffusion parameters
  let scentDecay: Float
  let scentDiffusion: Float
  let removedItemLifetime: UInt32

  public func hash(into hasher: inout Hasher) {
    hasher.combine(randomSeed)
    hasher.combine(maxStepsPerMove)
    hasher.combine(scentDimSize)
    hasher.combine(colorDimSize)
    hasher.combine(visionRange)
    hasher.combine(allowedMoves)
    hasher.combine(allowedTurns)
    hasher.combine(patchSize)
    hasher.combine(gibbsIterations)
    hasher.combine(numItems)
    hasher.combine(items)
    hasher.combine(agentColor.scalars)
    hasher.combine(moveConflictPolicy)
    hasher.combine(scentDecay)
    hasher.combine(scentDiffusion)
    hasher.combine(removedItemLifetime)
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
