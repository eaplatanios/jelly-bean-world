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

  internal func toC(
    in config: SimulatorConfig
  ) -> (
    itemProperties: CNELFramework.ItemProperties, 
    deallocate: () -> Void
  ) {
    let scent = self.scent.scalars
    let color = self.color.scalars
    let counts = config.items.indices.map { requiredItemCounts[$0, default: 0] }
    let costs = config.items.indices.map { requiredItemCosts[$0, default: 0] }

    // Allocate C arrays and copy data.
    let cScent = UnsafeMutablePointer<Float>.allocate(capacity: scent.count)
    let cColor = UnsafeMutablePointer<Float>.allocate(capacity: color.count)
    let cCounts = UnsafeMutablePointer<UInt32>.allocate(capacity: counts.count)
    let cCosts = UnsafeMutablePointer<UInt32>.allocate(capacity: costs.count)
    cScent.initialize(from: scent, count: scent.count)
    cColor.initialize(from: color, count: color.count)
    cCounts.initialize(from: counts, count: counts.count)
    cCosts.initialize(from: costs, count: costs.count)

    let cEnergyFunctions = energyFunctions.toC()

    return (
      itemProperties: CNELFramework.ItemProperties(
        name: name,
        scent: cScent,
        color: cColor,
        requiredItemCounts: cCounts, 
        requiredItemCosts: cCosts, 
        blocksMovement: blocksMovement,
        energyFunctions: cEnergyFunctions.energyFunctions),
      deallocate: { () in 
        cScent.deallocate()
        cColor.deallocate()
        cCounts.deallocate()
        cCosts.deallocate()
        cEnergyFunctions.deallocate()
      })
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

  internal func toC() -> (
    simulatorConfig: CNELFramework.SimulatorConfig, 
    deallocate: () -> Void
  ) {
    let (itemTypes, itemTypeDeallocators) = items
      .map { $0.toC(in: self) }
      .reduce(into: ([CNELFramework.ItemProperties](), [() -> Void]())) {
        $0.0.append($1.itemProperties)
        $0.1.append($1.deallocate)
      }
    let color = agentColor.scalars
    let cItemTypes = UnsafeMutablePointer<CNELFramework.ItemProperties>.allocate(capacity: itemTypes.count)
    let cColor = UnsafeMutablePointer<Float>.allocate(capacity: color.count)
    cItemTypes.initialize(from: itemTypes, count: itemTypes.count)
    cColor.initialize(from: color, count: color.count)

    return (
      simulatorConfig: CNELFramework.SimulatorConfig(
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
        itemTypes: cItemTypes,
        numItemTypes: UInt32(items.count),
        agentColor: cColor,
        movementConflictPolicy: moveConflictPolicy.toCMoveConflictPolicy(), 
        scentDecay: scentDecay, 
        scentDiffusion: scentDiffusion, 
        removedItemLifetime: removedItemLifetime),
      deallocate: { () in 
        cItemTypes.deallocate()
        cColor.deallocate()
        for deallocate in itemTypeDeallocators {
          deallocate()
        }
      })
  }
}
