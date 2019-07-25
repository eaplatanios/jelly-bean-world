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

    @inlinable
    internal init(fromC value: SimulatorConfig) {
      self.randomSeed = value.randomSeed
      self.maxStepsPerMove = value.maxStepsPerMove
      self.scentDimSize = value.scentDimSize
      self.colorDimSize = value.colorDimSize
      self.visionRange = value.visionRange
      self.allowedMoves = [
        .up: ActionPolicy(fromC: value.allowedMoveDirections.0),
        .down: ActionPolicy(fromC: value.allowedMoveDirections.1),
        .left: ActionPolicy(fromC: value.allowedMoveDirections.2),
        .right: ActionPolicy(fromC: value.allowedMoveDirections.3)]
      self.allowedTurns = [
        .front: ActionPolicy(fromC: value.allowedRotations.0),
        .back: ActionPolicy(fromC: value.allowedRotations.1),
        .left: ActionPolicy(fromC: value.allowedRotations.2),
        .right: ActionPolicy(fromC: value.allowedRotations.3)]
      self.noOpAllowed = value.noOpAllowed
      self.patchSize = value.patchSize
      self.mcmcIterations = value.mcmcIterations
      let cItems = UnsafeBufferPointer(start: value.itemTypes!, count: Int(value.numItemTypes))
      self.items = cItems.map { Item(fromC: $0) }
      self.agentColor = ShapedArray(
        shape: [Int(colorDimSize)],
        scalars: UnsafeBufferPointer(
          start: value.agentColor!,
          count: Int(colorDimSize)))
      self.moveConflictPolicy = MoveConflictPolicy(fromC: value.movementConflictPolicy)
      self.scentDecay = value.scentDecay
      self.scentDiffusion = value.scentDiffusion
      self.removedItemLifetime = value.removedItemLifetime
    }

    @inlinable
    internal func toC() -> (configuration: SimulatorConfig, deallocate: () -> Void) {
      let (items, itemDeallocators) = self.items
        .map { $0.toC(in: self) }
        .reduce(into: ([ItemProperties](), [() -> Void]())) {
          $0.0.append($1.item)
          $0.1.append($1.deallocate)
        }
      let color = agentColor.scalars
      let cItems = UnsafeMutablePointer<ItemProperties>.allocate(
        capacity: items.count)
      let cColor = UnsafeMutablePointer<Float>.allocate(capacity: color.count)
      cItems.initialize(from: items, count: items.count)
      cColor.initialize(from: color, count: color.count)
      return (
        configuration: SimulatorConfig(
          randomSeed: randomSeed,
          maxStepsPerMove: maxStepsPerMove,
          scentDimSize: scentDimSize,
          colorDimSize: colorDimSize,
          visionRange: visionRange,
          allowedMoveDirections: (
            allowedMoves[.up]?.toC() ?? ActionPolicy.disallowed.toC(),
            allowedMoves[.down]?.toC() ?? ActionPolicy.disallowed.toC(),
            allowedMoves[.left]?.toC() ?? ActionPolicy.disallowed.toC(),
            allowedMoves[.right]?.toC() ?? ActionPolicy.disallowed.toC()),
          allowedRotations: (
            allowedTurns[.front]?.toC() ?? ActionPolicy.disallowed.toC(),
            allowedTurns[.back]?.toC() ?? ActionPolicy.disallowed.toC(),
            allowedTurns[.left]?.toC() ?? ActionPolicy.disallowed.toC(),
            allowedTurns[.right]?.toC() ?? ActionPolicy.disallowed.toC()),
          noOpAllowed: noOpAllowed,
          patchSize: patchSize,
          mcmcIterations: mcmcIterations,
          itemTypes: cItems,
          numItemTypes: UInt32(items.count),
          agentColor: cColor,
          movementConflictPolicy: moveConflictPolicy.toC(),
          scentDecay: scentDecay,
          scentDiffusion: scentDiffusion,
          removedItemLifetime: removedItemLifetime),
        deallocate: { () in
          cItems.deallocate()
          cColor.deallocate()
          for deallocate in itemDeallocators {
            deallocate()
          }
        })
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

  @inlinable
  internal init(fromC value: ItemProperties) {
    fatalError("Not implemented yet!")
    // self.name = value.name // TODO: !!!!
    // self.scent = value.scent // TODO: !!!!
    // self.color = value.color // TODO: !!!!
    // self.requiredItemCounts = value.requiredItemCounts // TODO: !!!!
    // self.requiredItemCosts = value.requiredItemCosts // TODO: !!!!
    // self.blocksMovement = value.blocksMovement
    // self.energyFunctions = EnergyFunctions(fromC: value.energyFunctions)
  }

  @inlinable
  internal func toC(in configuration: Simulator.Configuration) -> (
    item: CNELFramework.ItemProperties,
    deallocate: () -> Void
  ) {
    let scent = self.scent.scalars
    let color = self.color.scalars
    let counts = configuration.items.indices.map { requiredItemCounts[$0, default: 0] }
    let costs = configuration.items.indices.map { requiredItemCosts[$0, default: 0] }

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
      item: CNELFramework.ItemProperties(
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

public struct EnergyFunctions: Hashable {
  @usableFromInline internal let intensityFn: IntensityFunction
  @usableFromInline internal let interactionFns: [InteractionFunction]

  @inlinable
  public init(intensityFn: IntensityFunction, interactionFns: [InteractionFunction]) {
    self.intensityFn = intensityFn
    self.interactionFns = interactionFns
  }

  @inlinable
  internal func toC() -> (energyFunctions: CNELFramework.EnergyFunctions, deallocate: () -> Void) {
    let cIntensityFn = intensityFn.toC()
    let (interactionFns, interactionFnDeallocators) = self.interactionFns
      .map { $0.toC() }
      .reduce(into: ([CNELFramework.InteractionFunction](), [() -> Void]())) {
        $0.0.append($1.interactionFunction)
        $0.1.append($1.deallocate)
      }
    let cInteractionFns = UnsafeMutablePointer<CNELFramework.InteractionFunction>.allocate(
      capacity: interactionFns.count)
    cInteractionFns.initialize(from: interactionFns, count: interactionFns.count)
    return (
      energyFunctions: CNELFramework.EnergyFunctions(
        intensityFn: cIntensityFn.intensityFunction,
        interactionFns: cInteractionFns,
        numInteractionFns: UInt32(interactionFns.count)),
      deallocate: { () in 
        cIntensityFn.deallocate()
        cInteractionFns.deallocate()
        for deallocate in interactionFnDeallocators {
          deallocate()
        }
      })
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

  @inlinable
  internal func toC() -> (
    intensityFunction: CNELFramework.IntensityFunction,
    deallocate: () -> Void
  ) {
    let cArgs = UnsafeMutablePointer<Float>.allocate(capacity: arguments.count)
    cArgs.initialize(from: arguments, count: arguments.count)
    return (
      intensityFunction: CNELFramework.IntensityFunction(
        id: id,
        args: cArgs,
        numArgs: UInt32(arguments.count)),
      deallocate: { () in cArgs.deallocate() })
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

  @inlinable
  internal func toC() -> (
    interactionFunction: CNELFramework.InteractionFunction,
    deallocate: () -> Void
  ) {
    let cArgs = UnsafeMutablePointer<Float>.allocate(capacity: arguments.count)
    cArgs.initialize(from: arguments, count: arguments.count)
    return (
      interactionFunction: CNELFramework.InteractionFunction(
        id: id,
        itemId: itemId,
        args: cArgs,
        numArgs: UInt32(arguments.count)),
      deallocate: { () in cArgs.deallocate() })
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
