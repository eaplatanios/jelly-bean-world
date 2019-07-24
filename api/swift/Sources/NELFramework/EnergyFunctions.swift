import CNELFramework

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

@inlinable
public func constantIntensity(_ value: Float) -> IntensityFunction {
  IntensityFunction(id: 1, arguments: [value])
}

@inlinable
public func piecewiseBoxInteraction(
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
public func crossInteraction(
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
