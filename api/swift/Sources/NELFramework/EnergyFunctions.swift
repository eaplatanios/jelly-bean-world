import CNELFramework

public struct EnergyFunctions : Hashable {
  let intensityFn: IntensityFunction
  let interactionFns: [InteractionFunction]

  public init(
    intensityFn: IntensityFunction, 
    interactionFns: [InteractionFunction]
  ) {
    self.intensityFn = intensityFn
    self.interactionFns = interactionFns
  }

  internal func toC() -> CNELFramework.EnergyFunctions {
    return CNELFramework.EnergyFunctions(
      intensityFn: intensityFn.toC(),
      interactionFns: interactionFns.map { $0.toC() },
      numInteractionFns: UInt32(interactionFns.count)
    )
  }
}

public struct IntensityFunction : Hashable {
  let id: UInt32
  let arguments: [Float]

  init(id: UInt32, arguments: [Float] = []) {
    self.id = id
    self.arguments = arguments
  }

  internal func toC() -> CNELFramework.IntensityFunction {
    return CNELFramework.IntensityFunction(
      id: self.id,
      args: self.arguments,
      numArgs: UInt32(self.arguments.count))
  }
}

public struct InteractionFunction : Hashable {
  let id: UInt32
  let itemId: UInt32
  let arguments: [Float]

  init(id: UInt32, itemId: UInt32, arguments: [Float] = []) {
    self.id = id
    self.itemId = itemId
    self.arguments = arguments
  }

  internal func toC() -> CNELFramework.InteractionFunction {
    return CNELFramework.InteractionFunction(
      id: self.id,
      itemId: self.itemId,
      args: self.arguments,
      numArgs: UInt32(self.arguments.count))
  }
}

public func constantIntensity(_ value: Float) -> IntensityFunction {
  return IntensityFunction(id: 1, arguments: [value])
}

public func piecewiseBoxInteraction(
  itemId: UInt32, 
  _ firstCutoff: Float, 
  _ secondCutoff: Float,
  _ firstValue: Float,
  _ secondValue: Float
) -> InteractionFunction {
  return InteractionFunction(
    id: 1,
    itemId: itemId,
    arguments: [
      firstCutoff, secondCutoff, 
      firstValue, secondValue])
}

public func crossInteraction(
  itemId: UInt32,
  _ nearCutoff: Float,
  _ farCutoff: Float,
  _ nearAxisAlignedValue: Float,
  _ nearMisalignedValue: Float,
  _ farAxisAlignedValue: Float,
  _ farMisalignedValue: Float
) -> InteractionFunction {
  return InteractionFunction(
    id: 2,
    itemId: itemId,
    arguments: [
      nearCutoff, farCutoff,
      nearAxisAlignedValue, nearMisalignedValue, 
      farAxisAlignedValue, farMisalignedValue])
}
