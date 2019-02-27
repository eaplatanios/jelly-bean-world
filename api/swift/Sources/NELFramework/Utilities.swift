import TensorFlow

internal func visionToShapedArray(
  for config: SimulatorConfig,
  _ buffer: UnsafeMutablePointer<Float>
) -> ShapedArray<Float> {
  let visionShape = [
    2 * Int(config.visionRange) + 1, 
    2 * Int(config.visionRange) + 1, 
    Int(config.colorDimSize)]
  let visionBuffer = UnsafeBufferPointer(
      start: buffer,
      count: Int(
        4 * config.visionRange + 2 + 
        config.colorDimSize))
  return ShapedArray(
    shape: visionShape,
    scalars: visionBuffer)
}

internal func scentToShapedArray(
  for config: SimulatorConfig,
  _ buffer: UnsafeMutablePointer<Float>
) -> ShapedArray<Float> {
  let scentShape = [Int(config.scentDimSize)]
  let scentBuffer = UnsafeBufferPointer(
      start: buffer,
      count: Int(config.scentDimSize))
  return ShapedArray(
    shape: scentShape,
    scalars: scentBuffer)
}

internal func itemCountsToDictionary(
  for config: SimulatorConfig,
  _ countsPointer: UnsafeMutablePointer<UInt32>
) -> [Item: UInt32] {
  let counts = Array(UnsafeBufferPointer(
    start: countsPointer, 
    count: Int(config.items.count)))
  var dict = [Item: UInt32]()
  for (index, item) in config.items.enumerated() {
    dict[item] = counts[index]
  }
  return dict
}
