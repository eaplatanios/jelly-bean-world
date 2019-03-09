@inlinable
internal func scentToArray(
  for config: SimulatorConfig,
  _ buffer: UnsafeMutablePointer<Float>
) -> (shape: [Int], values: [Float]) {
  let scentShape = [Int(config.scentDimSize)]
  let scentBuffer = UnsafeBufferPointer(start: buffer, count: scentShape[0])
  return (shape: scentShape, values: Array(scentBuffer))
}

@inlinable
internal func visionToArray(
  for config: SimulatorConfig,
  _ buffer: UnsafeMutablePointer<Float>
) -> (shape: [Int], values: [Float]) {
  let visionShape = [
    2 * Int(config.visionRange) + 1, 
    2 * Int(config.visionRange) + 1, 
    Int(config.colorDimSize)]
  let visionSize = Int(
    (2 * config.visionRange + 1) * 
    (2 * config.visionRange + 1) * 
    config.colorDimSize)
  let visionBuffer = UnsafeBufferPointer(start: buffer, count: visionSize)
  return (shape: visionShape, values: Array(visionBuffer))
}

@inlinable
internal func itemCountsToArray(
  for config: SimulatorConfig,
  _ countsPointer: UnsafeMutablePointer<UInt32>
) -> [UInt32] {
  return Array(UnsafeBufferPointer(
    start: countsPointer, 
    count: Int(config.items.count)))
}
