import Foundation
import JellyBeanWorld
import Python
import TensorFlow

PythonLibrary.useVersion(3, 7)
let mpl = Python.import("matplotlib")
mpl.use("TkAgg")

// Create a dummy agent delegate.
struct DummyAgent: Agent {
  @usableFromInline internal var counter: UInt64 = 0

  @inlinable
  public mutating func act(using state: AgentState) -> Action {
    counter += 1
    switch counter % 20 {
      case 0:  return .turn(direction: .left)
      case 5:  return .turn(direction: .left)
      case 10: return .turn(direction: .right)
      case 15: return .turn(direction: .right)
      default: return .move(direction: .up)
    }
  }

  @inlinable
  public func save(to file: URL) throws {
    try String(counter).write(to: file, atomically: true, encoding: .utf8)
  }

  @inlinable
  public mutating func load(from file: URL) throws {
    counter = try UInt64(String(contentsOf: file, encoding: .utf8))!
  }
}

// Create all the items that exist in the world.

let banana = Item(
  name: "banana",
  scent: ShapedArray([0.0, 1.0, 0.0]),
  color: ShapedArray([0.0, 1.0, 0.0]),
  requiredItemCounts: [0: 1], // Make bananas impossible to collect.
  requiredItemCosts: [:],
  blocksMovement: false,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(-5.3),
    interactionFns: [
      .piecewiseBox(itemId: 0,  10.0, 200.0,  0.0,  -6.0),
      .piecewiseBox(itemId: 1, 200.0,   0.0, -6.0,  -6.0),
      .piecewiseBox(itemId: 2,  10.0, 200.0, 2.0, -100.0)]))

let onion = Item(
  name: "onion",
  scent: ShapedArray([1.0, 0.0, 0.0]),
  color: ShapedArray([1.0, 0.0, 0.0]),
  requiredItemCounts: [1: 1], // Make onions impossible to collect.
  requiredItemCosts: [:],
  blocksMovement: false,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(-5.0),
    interactionFns: [
      .piecewiseBox(itemId: 0, 200.0, 0.0,   -6.0,   -6.0),
      .piecewiseBox(itemId: 2, 200.0, 0.0, -100.0, -100.0)]))

let jellyBean = Item(
  name: "jellyBean",
  scent: ShapedArray([0.0, 0.0, 1.0]),
  color: ShapedArray([0.0, 0.0, 1.0]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(-5.3),
    interactionFns: [
      .piecewiseBox(itemId: 0,  10.0, 200.0,    2.0, -100.0),
      .piecewiseBox(itemId: 1, 200.0,   0.0, -100.0, -100.0),
      .piecewiseBox(itemId: 2,  10.0, 200.0,  0.0,   -6.0)]))

let wall = Item(
  name: "wall",
  scent: ShapedArray([0.0, 0.0, 0.0]),
  color: ShapedArray([0.5, 0.5, 0.5]),
  requiredItemCounts: [3: 1], // Make walls impossible to collect.
  requiredItemCosts: [:],
  blocksMovement: true,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(0.0),
    interactionFns: [.cross(itemId: 3, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0)]))

// Create the simulator.

let configuration = Simulator.Configuration(
  randomSeed: 1234567890,
  maxStepsPerMove: 1,
  scentDimensionality: 3,
  colorDimensionality: 3,
  visionRange: 5,
  movePolicies: [.up: .allowed],
  turnPolicies: [.left: .allowed, .right: .allowed],
  noOpAllowed: false,
  patchSize: 32,
  mcmcIterations: 4000,
  items: [banana, onion, jellyBean, wall],
  agentColor: [0.0, 0.0, 1.0],
  moveConflictPolicy: .firstComeFirstServe,
  scentDecay: 0.4,
  scentDiffusion: 0.14,
  removedItemLifetime: 2000)
let simulator = Simulator(using: configuration)

// Create the agents.
print("Creating agents.")
let numAgents = 1
var agents = [Agent]()
while agents.count < numAgents {
  let agent = DummyAgent()
  simulator.add(agent: agent)
  agents.append(agent)
}

print("Starting simulation.")
let painter = MapVisualizer(
  for: simulator, 
  bottomLeft: Position(x: -70, y: -70), 
  topRight: Position(x: 70, y: 70))
var startTime = Date().timeIntervalSince1970
var elapsed = Float(0.0)
var simulationStartTime = simulator.time
for _ in 0..<100000000 {
  simulator.step()
  let interval = Date().timeIntervalSince1970 - startTime
  if interval > 1.0 {
    elapsed += Float(interval)
    let speed = Float(simulator.time - simulationStartTime) / elapsed
    print("\(speed) simulation steps per second.")
    startTime = Date().timeIntervalSince1970
  }
  painter.draw()
}
