import Foundation
import NELFramework
import Python
import TensorFlow

PythonLibrary.useVersion(3, 7)
let mpl = Python.import("matplotlib")
mpl.use("TkAgg")

// Create a dummy agent delegate.
final class DummyAgentDelegate : AgentDelegate {
  private var counter: UInt64 = 0

  func act(_ agent: Agent) {
    self.counter += 1
    switch self.counter % 20 {
      case 0:  agent.turn(towards: .left)
      case 5:  agent.turn(towards: .left)
      case 10: agent.turn(towards: .right)
      case 15: agent.turn(towards: .right)
      default: agent.move(towards: .up, by: 1)
    }
  }

  func save(_ agent: Agent, to file: URL) throws {
    try String(counter).write(
      to: file, 
      atomically: true, 
      encoding: .utf8)
  }

  func load(_ agent: Agent, from file: URL) throws {
    self.counter = try UInt64(String(
      contentsOf: file, 
      encoding: .utf8))!
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
    intensityFn: constantIntensity(-5.3),
    interactionFns: [
      piecewiseBoxInteraction(itemId: 0,  10.0, 200.0,  0.0,  -6.0),
      piecewiseBoxInteraction(itemId: 1, 200.0,   0.0, -6.0,  -6.0),
      piecewiseBoxInteraction(itemId: 2,  10.0, 200.0, 2.0, -100.0)]))

let onion = Item(
  name: "onion",
  scent: ShapedArray([1.0, 0.0, 0.0]),
  color: ShapedArray([1.0, 0.0, 0.0]),
  requiredItemCounts: [1: 1], // Make onions impossible to collect.
  requiredItemCosts: [:],
  blocksMovement: false,
  energyFunctions: EnergyFunctions(
    intensityFn: constantIntensity(-5.0),
    interactionFns: [
      piecewiseBoxInteraction(itemId: 0, 200.0, 0.0,   -6.0,   -6.0),
      piecewiseBoxInteraction(itemId: 2, 200.0, 0.0, -100.0, -100.0)]))

let jellyBean = Item(
  name: "jellyBean",
  scent: ShapedArray([0.0, 0.0, 1.0]),
  color: ShapedArray([0.0, 0.0, 1.0]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  energyFunctions: EnergyFunctions(
    intensityFn: constantIntensity(-5.3),
    interactionFns: [
      piecewiseBoxInteraction(itemId: 0,  10.0, 200.0,    2.0, -100.0),
      piecewiseBoxInteraction(itemId: 1, 200.0,   0.0, -100.0, -100.0),
      piecewiseBoxInteraction(itemId: 2,  10.0, 200.0,  0.0,   -6.0)]))

let wall = Item(
  name: "wall",
  scent: ShapedArray([0.0, 0.0, 0.0]),
  color: ShapedArray([0.5, 0.5, 0.5]),
  requiredItemCounts: [3: 1], // Make walls impossible to collect.
  requiredItemCosts: [:],
  blocksMovement: true,
  energyFunctions: EnergyFunctions(
    intensityFn: constantIntensity(0.0),
    interactionFns: [
      crossInteraction(itemId: 3, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0)]))

// Create the simulator.

let config = SimulatorConfig(
  randomSeed: 1234567890,
  maxStepsPerMove: 1,
  scentDimSize: 3,
  colorDimSize: 3,
  visionRange: 5,
  allowedMoves: [.up],
  allowedTurns: [.left, .right],
  patchSize: 32,
  gibbsIterations: 10,
  items: [banana, onion, jellyBean, wall],
  agentColor: [0.0, 0.0, 1.0],
  moveConflictPolicy: .firstComeFirstServe,
  scentDecay: 0.4,
  scentDiffusion: 0.14,
  removedItemLifetime: 2000)
let simulator = Simulator(using: config)

// Create the agents.
print("Creating agents.")
let numAgents = 1
var agents = [Agent]()
while agents.count < numAgents {
  agents.append(Agent(
    in: simulator, 
    with: DummyAgentDelegate()))
}

print("Starting simulation.")
let painter = MapVisualizer(
  for: simulator, 
  bottomLeft: Position(x: -70, y: -70), 
  topRight: Position(x: 70, y: 70))
var startTime = CFAbsoluteTimeGetCurrent()
var elapsed = Float(0.0)
var simulationStartTime = simulator.time
for _ in 0..<100000000 {
  simulator.step()
  let interval = CFAbsoluteTimeGetCurrent() - startTime
  if interval > 1.0 {
    elapsed += Float(interval)
    let speed = Float(simulator.time - simulationStartTime) / elapsed
    print("\(speed) simulation steps per second.")
    startTime = CFAbsoluteTimeGetCurrent()
  }
  painter.draw()
}
