// Copyright 2019, The Jelly Bean World Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

import Foundation
import JellyBeanWorld
import Logging
import Python
import TensorFlow

// Create a dummy agent delegate.
fileprivate struct DummyAgent: JellyBeanWorld.Agent {
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

public func runDummyExperiment() {
  PythonLibrary.useVersion(3, 7)
  let mpl = Python.import("matplotlib")
  mpl.use("TkAgg")

  let simulator = try! Simulator(using: simulatorConfiguration)

  // Create the agents.
  logger.info("Creating agents.")
  let numAgents = 1
  var agents = [JellyBeanWorld.Agent]()
  while agents.count < numAgents {
    let agent = DummyAgent()
    try! simulator.add(agent: agent)
    agents.append(agent)
  }

  logger.info("Starting simulation.")
  let painter = MapVisualizer(
    for: simulator, 
    bottomLeft: Position(x: -70, y: -70), 
    topRight: Position(x: 70, y: 70),
    agentPerspective: false)
  var startTime = Date().timeIntervalSince1970
  var elapsed = Float(0.0)
  let simulationStartTime = simulator.time
  for _ in 0..<100000000 {
    try! simulator.step()
    let interval = Date().timeIntervalSince1970 - startTime
    if interval > 1.0 {
      elapsed += Float(interval)
      let speed = Float(simulator.time - simulationStartTime) / elapsed
      logger.info("\(speed) simulation steps per second.")
      startTime = Date().timeIntervalSince1970
    }
    try! painter.draw()
  }
}
