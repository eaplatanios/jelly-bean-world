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

import JellyBeanWorld
import ReinforcementLearning
import TensorFlow

public let banana = Item(
  name: "Banana",
  scent: ShapedArray([0.0, 1.0, 0.0]),
  color: ShapedArray([0.0, 1.0, 0.0]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  visualOcclusion: 0.0,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(-5.3),
    interactionFns: [
      .piecewiseBox(itemId: 0,  10.0, 200.0,  0.0,  -6.0),
      .piecewiseBox(itemId: 1, 200.0,   0.0, -6.0,  -6.0),
      .piecewiseBox(itemId: 2,  10.0, 200.0, 2.0, -100.0)]))

public let onion = Item(
  name: "Onion",
  scent: ShapedArray([1.0, 0.0, 0.0]),
  color: ShapedArray([1.0, 0.0, 0.0]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  visualOcclusion: 0.0,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(-5.0),
    interactionFns: [
      .piecewiseBox(itemId: 0, 200.0, 0.0,   -6.0,   -6.0),
      .piecewiseBox(itemId: 2, 200.0, 0.0, -100.0, -100.0)]))

public let jellyBean = Item(
  name: "JellyBean",
  scent: ShapedArray([0.0, 0.0, 1.0]),
  color: ShapedArray([0.0, 0.0, 1.0]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  visualOcclusion: 0.0,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(-5.3),
    interactionFns: [
      .piecewiseBox(itemId: 0,  10.0, 200.0,    2.0, -100.0),
      .piecewiseBox(itemId: 1, 200.0,   0.0, -100.0, -100.0),
      .piecewiseBox(itemId: 2,  10.0, 200.0,  0.0,   -6.0)]))

public let wall = Item(
  name: "Wall",
  scent: ShapedArray([0.0, 0.0, 0.0]),
  color: ShapedArray([0.5, 0.5, 0.5]),
  requiredItemCounts: [3: 1], // Make walls impossible to collect.
  requiredItemCosts: [:],
  blocksMovement: true,
  visualOcclusion: 0.5,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(0.0),
    interactionFns: [
      .cross(itemId: 3, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0)]))

public func simulatorConfiguration(
  randomSeed: UInt32,
  agentFieldOfView: Float
) -> Simulator.Configuration {
  Simulator.Configuration(
    randomSeed: randomSeed,
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
    agentColor: [0.0, 0.0, 0.0],
    agentFieldOfView: agentFieldOfView,
    moveConflictPolicy: .firstComeFirstServe,
    scentDecay: 0.4,
    scentDiffusion: 0.14,
    removedItemLifetime: 2000)
}
