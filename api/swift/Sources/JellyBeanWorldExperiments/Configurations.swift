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

fileprivate let rebuttal: Bool = true

public let banana = Item(
  name: "Banana",
  scent: rebuttal ? ShapedArray([0.0, 1.0, 0.0]) : ShapedArray([1.92, 1.76, 0.40]),
  color: rebuttal ? ShapedArray([0.0, 1.0, 0.0]) : ShapedArray([0.96, 0.88, 0.20]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  visualOcclusion: 0.0,
  energyFunctions: EnergyFunctions(
    intensityFn: rebuttal ? .constant(-5.3) : .constant(1.5),
    interactionFns: rebuttal ?
      [
        .piecewiseBox(itemId: 0,  10.0, 200.0,  0.0,    -6.0),
        .piecewiseBox(itemId: 1, 200.0,   0.0, -6.0,    -6.0),
        .piecewiseBox(itemId: 2,  10.0, 200.0,  2.0,  -100.0)] :
      [
        .piecewiseBox(itemId: 0, 10.0,  100.0,  0.0,    -6.0),
        .piecewiseBox(itemId: 2, 10.0,  100.0,  2.0,  -100.0),
        .piecewiseBox(itemId: 4, 50.0, 100.0, -100.0, -100.0)]))
public let onion = Item(
  name: "Onion",
  scent: rebuttal ? ShapedArray([1.0, 0.0, 0.0]) : ShapedArray([0.68, 0.01, 0.99]),
  color: rebuttal ? ShapedArray([1.0, 0.0, 0.0]) : ShapedArray([0.68, 0.01, 0.99]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  visualOcclusion: 0.0,
  energyFunctions: EnergyFunctions(
    intensityFn: rebuttal ? .constant(-5.0) : .constant(-3.0),
    interactionFns: rebuttal ?
      [
        .piecewiseBox(itemId: 0, 200.0, 0.0,   -6.0,   -6.0),
        .piecewiseBox(itemId: 2, 200.0, 0.0, -100.0, -100.0)] :
      []))
public let jellyBean = Item(
  name: "JellyBean",
  scent: rebuttal ? ShapedArray([0.0, 0.0, 1.0]) : ShapedArray([1.64, 0.54, 0.40]),
  color: rebuttal ? ShapedArray([0.0, 0.0, 1.0]) : ShapedArray([0.82, 0.27, 0.20]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  visualOcclusion: 0.0,
  energyFunctions: EnergyFunctions(
    intensityFn: rebuttal ? .constant(-5.3) : .constant(1.5),
    interactionFns: rebuttal ?
      [
        .piecewiseBox(itemId: 0,  10.0, 200.0,    2.0, -100.0),
        .piecewiseBox(itemId: 1, 200.0,   0.0, -100.0, -100.0),
        .piecewiseBox(itemId: 2,  10.0, 200.0,   0.0,    -6.0)] :
      [
        .piecewiseBox(itemId: 0, 10.0,  100.0,  2.0,  -100.0),
        .piecewiseBox(itemId: 2, 10.0,  100.0,  0.0,    -6.0),
        .piecewiseBox(itemId: 4, 50.0, 100.0, -100.0, -100.0)]))
public let truffle = Item(
  name: "Truffle",
  scent: ShapedArray([8.40, 4.80, 2.60]),
  color: ShapedArray([0.42, 0.24, 0.13]),
  requiredItemCounts: [:],
  requiredItemCosts: [:],
  blocksMovement: false,
  visualOcclusion: 0.0,
  energyFunctions: EnergyFunctions(
    intensityFn: .constant(0.0),
    interactionFns: [
      .piecewiseBox(itemId: 4,  4.0,  200.0,  2.0,  0.0),
      .piecewiseBox(itemId: 5, 30.0, 1000.0, -0.3, -1.0)]))

public func simulatorConfiguration(
  randomSeed: UInt32,
  agentFieldOfView: Int,
  enableVisualOcclusion: Bool
) -> Simulator.Configuration {
  let wall = Item(
    name: "Wall",
    scent: rebuttal ? ShapedArray([0.0, 0.0, 0.0]) : ShapedArray([0.00, 0.00, 0.00]),
    color: rebuttal ? ShapedArray([0.5, 0.5, 0.5]) : ShapedArray([0.20, 0.47, 0.67]),
    requiredItemCounts: [3: 1], // Make walls impossible to collect.
    requiredItemCosts: [:],
    blocksMovement: true,
    visualOcclusion: enableVisualOcclusion ? 1.0 : 0.0,
    energyFunctions: EnergyFunctions(
      intensityFn: rebuttal ? .constant(0.0) : .constant(-12.0),
      interactionFns: rebuttal ?
        [.cross(itemId: 3, 10.0, 15.0, 20.0, -200.0, -20.0, 1.0)] :
        [.cross(itemId: 3, 20.0, 40.0, 8.0, -1000.0, -1000.0, -1.0)]))
  let tree = Item(
    name: "Tree",
    scent: ShapedArray([0.00, 0.47, 0.06]),
    color: ShapedArray([0.00, 0.47, 0.06]),
    requiredItemCounts: [4: 1], // Make trees impossible to collect.
    requiredItemCosts: [:],
    blocksMovement: false,
    visualOcclusion: enableVisualOcclusion ? 0.1 : 0.0,
    energyFunctions: EnergyFunctions(
      intensityFn: .constant(2.0),
      interactionFns: [
        .piecewiseBox(itemId: 4, 100.0, 500.0, 0.0, -0.1)]))

  return Simulator.Configuration(
    randomSeed: randomSeed,
    maxStepsPerMove: 1,
    scentDimensionality: 3,
    colorDimensionality: 3,
    visionRange: rebuttal ? 5 : 8,
    movePolicies: [.up: .allowed],
    turnPolicies: [.left: .allowed, .right: .allowed],
    noOpAllowed: false,
    patchSize: rebuttal ? 32 : 64,
    mcmcIterations: rebuttal ? 4000 : 10000,
    items: rebuttal ?
      [banana, onion, jellyBean, wall] :
      [banana, onion, jellyBean, wall, tree, truffle],
    agentColor: [0.0, 0.0, 0.0],
    agentFieldOfView: Float(agentFieldOfView) * .pi / 180.0,
    moveConflictPolicy: .firstComeFirstServe,
    scentDecay: 0.4,
    scentDiffusion: 0.14,
    removedItemLifetime: 2000)
}
