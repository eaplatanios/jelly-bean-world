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

public func runBeamSearch(forMapIn file: URL, stepCount: Int, beamWidth: Int = 100000) {
  let rows = try! String(contentsOf: file, encoding: .utf8)
    .components(separatedBy: .newlines)
    .map { $0.components(separatedBy: ",") }
  let bottomLeft = Position(x: Int(rows[0][0])!, y: Int(rows[0][1])!)
  let topRight = Position(x: Int(rows[1][0])!, y: Int(rows[1][1])!)
  var items = Set<Position>()
  var blockedCells = Set<Position>()
  for row in rows.dropFirst(2) {
    if row.count != 3 { continue }
    let itemID = Int(row[0])!
    let position = Position(x: Int(row[1])!, y: Int(row[2])!)
    if itemID == 2 {
      items.update(with: position)
    } else if itemID == 3 {
      blockedCells.update(with: position)
    }
  }
  let map = Map(
    bottomLeft: bottomLeft,
    topRight: topRight,
    items: items,
    blockedCells: blockedCells)
  let position = Position(x: (topRight.x + bottomLeft.x) / 2, y: (topRight.y + bottomLeft.y) / 2)
  let direction = .up
  var heap = Heap<State>(elements: [], priorityFunction: { $0.items.count < $1.items.count })
  heap.enqueue(State(position: position, direction: direction, items: []))
  for step in 0..<stepCount {
    var previousStates = heap.elements
    previousStates.shuffle()
    heap.clear()
    for previousState in previousStates {
      for action in Action.allCases {
        let position = previousState.position.transform(
          using: action,
          facing: previousState.direction)
        if position.x < map.bottomLeft.x || position.x > map.topRight.x { continue }
        if position.y < map.bottomLeft.y || position.y > map.topRight.y { continue }
        if map.blockedCells.contains(position) { continue }
        let direction = previousState.direction.transform(using: action)
        var items = previousState.items
        if map.items.contains(position) { items.update(with: position) }
        heap.enqueue(State(
          position: position,
          direction: direction,
          items: items))
      }
    }
    while heap.count > beamWidth {
      heap.dequeue()
    }
    let maxRewardRate = heap.elements.map { Float($0.items.count) / Float(step) }.max()!
    logger.info("Step \(step) Maximum Reward Rate: \(maxRewardRate) pts/step")
  }
}

fileprivate struct Position: Hashable {
  fileprivate let x: Int
  fileprivate let y: Int
}

fileprivate enum Direction: Int, Hashable {
  case up, down, left, right
}

fileprivate enum Action: CaseIterable {
  case moveForward, turnLeft, turnRight
}

extension Position {
  fileprivate func transform(using action: Action, facing direction: Direction) -> Position {
    switch (action, direction) {
    case (.moveForward, .up): return Position(x: x, y: y + 1)
    case (.moveForward, .down): return Position(x: x, y: y - 1)
    case (.moveForward, .left): return Position(x: x - 1, y: y)
    case (.moveForward, .right): return Position(x: x + 1, y: y)
    case _: return self
    }
  }
}

extension Direction {
  fileprivate func transform(using action: Action) -> Direction {
    switch (action, self) {
    case (.turnLeft, .up): return .left
    case (.turnLeft, .down): return .right
    case (.turnLeft, .left): return .down
    case (.turnLeft, .right): return .up
    case (.turnRight, .up): return .right
    case (.turnRight, .down): return .left
    case (.turnRight, .left): return .up
    case (.turnRight, .right): return .down
    case _: return self
    }
  }
}

fileprivate struct Map {
  fileprivate let bottomLeft: Position
  fileprivate let topRight: Position
  fileprivate let items: Set<Position>
  fileprivate let blockedCells: Set<Position>
}

fileprivate struct State: Hashable {
  fileprivate let position: Position
  fileprivate let direction: Direction
  fileprivate let items: Set<Position>

  fileprivate func hash(into hasher: inout Hasher) {
    hasher.combine(position)
    hasher.combine(direction)
  }
}

fileprivate struct Heap<Element> {
  var elements: [Element]
  let priorityFunction: (Element, Element) -> Bool

  var isEmpty: Bool { elements.isEmpty }
  var count: Int { elements.count }

  init(elements: [Element] = [], priorityFunction: @escaping (Element, Element) -> Bool) {
    self.elements = elements
    self.priorityFunction = priorityFunction
    buildHeap()
  }

  mutating func buildHeap() {
    for index in (0 ..< count / 2).reversed() {
      siftDown(elementAtIndex: index)
    }
  }

  mutating func clear() {
    elements = []
  }

  func peek() -> Element? { elements.first }
  func isRoot(_ index: Int) -> Bool { index == 0 }
  func leftChildIndex(of index: Int) -> Int { (2 * index) + 1 }
  func rightChildIndex(of index: Int) -> Int { (2 * index) + 2 }
  func parentIndex(of index: Int) -> Int { (index - 1) / 2 }

  func isHigherPriority(at firstIndex: Int, than secondIndex: Int) -> Bool {
    priorityFunction(elements[firstIndex], elements[secondIndex])
  }

  func highestPriorityIndex(of parentIndex: Int, and childIndex: Int) -> Int {
    guard childIndex < count && isHigherPriority(at: childIndex, than: parentIndex) else {
      return parentIndex
    }
    return childIndex
  }

  func highestPriorityIndex(for parent: Int) -> Int {
    highestPriorityIndex(of:
      highestPriorityIndex(of: parent, and: leftChildIndex(of: parent)),
      and: rightChildIndex(of: parent))
  }

  mutating func swapElement(at firstIndex: Int, with secondIndex: Int) {
    guard firstIndex != secondIndex else { return }
    elements.swapAt(firstIndex, secondIndex)
  }

  mutating func enqueue(_ element: Element) {
    elements.append(element)
    siftUp(elementAtIndex: count - 1)
  }

  @discardableResult
  mutating func dequeue() -> Element? {
    guard !isEmpty else { return nil }
    swapElement(at: 0, with: count - 1)
    let element = elements.removeLast()
    if !isEmpty { siftDown(elementAtIndex: 0) }
    return element
  }

  mutating func siftUp(elementAtIndex index: Int) {
    let parent = parentIndex(of: index)
    guard !isRoot(index), isHigherPriority(at: index, than: parent) else { return }
    swapElement(at: index, with: parent)
    siftUp(elementAtIndex: parent)
  }

  mutating func siftDown(elementAtIndex index: Int) {
    let childIndex = highestPriorityIndex(for: index)
    if index == childIndex { return }
    swapElement(at: index, with: childIndex)
    siftDown(elementAtIndex: childIndex)
  }
}
