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

import TensorFlow

extension Array where Element == Float {
  internal var sum: Float { reduce(0, +) }

  internal var mean: Float { sum / Float(count) }

  internal var median: Float {
    let sortedArray = sorted()
    if count % 2 != 0 {
      return Float(sortedArray[count / 2])
    } else {
      return Float(sortedArray[count / 2] + sortedArray[count / 2 - 1]) / 2.0
    }
  }

  internal var standardDeviation: Element {
    let mean = self.mean
    let variance = map { ($0 - mean) * ($0 - mean) }
    return TensorFlow.sqrt(variance.mean)
  }

  internal func scan<T>(_ initial: T, _ f: (T, Element) -> T) -> [T] {
    reduce([initial], { (listSoFar: [T], next: Element) -> [T] in
      listSoFar + [f(listSoFar.last!, next)]
    })
  }
}

extension Sequence where Element: Collection, Element.Index == Int {
  public func transposed(prefixWithMaxLength max: Int = .max) -> [[Element.Element]] {
    var transposed: [[Element.Element]] = []
    let n = Swift.min(max, self.min { $0.count < $1.count }?.count ?? 0)
    transposed.reserveCapacity(n)
    for i in 0..<n { transposed.append(map{ $0[i] }) }
    return transposed
  }
}

@usableFromInline
internal struct Deque<Scalar: FloatingPoint> {
  @usableFromInline internal let size: Int
  @usableFromInline internal var buffer: [Scalar]
  @usableFromInline internal var index: Int
  @usableFromInline internal var full: Bool

  @inlinable
  init(size: Int) {
    self.size = size
    self.buffer = [Scalar](repeating: 0, count: size)
    self.index = 0
    self.full = false
  }

  @inlinable
  mutating func push(_ value: Scalar) {
    buffer[index] = value
    index += 1
    full = full || index == buffer.count
    index = index % buffer.count
  }

  @inlinable
  mutating func reset() {
    index = 0
    full = false
  }

  @inlinable
  func sum() -> Scalar {
    return full ? buffer.reduce(0, +) : buffer[0..<index].reduce(0, +)
  }

  @inlinable
  func mean() -> Scalar {
    let sum = full ? buffer.reduce(0, +) : buffer[0..<index].reduce(0, +)
    let count = full ? buffer.count : index
    return sum / Scalar(count)
  }
}
