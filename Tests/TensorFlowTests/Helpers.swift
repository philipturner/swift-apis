// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import XCTest

@testable import TensorFlow

internal func assertEqual<T: TensorFlowFloatingPoint>(
  _ x: [T], _ y: [T], accuracy: T, _ message: String = "",
  file: StaticString = #file, line: UInt = #line
) {
  for (x, y) in zip(x, y) {
    if x.isNaN || y.isNaN {
      XCTAssertTrue(
        x.isNaN && y.isNaN,
        "\(x) is not equal to \(y) - \(message)",
        file: file, line: line)
      continue
    }
    XCTAssertEqual(x, y, accuracy: accuracy, message, file: file, line: line)
  }
}

internal func assertEqual<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>, _ y: Tensor<T>, accuracy: T, _ message: String = "",
  file: StaticString = #file, line: UInt = #line
) {
  assertEqual(x.scalars, y.scalars, accuracy: accuracy, message, file: file, line: line)
}

func withTensorLeakChecking(
  file: StaticString = #file,
  line: UInt = #line,
  _ body: () throws -> Void
) rethrows {
  let initialTensorCount = Context.local.globalTensorCount
  try body()
  let tensorCountDifference = Context.local.globalTensorCount - initialTensorCount
  XCTAssertGreaterThanOrEqual(tensorCountDifference, 0, "Negative tensor count?")
  XCTAssertEqual(tensorCountDifference, 0, "Memory leaks found", file: file, line: line)
}

extension Float: PointwiseMultiplicative {
  public var reciprocal: Float { 1 / self }
  public static func .* (lhs: Float, rhs: Float) -> Float { lhs * rhs }
}

struct Multiply: Layer {
  var coefficient: Float

  @differentiable(reverse)
  func callAsFunction(_ input: Float) -> Float {
    return coefficient * input
  }
}

func factorial(_ n: Int) -> Int {
  var result: Int = 1
  for i in 2...n {
    result *= i
  }
  return result
}

struct Zip2TensorGroup<T: TensorGroup, U: TensorGroup>: TensorGroup {
  var first: T
  var second: U

  init(_ first: T, _ second: U) {
    self.first = first
    self.second = second
  }

  static var _typeList: [TensorDataType] { return T._typeList + U._typeList }

  init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    first = .init(_owning: tensorHandles)
    second = .init(_owning: tensorHandles?.advanced(by: Int(T._tensorHandleCount)))
  }

  func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    var ptr = address
    first._unpackTensorHandles(into: ptr)
    ptr = ptr!.advanced(by: Int(first._tensorHandleCount))
    second._unpackTensorHandles(into: ptr)
  }

  var _tensorHandles: [_AnyTensorHandle] {
    first._tensorHandles + second._tensorHandles
  }

  init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    let firstStart = _handles.startIndex
    let firstEnd = _handles.index(
      firstStart, offsetBy: Int(T._tensorHandleCount))
    self.first = T.init(_handles: _handles[firstStart..<firstEnd])
    self.second = U.init(_handles: _handles[firstEnd..<_handles.endIndex])
  }
}
