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

/// A 2-tuple-like struct that conforms to TensorGroup that represents a tuple of 2 types conforming
/// to `TensorGroup`.
@frozen
public struct Zip2TensorGroup<T: TensorGroup, U: TensorGroup>: TensorGroup {
  public var first: T
  public var second: U

  public init(_ first: T, _ second: U) {
    self.first = first
    self.second = second
  }

  public static var _typeList: [TensorDataType] { return T._typeList + U._typeList }

  public init(_owning tensorHandles: UnsafePointer<CTensorHandle>?) {
    first = .init(_owning: tensorHandles)
    second = .init(_owning: tensorHandles?.advanced(by: Int(T._tensorHandleCount)))
  }

  public func _unpackTensorHandles(into address: UnsafeMutablePointer<CTensorHandle>?) {
    var ptr = address
    first._unpackTensorHandles(into: ptr)
    ptr = ptr!.advanced(by: Int(first._tensorHandleCount))
    second._unpackTensorHandles(into: ptr)
  }

  public var _tensorHandles: [_AnyTensorHandle] {
    first._tensorHandles + second._tensorHandles
  }

  public init<C: RandomAccessCollection>(
    _handles: C
  ) where C.Element: _AnyTensorHandle {
    let firstStart = _handles.startIndex
    let firstEnd = _handles.index(
      firstStart, offsetBy: Int(T._tensorHandleCount))
    self.first = T.init(_handles: _handles[firstStart..<firstEnd])
    self.second = U.init(_handles: _handles[firstEnd..<_handles.endIndex])
  }
}
