// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

import _Differentiation

extension Tensor {
  /// Returns true if the physical scalar type is reduced precision.
  ///
  /// Currently, reduced precision physical scalar types include only `BFloat16`.
  public var isReducedPrecision: Bool {
    fatalError()
//    return device.backend == .XLA && xlaTensor.physicalScalarType == XLATensorScalarType_BFloat16
  }

  /// Promotes a scalar to a tensor with the same device and precision as the given tensor.
  // TODO (SR-12968): Mark `tensor` with `@noDerivative` and remove custom vjp below.
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(_ value: Scalar, deviceAndPrecisionLike tensor: Tensor) {
    fatalError()
//    let device = tensor.device
//    let tmp = Tensor(value, on: device)
//    self = tensor.isReducedPrecision ? tmp.toReducedPrecision : tmp
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  // TODO (SR-12968): Remove when `tensor` can be marked `@noDerivative` in `init`.
  // This currently places the pullback results of `tensor` on the correct device.
  @usableFromInline
  @derivative(of: init(_:deviceAndPrecisionLike:))
  static func vjpInitDeviceAndPrecisionLike(
    _ value: Scalar,
    deviceAndPrecisionLike tensor: Tensor
  ) -> (value: Tensor, pullback: (Tensor) -> (Scalar, Tensor)) {
    fatalError()
//    // Get device and precision in forward pass to avoid capturing `tensor` in pullback.
//    let device = tensor.device
//    let useReducedPrecision = tensor.isReducedPrecision
//    let result = Tensor(value, on: device)
//    return (useReducedPrecision ? result.toReducedPrecision : result, {
//      let tmp = Tensor(0, on: device)
//      return ($0.scalarized(), useReducedPrecision ? tmp.toReducedPrecision : tmp)
//    })
  }
}

extension Tensor {//}: ReducedPrecisionConvertible, _ReducedPrecisionConvertible {
  /// Returns a copy of `self` converted to `BFloat16` physical scalar type.
  public var toReducedPrecision: Self {
    fatalError()
//    if isReducedPrecision {
//      fatalError("Must not already have reduced precision")
//    }
//    if Scalar.self != Float.self {
//      fatalError("Reduced precision is only supported for Float tensors")
//    }
//    return _Raw.physicalCast(self, destType: BFloat16.self)
  }

  /// Returns a copy of `self` converted to `Scalar` physical scalar type.
  public var toFullPrecision: Self {
    fatalError()
//    if !isReducedPrecision {
//      fatalError("Must have reduced precision")
//    }
//    if Scalar.self != Float.self {
//      fatalError("Reduced precision is only supported for Float tensors")
//    }
//    return _Raw.physicalCast(self, destType: Scalar.self)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @usableFromInline
  @derivative(of: toReducedPrecision)
  func _vjpToReducedPrecision() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
//    (toReducedPrecision, { $0.toFullPrecision })
  }

  @usableFromInline
  @derivative(of: toFullPrecision)
  func _vjpToFullPrecision() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
//    (toFullPrecision, { $0.toReducedPrecision })
  }
}
