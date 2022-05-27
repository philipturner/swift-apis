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

import _Differentiation
import CTensorFlow

infix operator .==: ComparisonPrecedence
infix operator .!=: ComparisonPrecedence

/// Special protocol for calling tensorflow operations that take heterogeneous arrays as input.
public protocol AnyTensor {
  var _rawTensorHandle: CTensorHandle { get }
  var _tensorFlowDataType: TensorDataType { get }
  var scalarType: TensorFlowScalar.Type { get }
}

@frozen
public struct Tensor<Scalar: TensorFlowScalar> {
  /// The underlying `TensorHandle`.
  /// - Note: `handle` is public to allow user defined ops, but should not normally be used.
  public let handle: TensorHandle<Scalar>

  /// An internal marker to identify scalar zero tensors, for use in optimizations.
  @usableFromInline
  internal var _isScalarZero = false


  @inlinable
  public init(handle: TensorHandle<Scalar>) {
    self.handle = handle
  }
}

extension Tensor: AnyTensor {
  public var _rawTensorHandle: CTensorHandle { return handle._cTensorHandle }
  public var _tensorFlowDataType: TensorDataType { return Scalar.tensorFlowDataType }
  public var scalarType: TensorFlowScalar.Type { return Scalar.self }
}

//===------------------------------------------------------------------------------------------===//
// Tensor Properties
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  /// The number of dimensions of the `Tensor`.
  public var rank: Int {
    @_semantics("autodiff.nonvarying")
    get { handle.rank }
  }

  /// The shape of the `Tensor`.
  public var shape: TensorShape {
    @_semantics("autodiff.nonvarying")
    get { handle.shape }
  }

  /// The number of scalars in the `Tensor`.
  @inlinable
  public var scalarCount: Int {
    @_semantics("autodiff.nonvarying")
    get { shape.contiguousSize }
  }

  /// The rank of the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  public var rankTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
//      return _Raw.rank(self)
    }
  }

  /// The dimensions of the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  public var shapeTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
//      return _Raw.shape(self)
    }
  }

  /// The number of scalars in the tensor, represented as a `Tensor<Int32>`.
  @inlinable
  public var scalarCountTensor: Tensor<Int32> {
    @_semantics("autodiff.nonvarying")
    get {
      fatalError()
//      return _Raw.size(self)
    }
  }
}

//===------------------------------------------------------------------------------------------===//
// Scalar Conversion
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  /// Returns `true` if `rank` is equal to 0 and `false` otherwise.
  @inlinable
  public var isScalar: Bool {
    return rank == 0
  }

  /// Returns the single scalar element if `rank` is equal to 0 and `nil`
  /// otherwise.
  @inlinable
  public var scalar: Scalar? {
    fatalError()
//    isScalar ? scalars[0] : nil
  }

  /// Reshape to scalar.
  /// - Precondition: The tensor has exactly one scalar.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func scalarized() -> Scalar {
    fatalError()
//    precondition(
//      shape.contiguousSize == 1,
//      "This tensor must have exactly one scalar but contains \(shape.contiguousSize).")
//    return scalars[0]
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: scalarized)
  func _vjpScalarized() -> (value: Scalar, pullback: (Scalar) -> Tensor) {
    fatalError()
//    let device = self.device
//    return (scalarized(), { v in Tensor(v, on: device) })
  }
}

extension TensorFlowScalar {
  @inlinable
  public init?(_ tensor: Tensor<Self>) {
    guard let scalar = tensor.scalar else {
      return nil
    }
    self = scalar
  }
}


extension Tensor {
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public init(shape: TensorShape, scalars: [Scalar], on device: Device = .default) {
    fatalError()
//    precondition(

  }

  /// Creates a tensor with the specified shape and contiguous scalars in row-major order.
  ///
  /// - Parameters:
  ///   - shape: The shape of the tensor.
  ///   - scalars: The scalar contents of the tensor.
  /// - Precondition: The product of the dimensions of the shape must equal the number of scalars.
  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>,
    on device: Device = .default
  ) {
    fatalError()

  }


  @inlinable
  public init(
    shape: TensorShape,
    scalars: [Scalar],
    toReducedPrecision: Bool,
    directlyOn device: Device
  ) {
    fatalError()

  }

  public init(
    shape: TensorShape,
    scalars: UnsafeBufferPointer<Scalar>,
    toReducedPrecision: Bool,
    directlyOn device: Device
  ) {
    fatalError()

  }

}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
//  @derivative(of: init(_:on:))
  static func _vjpInit(_ scalars: [Scalar], on device: Device = .default) -> (
    value: Tensor, pullback: (Tensor) -> Array<Scalar>.TangentVector
  ) {
    fatalError()
//    (
//      value: Tensor(scalars, on: device),
//      pullback: { v in
//        Array<Scalar>.TangentVector(v.scalars)
//      }
//    )
  }

  @inlinable
  @derivative(of: init(shape:scalars:on:))
  static func _vjpInit(
    shape: TensorShape, scalars: [Scalar], on device: Device = .default
  ) -> (value: Tensor, pullback: (Tensor) -> Array<Scalar>.TangentVector) {
    fatalError()
//    (
//      value: Tensor(shape: shape, scalars: scalars, on: device),
//      pullback: { v in
//        Array<Scalar>.TangentVector(v.scalars)
//      }
//    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Equatable
//===------------------------------------------------------------------------------------------===//

extension Tensor: Equatable where Scalar: Equatable {
  @inlinable
  public static func == (lhs: Tensor, rhs: Tensor) -> Bool {
    fatalError()
//    guard lhs.shape == rhs.shape else {
//      return false
//    }
//    return (lhs .== rhs).all()
  }

  @inlinable
  public static func != (lhs: Tensor, rhs: Tensor) -> Bool {
    fatalError()
//    guard lhs.shape == rhs.shape else {
//      return true
//    }
//    return (lhs .!= rhs).any()
  }
}

//===------------------------------------------------------------------------------------------===//
// Additive Group
//===------------------------------------------------------------------------------------------===//

extension Tensor: AdditiveArithmetic where Scalar: Numeric {
  /// The scalar zero tensor.
  public static var zero: Tensor {
    fatalError()
//    var zero = Tensor(0, on: _DeviceThreadLocalState.local.currentDevice)
//    if _DeviceThreadLocalState.local.isReducedPrecision {
//      zero = zero.toReducedPrecision
//    }
//    zero._isScalarZero = true
//    return zero
  }

  /// Adds two tensors and produces their sum.
  /// - Note: `+` supports broadcasting.
  @inlinable
//  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    fatalError()
//    if lhs._isScalarZero {
//      return rhs
//    } else if rhs._isScalarZero {
//      return lhs
//    }
//    return _Raw.addV2(lhs, rhs)
  }

  /// Subtracts one tensor from another and produces their difference.
  /// - Note: `-` supports broadcasting.
  @inlinable
//  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    fatalError()
//    if rhs._isScalarZero {
//      return lhs
//    }
//    return _Raw.sub(lhs, rhs)
  }
}


//===------------------------------------------------------------------------------------------===//
// Differentiable
//===------------------------------------------------------------------------------------------===//

extension Tensor: Differentiable /*& EuclideanDifferentiable*/ where Scalar: TensorFlowFloatingPoint {
  public typealias TangentVector = Tensor

}
