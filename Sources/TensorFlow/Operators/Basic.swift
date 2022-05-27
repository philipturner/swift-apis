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

infix operator .!=: ComparisonPrecedence

/// Returns a tensor with the same shape and scalars as the specified tensor.
@inlinable
@differentiable(reverse where Scalar: TensorFlowFloatingPoint)
public func identity<Scalar>(_ x: Tensor<Scalar>) -> Tensor<Scalar> {
  x
}

//===------------------------------------------------------------------------------------------===//
// Shape Transformations
//===------------------------------------------------------------------------------------------===//

extension TensorFlowScalar {
  /// Convert to a tensor with the specified rank, with all dimensions equal to `1`.
  @inlinable
  public func makeTensor(rank: Int, on device: Device = .default) -> Tensor<Self> {
    return Tensor(repeating: self, shape: TensorShape(rank), on: device)
  }
}

extension Tensor {
  /// Unpacks the given dimension of a rank-`R` tensor into multiple rank-`(R-1)` tensors.
  /// Unpacks `N` tensors from this tensor by chipping it along the `axis` dimension, where `N`
  /// is inferred from this tensor's shape. For example, given a tensor with shape
  /// `[A, B, C, D]`:
  ///
  ///   - If `axis == 0` then the `i`-th tensor in the returned array is the slice
  ///     `self[i, :, :, :]` and each tensor in that array will have shape `[B, C, D]`.
  ///     (Note that the dimension unpacked along is gone, unlike
  ///     `Tensor.split(numSplits:alongAxis)`, or `Tensor.split(sizes:alongAxis)`).
  ///   - If `axis == 1` then the `i`-th tensor in the returned array is the slice
  ///     `value[:, i, :, :]` and each tensor in that array will have shape `[A, C, D]`.
  ///   - Etc.
  ///
  /// This is the opposite of `Tensor.init(stacking:alongAxis:)`.
  ///
  /// - Parameters:
  ///   - axis: Dimension along which to unstack. Negative values wrap around.
  ///
  /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of
  ///   the provided tensors.
  ///
  /// - Returns: Array containing the unstacked tensors.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func unstacked(alongAxis axis: Int = 0) -> [Tensor] {
    ensureValid(axis: axis)
    let posAxis = axis < 0 ? axis + rank : axis
    return _Raw.unpack(value: self, num: Int64(shape[posAxis]), axis: Int64(posAxis))
  }

  /// Splits a tensor into multiple tensors. The tensor is split along dimension `axis` into
  /// `count` smaller tensors. This requires that `count` evenly divides `shape[axis]`.
  ///
  /// For example:
  /// ```
  /// // 'value' is a tensor with shape [5, 30]
  /// // Split 'value' into 3 tensors along dimension 1:
  /// let parts = value.split(count: 3, alongAxis: 1)
  /// parts[0] // has shape [5, 10]
  /// parts[1] // has shape [5, 10]
  /// parts[2] // has shape [5, 10]
  /// ```
  ///
  /// - Parameters:
  ///   - count: Number of splits to create.
  ///   - axis: The dimension along which to split this tensor. Negative values wrap around.
  ///
  /// - Precondition: `count` must divide the size of dimension `axis` evenly.
  /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of
  ///   the provided tensors.
  ///
  /// - Returns: An array containing the tensors part.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func split(count: Int, alongAxis axis: Int = 0) -> [Tensor] {
    ensureValid(axis: axis)
    let canonicalAxis = axis < 0 ? axis + rank : axis
    precondition(
      shape[canonicalAxis] % count == 0,
      "Number of ways to split should evenly divide the split dimension.")
    return _Raw.split(splitDim: canonicalAxis, value: self, numSplit: Int64(count))
  }

  /// Splits a tensor into multiple tensors. The tensor is split  into `sizes.shape[0]` pieces.
  /// The shape of the `i`-th piece has the same shape as this tensor except along dimension
  /// `axis` where the size is `sizes[i]`.
  ///
  /// For example:
  /// ```
  /// // 'value' is a tensor with shape [5, 30]
  /// // Split 'value' into 3 tensors with sizes [4, 15, 11] along dimension 1:
  /// let parts = value.split(sizes: Tensor<Int32>([4, 15, 11]), alongAxis: 1)
  /// parts[0] // has shape [5, 4]
  /// parts[1] // has shape [5, 15]
  /// parts[2] // has shape [5, 11]
  /// ```
  ///
  /// - Parameters:
  ///   - sizes: 1-D tensor containing the size of each split.
  ///   - axis: Dimension along which to split this tensor. Negative values wrap around.
  ///
  /// - Precondition: The values in `sizes` must add up to the size of dimension `axis`.
  /// - Precondition: `axis` must be in the range `[-rank, rank)`, where `rank` is the rank of
  ///   the provided tensors.
  ///
  /// - Returns: Array containing the tensors parts.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func split(sizes: Tensor<Int32>, alongAxis axis: Int = 0) -> [Tensor] {
    fatalError()
//    ensureValid(axis: axis)
//    precondition(
//      shapeTensor[axis] == sizes.sum(),
//      "The values in sizes must add up to the size of dimension axis.")
//    return _Raw.splitV(
//      value: self,
//      sizeSplits: sizes,
//      splitDim: Tensor<Int32>(Int32(axis), on: device),
//      numSplit: Int64(sizes.shape[0]))
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func split(sizes: [Int], alongAxis axis: Int = 0) -> [Tensor] {
    ensureValid(axis: axis)
    let canonicalAxis = axis < 0 ? axis + rank : axis
    precondition(
      shape[canonicalAxis] == sizes.reduce(0, +),
      "The values in sizes must add up to the size of dimension axis.")
    return _Raw.splitV(
      value: self,
      sizeSplits: sizes,
      splitDim: canonicalAxis)
  }

  /// Returns a tiled tensor, constructed by tiling this tensor.
  ///
  /// This constructor creates a new tensor by replicating this tensor `multiples` times. The
  /// constructed tensor's `i`'th dimension has `self.shape[i] * multiples[i]` elements, and the
  /// values of this tensor are replicated `multiples[i]` times along the `i`'th dimension. For
  /// example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.
  ///
  /// - Precondition: The expected `rank` of multiples must be `1`.
  /// - Precondition: The shape of `multiples` must be `[tensor.rank]`.
  /// - Precondition: All scalars in `multiples` must be non-negative.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func tiled(multiples: [Int]) -> Tensor {
    precondition(
      multiples.allSatisfy { $0 >= 0 },
      "All scalars in multiples must be non-negative.")
    return _Raw.tile(self, multiples: multiples)
  }

  /// Returns a tiled tensor, constructed by tiling this tensor.
  ///
  /// This constructor creates a new tensor by replicating this tensor `multiples` times. The
  /// constructed tensor's `i`'th dimension has `self.shape[i] * multiples[i]` elements, and the
  /// values of this tensor are replicated `multiples[i]` times along the `i`'th dimension. For
  /// example, tiling `[a b c d]` by `[2]` produces `[a b c d a b c d]`.
  ///
  /// - Precondition: The expected `rank` of multiples must be `1`.
  /// - Precondition: The shape of `multiples` must be `[tensor.rank]`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func tiled(multiples: Tensor<Int32>) -> Tensor {
    precondition(multiples.rank == 1, "The expected rank of multiples must be 1.")
    precondition(
      rank == multiples.shapeTensor.scalarized(),
      "The shape of multiples must be [tensor.rank].")
    return _Raw.tile(self, multiples: multiples)
  }

  /// Reshape to the shape of the specified `Tensor`.
  /// - Precondition: The number of scalars matches the new shape.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped<T>(like other: Tensor<T>) -> Tensor {
    reshaped(toShape: other.shapeTensor)
  }

  /// Reshape to the specified shape.
  /// - Precondition: The number of scalars matches the new shape.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped(to newShape: TensorShape) -> Tensor {
    _Raw.reshape(self, shape: newShape.dimensions.map(Int64.init))
  }

  /// Reshape to the specified `Tensor` representing a shape.
  /// - Precondition: The number of scalars matches the new shape.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reshaped(toShape newShape: Tensor<Int32>) -> Tensor {
    return _Raw.reshape(self, shape: newShape)
  }

  /// Return a copy of the tensor collapsed into a 1-D `Tensor`, in row-major order.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func flattened() -> Tensor {
    reshaped(to: [-1])
  }

  /// Returns a shape-expanded `Tensor`, with a dimension of 1 inserted at the specified shape
  /// indices.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func expandingShape(at axes: Int...) -> Tensor {
    expandingShape(at: axes)
  }

  /// Returns a shape-expanded `Tensor`, with a dimension of 1 inserted at the
  /// specified shape indices.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func expandingShape(at axes: [Int]) -> Tensor {
    var resultShape = self.shape.dimensions.map { Int64($0) }
    for i in axes {
      var dim = i
      if dim < 0 { dim += resultShape.count + 1 }
      resultShape.insert(1, at: dim)
    }
    return _Raw.reshape(self, shape: resultShape)
  }

  /// Returns a rank-lifted `Tensor` with a leading dimension of 1.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func rankLifted() -> Tensor {
    expandingShape(at: 0)
  }

  /// Removes the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
  /// specified, then all dimensions of size 1 will be removed.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func squeezingShape(at axes: Int...) -> Tensor {
    squeezingShape(at: axes)
  }

  /// Removes the specified dimensions of size 1 from the shape of a tensor. If no dimensions are
  /// specified, then all dimensions of size 1 will be removed.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func squeezingShape(at axes: [Int]) -> Tensor {
    _Raw.squeeze(self, squeezeDims: axes.map(Int32.init))
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: unstacked)
  func _vjpUnstacked(
    alongAxis axis: Int = 0
  ) -> (value: [Tensor], pullback: (Array<Tensor>.TangentVector) -> Tensor) {
    let result = unstacked(alongAxis: axis)
    return (result, { v in Tensor(stacking: v.base, alongAxis: axis) })
  }

  @inlinable
  @derivative(of: tiled)
  func _vjpTiled(multiples: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (
      tiled(multiples: multiples),
      { [shape = shapeTensor] v in
        let splitShape = Tensor<Int32>(stacking: [multiples, shape]).transposed()
          .flattened()
        let axes = Tensor<Int32>(
          rangeFrom: 0, to: Int32(splitShape.scalarCount), stride: 2, on: device)
        return v.reshaped(toShape: splitShape).sum(squeezingAxes: axes)
      }
    )
  }

  @inlinable
  @derivative(of: tiled)
  func _vjpTiled(multiples: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    (
      tiled(multiples: multiples),
      { v in
        let splits = zip(multiples, shape.dimensions).flatMap { [$0, $1] }
        let axes = Array(stride(from: 0, to: splits.count, by: 2))
        return v.reshaped(to: TensorShape(splits)).sum(squeezingAxes: axes)
      }
    )
  }

  @inlinable
  @derivative(of: split)
  func _vjpSplit(
    count: Int,
    alongAxis axis: Int = 0
  ) -> (value: [Tensor], pullback: (Array<Tensor>.TangentVector) -> Tensor) {
    let result = split(count: count, alongAxis: axis)
    return (result, { v in Tensor(concatenating: v.base, alongAxis: axis) })
  }

  @inlinable
  @derivative(of: split)
  func _vjpSplit(
    sizes: Tensor<Int32>,
    alongAxis axis: Int = 0
  ) -> (value: [Tensor], pullback: (Array<Tensor>.TangentVector) -> Tensor) {
    let result = split(sizes: sizes, alongAxis: axis)
    return (result, { v in Tensor(concatenating: v.base, alongAxis: axis) })
  }

  @inlinable
  @derivative(of: split)
  func _vjpSplit(
    sizes: [Int],
    alongAxis axis: Int = 0
  ) -> (value: [Tensor], pullback: (Array<Tensor>.TangentVector) -> Tensor) {
    let result = split(sizes: sizes, alongAxis: axis)
    return (result, { v in Tensor(concatenating: v.base, alongAxis: axis) })
  }

  @inlinable
  @derivative(of: reshaped)
  func _vjpReshaped(toShape newShape: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let value = reshaped(toShape: newShape)
    return (value, { [shape = shapeTensor] v in v.reshaped(toShape: shape) })
  }

  @inlinable
  @derivative(of: reshaped)
  func _vjpReshaped(toShape newShape: TensorShape) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let value = reshaped(to: newShape)
    return (value, { [shape = shape] v in v.reshaped(to: shape) })
  }

  @inlinable
  @derivative(of: expandingShape)
  func _vjpExpandingShape(at axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = self.expandingShape(at: axes)
    return (value, { v in v.squeezingShape(at: axes) })
  }

  @inlinable
  @derivative(of: squeezingShape)
  func _vjpSqueezingShape(at axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = squeezingShape(at: axes)
    return (value, { [shape = shapeTensor] v in v.reshaped(toShape: shape) })
  }
}

//===------------------------------------------------------------------------------------------===//
// Other Tensor Transformations
//===------------------------------------------------------------------------------------------===//

infix operator ++: AdditionPrecedence

extension Tensor {
  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(permutation: Tensor<Int32>) -> Tensor {
    _Raw.transpose(self, perm: permutation)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(withPermutations permutations: Tensor<Int32>) -> Tensor {
    transposed(permutation: permutations)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(permutation: [Int]) -> Tensor {
    _Raw.transpose(self, perm: permutation)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(withPermutations permutations: [Int]) -> Tensor {
    transposed(permutation: permutations)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(permutation: Int...) -> Tensor {
    transposed(permutation: permutation)
  }

  /// Returns a transposed tensor, with dimensions permuted in the specified order.
  @available(*, deprecated, renamed: "transposed(permutation:)")
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed(withPermutations permutations: Int...) -> Tensor {
    transposed(permutation: permutations)
  }

  /// Returns a transposed tensor, with dimensions permuted in reverse order.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func transposed() -> Tensor {
    return transposed(permutation: Array(stride(from: Int(rank - 1), to: -1, by: -1)))
  }

  /// Returns a tensor with specified dimensions reversed.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  /// - Precondition: There must be no duplication in `axes`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reversed(inAxes axes: Tensor<Int32>) -> Tensor {
    ensureValid(axes: axes)
    return _Raw.reverseV2(self, axis: axes)
  }

  /// Returns a tensor with specified dimensions reversed.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  /// - Precondition: There must be no duplication in `axes`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reversed(inAxes axes: [Int]) -> Tensor {
    precondition(
      axes.count == Set(axes.map { $0 < 0 ? $0 + rank : $0 }).count,
      "There must be no duplication in axes.")
    let axes = axes.map(Int32.init)
    return reversed(inAxes: Tensor<Int32>(axes, on: device))
  }

  /// Returns a tensor with specified dimensions reversed.
  /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
  /// - Precondition: There must be no duplication in `axes`.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func reversed(inAxes axes: Int...) -> Tensor {
    reversed(inAxes: axes)
  }

  /// Returns a concatenated tensor along the specified axis.
  /// - Precondition: The tensors must have the same dimensions, except for the
  ///   specified axis.
  /// - Precondition: The axis must be in the range `-rank..<rank`.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public func concatenated(with other: Tensor, alongAxis axis: Int = 0) -> Tensor {
    return Tensor(concatenating: [self, other], alongAxis: axis)
  }

  /// Concatenation operator.
  /// - Note: `++` is a custom operator that does not exist in Swift, but does
  ///   in Haskell/Scala. Its addition is not an insignificant language change
  ///   and may be controversial. The existence/naming of `++` will be discussed
  ///   during a later API design phase.
  @inlinable
  @differentiable(reverse where Scalar: TensorFlowFloatingPoint)
  public static func ++ (lhs: Tensor, rhs: Tensor) -> Tensor {
    return lhs.concatenated(with: rhs)
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func gathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>,
    alongAxis axis: Int = 0
  ) -> Tensor {
    ensureValid(axis: axis)
    return _Raw.gatherV2(
      params: self, indices: indices, axis: Tensor<Int32>(Int32(axis), on: device))
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func batchGathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>,
    alongAxis axis: Int = 1,
    batchDimensionCount: Int = 1
  ) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func gathering(where mask: Tensor<Bool>, alongAxis axis: Int = 0) -> Tensor {
    fatalError()
  }
}

@usableFromInline
@noDerivative
internal func invertPermutationArray<T: TensorFlowIndex>(_ permutation: [T]) -> [T] {
  let size = permutation.count
  var inverted = [T](repeating: -1, count: size)
  for i in 0..<size {
    let d = permutation[i]
    if d < 0 || d >= size {
      fatalError("\(d) is not between 0 and \(size)")
    }
    if inverted[Int(d)] != -1 {
      fatalError("\(d) is duplicated in the input.")
    }
    inverted[Int(d)] = T(i)
  }
  return inverted
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: transposed(permutation:))
  func _vjpTransposed(permutation: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    let value = transposed(permutation: permutation)
    return (value, { $0.transposed(permutation: _Raw.invertPermutation(permutation)) })
  }

  @inlinable
  @derivative(of: transposed(permutation:))
  func _vjpTransposed(permutation: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = transposed(permutation: permutation)
    let inverted = invertPermutationArray(permutation.map { Int64($0) })
    return (value, { $0.transposed(permutation: inverted.map { Int($0) }) })
  }

  @inlinable
  @derivative(of: transposed(permutation:))
  func _vjpTransposed(permutation: Int...) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = transposed(permutation: permutation)
    let inverted = invertPermutationArray(permutation.map { Int64($0) })
    return (value, { $0.transposed(permutation: inverted.map { Int($0) }) })
  }

  @inlinable
  @derivative(of: transposed)
  func _vjpTransposed() -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return (transposed(), { $0.transposed() })
  }

  @inlinable
  @derivative(of: reversed)
  func _vjpReversed(inAxes axes: Tensor<Int32>) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return (reversed(inAxes: axes), { $0.reversed(inAxes: axes) })
  }

  @inlinable
  @derivative(of: reversed)
  func _vjpReversed(inAxes axes: [Int]) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return (reversed(inAxes: axes), { $0.reversed(inAxes: axes) })
  }

  @inlinable
  @derivative(of: reversed)
  func _vjpReversed(inAxes axes: Int...) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    return (reversed(inAxes: axes), { $0.reversed(inAxes: axes) })
  }

  @inlinable
  @derivative(of: concatenated)
  func _vjpConcatenated(
    with other: Tensor,
    alongAxis axis: Int
  ) -> (value: Tensor, pullback: (Tensor) -> (Tensor, Tensor)) {
    let posAxis = axis < 0 ? axis + rank : axis
    let splits = [shape[posAxis], other.shape[posAxis]]
    return (
      concatenated(with: other, alongAxis: axis),
      { result in
        let gradients = result.split(sizes: splits, alongAxis: axis)
        return (gradients[0], gradients[1])
      }
    )
  }

  @inlinable
  @derivative(of: gathering)
  func _vjpGathering<Index: TensorFlowIndex>(
    atIndices indices: Tensor<Index>,
    alongAxis axis: Int = 0
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()

  }
}

extension Tensor {
  /// Returns the locations of non-zero / true values in this tensor.
  ///
  /// The coordinates are returned in a 2-D tensor where the first dimension (rows) represents the
  /// number of non-zero elements, and the second dimension (columns) represents the coordinates
  /// of the non-zero elements. Keep in mind that the shape of the output tensor can vary
  /// depending on how many true values there are in this tensor. Indices are output in row-major
  /// order.
  ///
  /// For example:
  /// ```
  /// // 'input' is [[true, false], [true, false]]
  /// // 'input' has 2 true values and so the output has 2 rows.
  /// // 'input' has rank of 2, and so the second dimension of the output has size 2.
  /// input.nonZeroIndices() // is [[0, 0], [1, 0]]
  ///
  /// // 'input' is [[[ true, false], [ true, false]],
  /// //             [[false,  true], [false,  true]],
  /// //             [[false, false], [false,  true]]]
  /// // 'input' has 5 true values and so the output has 5 rows.
  /// // 'input' has rank 3, and so the second dimension of the output has size 3.
  /// input.nonZeroIndices() // is [[0, 0, 0],
  ///                        //     [0, 1, 0],
  ///                        //     [1, 0, 1],
  ///                        //     [1, 1, 1],
  ///                        //     [2, 1, 1]]
  /// ```
  ///
  /// - Returns: A tensor with shape `(num_true, rank(condition))`.
  @inlinable
  public func nonZeroIndices() -> Tensor<Int64> {
    return _Raw.where_(self)
  }
}

//===------------------------------------------------------------------------------------------===//
// Broadcasting
//===------------------------------------------------------------------------------------------===//

// TODO: What about precedence? Why is this operator used for broadcasting?
infix operator .=

extension Tensor {
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func broadcasted(toShape shape: Tensor<Int32>) -> Tensor {
    return _Raw.broadcastTo(self, shape: shape)
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func broadcasted(to shape: TensorShape) -> Tensor {
    return broadcasted(toShape: Tensor<Int32>({ shape.dimensions.map(Int32.init) }(), on: device))
  }

  /// Broadcast to the same shape as the specified `Tensor`.
  /// - Precondition: The specified shape must be compatible for broadcasting.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func broadcasted<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor {
    return broadcasted(toShape: other.shapeTensor)
  }

  @inlinable
  public static func .= (lhs: inout Tensor, rhs: Tensor) {
    lhs = rhs.broadcasted(like: lhs)
  }
}

extension Tensor where Scalar: Numeric {
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func unbroadcasted(toShape otherShape: Tensor<Int32>) -> Tensor {
    // TODO: Simplify this once differentiating control flow is supported.
    return unbroadcasted(
      to: {
        precondition(otherShape.rank == 1)
        return TensorShape(otherShape.scalars.map(Int.init))
      }())
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func unbroadcasted<OtherScalar>(like other: Tensor<OtherScalar>) -> Tensor {
    return unbroadcasted(toShape: other.shapeTensor)
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func unbroadcasted(to shape: TensorShape) -> Tensor {
    let dimensions = self.shape.dimensions
    var otherDimensions = shape.dimensions
    let rankDifference = dimensions.count - otherDimensions.count
    precondition(
      rankDifference >= 0,
      """
      The rank of 'self' must be greater than or equal to the number of \
      dimensions in the destination shape
      """)
    if rankDifference > 0 {
      otherDimensions.insert(contentsOf: repeatElement(1, count: rankDifference), at: 0)
    }
    assert(dimensions.count == otherDimensions.count)
    var axes: [Int] = []
    axes.reserveCapacity(dimensions.count)
    for (i, (dim, otherDim)) in zip(dimensions, otherDimensions).enumerated() {
      if dim == otherDim { continue }
      if otherDim == 1 {
        axes.append(i)
        continue
      }
      preconditionFailure("Cannot unbroadcast \(self.shape) to \(shape)")
    }
    return sum(alongAxes: axes).reshaped(to: shape)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: broadcasted)
  func _vjpBroadcasted(toShape shape: Tensor<Int32>) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (
      broadcasted(toShape: shape),
      { [originalShape = shapeTensor] v in
        v.unbroadcasted(toShape: originalShape)
      }
    )
  }

  @inlinable
  @derivative(of: unbroadcasted)
  func _vjpUnbroadcasted(to shape: TensorShape) -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    return (
      unbroadcasted(to: shape),
      { [originalShape = shapeTensor] v in
        v.broadcasted(toShape: originalShape)
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Padding
//===------------------------------------------------------------------------------------------===//

extension Tensor where Scalar: Numeric {
  /// A mode that dictates how a tensor is padded.
  public enum PaddingMode {
    /// Pads with constant value.
    case constant(Scalar)
    /// Mirrors values along padding dimensions, excluding the edge value.
    case reflect
    /// Mirrors values along padding dimensions, including the edge value.
    case symmetric
  }

  /// Returns a tensor padded with constant according to the specified padding sizes.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func padded(forSizes sizes: [(before: Int, after: Int)], with value: Scalar = 0)
    -> Tensor
  {
    padded(forSizes: sizes, mode: .constant(value))
  }

  /// Returns a padded tensor according to the specified padding sizes and mode.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func padded(forSizes sizes: [(before: Int, after: Int)], mode: PaddingMode) -> Tensor {
    let paddings = Tensor<Int32>(
      shape: [sizes.count, 2],
      scalars: sizes.flatMap { [Int32($0.before), Int32($0.after)] }, on: device)
    switch mode {
    case .constant(let constantValue):
      return _Raw.padV2(self, paddings: paddings, constantValues: Tensor(constantValue, on: device))
    case .reflect:
      return _Raw.mirrorPad(self, paddings: paddings, mode: .reflect)
    case .symmetric:
      return _Raw.mirrorPad(self, paddings: paddings, mode: .symmetric)
    }
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: padded)
  func _vjpPadded(
    forSizes sizes: [(before: Int, after: Int)],
    mode: PaddingMode
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let result = padded(forSizes: sizes, mode: mode)
    return (
      result,
      { [rank = rank, shape = shapeTensor] v in
        let device = v.device
        let paddings = Tensor<Int32>(
          shape: [sizes.count, 2],
          scalars: sizes.flatMap { [Int32($0.before), Int32($0.after)] }, on: device)
        switch mode {
        case .constant:
          let padBefore = _Raw.slice(
            paddings,
            begin: Tensor<Int32>([0, 0], on: device),
            size: Tensor<Int32>([Int32(rank), 1], on: device))
          let begin = padBefore.reshaped(to: [-1])
          return v.slice(lowerBounds: begin, sizes: shape)
        case .reflect:
          return _Raw.mirrorPadGrad(v, paddings: paddings, mode: .reflect)
        case .symmetric:
          return _Raw.mirrorPadGrad(v, paddings: paddings, mode: .symmetric)
        }
      }
    )
  }
}

//===------------------------------------------------------------------------------------------===//
// Indexing and Slicing
//===------------------------------------------------------------------------------------------===//

// TODO: Negative indexing and strides syntax.

extension Tensor {
  /// Extracts a slice from the tensor defined by lower and upper bounds for
  /// each dimension.
  ///
  /// - Parameter lowerBounds: The lower bounds at each dimension.
  /// - Parameter upperBounds: The upper bounds at each dimension.
  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func slice(lowerBounds: [Int], upperBounds: [Int]) -> Tensor {
    // TODO: Precondition `lowerBounds.count == upperBounds.count`,
    // preferably in graph.
    // TODO: Differentiating control flow is not supported yet, thus the thunks.
    let zipped = zip(upperBounds, lowerBounds)
    let sizes = withoutDerivative(at: zipped) { zipped in zipped.map { $0 - $1 } }
    return slice(lowerBounds: lowerBounds, sizes: sizes)
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func slice(lowerBounds: Tensor<Int32>, sizes: Tensor<Int32>) -> Tensor {
    return _Raw.slice(self, begin: lowerBounds, size: sizes)
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  public func slice(lowerBounds: [Int], sizes: [Int]) -> Tensor {
    return _Raw.slice(self, begin: lowerBounds, size: sizes)
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @inlinable
  @derivative(of: slice)
  internal func _vjpSlice(
    lowerBounds: Tensor<Int32>,
    sizes: Tensor<Int32>
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    _vjpSlice(
      lowerBounds: lowerBounds.scalars.map { Int($0) }, sizes: sizes.scalars.map { Int($0) })
  }

  @inlinable
  @derivative(of: slice(lowerBounds:sizes:))
  internal func _vjpSlice(
    lowerBounds: [Int],
    sizes: [Int]
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    let value = slice(lowerBounds: lowerBounds, sizes: sizes)
    let afterPaddings = zip(zip(shape, value.shape).map { $0 - $1 }, lowerBounds).map { $0 - $1 }
    return (
      value,
      { v in
        let linearizedPaddings = zip(lowerBounds, afterPaddings).flatMap { [$0, $1] }
        return _Raw.pad(v, paddings: linearizedPaddings)
      }
    )
  }
}

extension Tensor {
  @inlinable
  func subscript2(_ indexPath: Int32) -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(reverse, wrt: self where Scalar: TensorFlowFloatingPoint)
  subscript(_ ranges: Int64) -> Tensor {
    get {
      return self.subscript2(Int32(2))
    }
  }
}

extension Tensor where Scalar: TensorFlowFloatingPoint {
  @usableFromInline
  @derivative(of: subscript2)
  func _vjpSubscript(
    _ indexPath: Int32
  ) -> (value: Tensor, pullback: (Tensor) -> Tensor) {
    fatalError()
  }
}


//
//public protocol QuensorRangeExpression {
//  var tensorRange: TensorRange { get }
//}
//
//public struct Quensor<Scalar> {
////  public var x: Scalar
//
//  @inlinable
//  subscript(_ indexPath: Int32) -> Quensor {
//    get {
//      fatalError()
//    }
//    set {
//      fatalError()
//    }
//  }
//
//  @inlinable
//  @differentiable(reverse, wrt: self where Scalar: FloatingPoint)
//  subscript(_ ranges: TensorRangeExpression...) -> Quensor {
//    get {
//      return self[{ Int32(2) }()]
//    }
//    set {
//      self[{ Int32(2) }()] = newValue
//    }
//  }
//}
//
//extension Quensor where Scalar: FloatingPoint {
//  @usableFromInline
//  @derivative(of: subscript(_:))
//  func _vjpSubscript(
//    _ indexPath: Int32
//  ) -> (value: Quensor, pullback: (Quensor) -> Quensor) {
//    fatalError()
//  }
//}
//
//extension Quensor: Equatable where Scalar: Equatable {
//  public static func == (lhs: Quensor, rhs: Quensor) -> Bool {
//    fatalError()
//  }
//}
//
//extension Quensor: AdditiveArithmetic where Scalar: Numeric {
//  public static var zero: Quensor {
//    fatalError()
//  }
//
//  public static func + (lhs: Quensor, rhs: Quensor) -> Quensor {
//    fatalError()
//  }
//
//  public static func - (lhs: Quensor, rhs: Quensor) -> Quensor {
//    fatalError()
//  }
//}
//
//extension Quensor: Differentiable where Scalar: FloatingPoint {
//  public typealias TangentVector = Quensor
//}

//===------------------------------------------------------------------------------------------===//
// Precondition utilities
//===------------------------------------------------------------------------------------------===//

extension Tensor {
  /// Returns `true` iff `k` denotes an axis of `self`.
  @usableFromInline
  internal func isValid<T: BinaryInteger>(axis k: T) -> Bool {
    let axis = Int(k)
    return axis >= -rank && axis < rank
  }

  /// Returns `true` iff each element of `axes` denotes an axis of `self`.
  @usableFromInline
  internal func areValid<T: BinaryInteger>(axes: [T]) -> Bool {
    return axes.allSatisfy { isValid(axis: $0) }
  }

  /// Returns `true` iff each element of `axes` denotes an axis of `self`.
  ///
  /// - Precondition: `axes` has rank 0 or rank 1.
  @usableFromInline
  internal func areValid(
    axes: Tensor<Int32>,
    file: StaticString = #file,
    line: UInt = #line
  ) -> Bool {
    precondition(
      axes.rank < 2,
      "Axes must have rank 0 or rank 1; axes has rank \(axes.rank) with values \(axes.scalars).",
      file: file,
      line: line)
    return areValid(axes: axes.scalars)
  }

  /// Checks that each element of `axes` denotes an axis of `self`, and stops the program with a
  /// diagnostic otherwise.
  @usableFromInline
  func ensureValid(
    axes: Tensor<Int32>,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    precondition(
      areValid(axes: axes, file: file, line: line),
      "All axes must be in `-rank..<rank` when calling \(function) (rank: \(rank), axes: \(axes))",
      file: file,
      line: line)
  }

  /// Checks that each element of `axes` denotes an axis of `self`, and stops the program with a
  /// diagnostic otherwise.
  @usableFromInline
  func ensureValid(
    axes: [Int],
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    precondition(
      areValid(axes: axes),
      "All axes must be in `-rank..<rank` when calling \(function) (rank: \(rank), axes: \(axes))",
      file: file,
      line: line)
  }

  /// Checks that `k` denotes an axis of `self`, and stops the program with a diagnostic otherwise.
  @usableFromInline
  func ensureValid(
    axis k: Int,
    function: StaticString = #function,
    file: StaticString = #file,
    line: UInt = #line
  ) {
    precondition(
      isValid(axis: k),
      "Axis must be in `-rank..<rank` when calling \(function) (rank: \(rank), axis: \(k))",
      file: file,
      line: line)
  }
}
