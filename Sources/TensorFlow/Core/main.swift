import _Differentiation

public struct Tensor: Differentiable & AdditiveArithmetic {
  @inlinable
  func subscriptIndexPath() -> Tensor {
    fatalError()
  }

  @inlinable
  @differentiable(reverse, wrt: self)
  func subscriptRanges() -> Tensor {
    subscriptIndexPath()
  }
  
  @usableFromInline
  @derivative(of: subscriptIndexPath)
  func _vjpSubscriptIndexPath() -> (
    value: Tensor, pullback: (Tensor) -> Tensor
  ) {
    fatalError()
  }
}
