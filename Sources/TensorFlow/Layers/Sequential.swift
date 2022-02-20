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
@_spi(Reflection) import Swift

public protocol _Layer: Differentiable {
  associatedtype Input
  associatedtype Output: Differentiable

  @differentiable(reverse)
  func callAsFunction(_ input: Input) -> Output
}

public protocol _KeyPathIterable {
  associatedtype AllKeyPaths: Sequence
    where AllKeyPaths.Element == PartialKeyPath<Self>
}

public struct Sequential<Layer1: _Layer & _KeyPathIterable, Layer2: _Layer>: _Layer
where
  Layer1.Output == Layer2.Input
{
  public var layer1: Layer1
  public var layer2: Layer2

  @differentiable(reverse)
  public func callAsFunction(_ input: Layer1.Input) -> Layer2.Output {
    layer2(layer1(input))
  }
}


//public struct Sequential<Layer1: _Module, Layer2: _Layer>: _Module
//where
//  Layer1.Output == Layer2.Input
//{
//  public var layer1: Layer1
//  public var layer2: Layer2
//
//  public init(_ layer1: Layer1, _ layer2: Layer2) {
//    self.layer1 = layer1
//    self.layer2 = layer2
//  }
//
//  @differentiable(reverse, wrt: self)
//  public func callAsFunction(_ input: Layer1.Input) -> Layer2.Output {
//    layer2(layer1(input))
//  }
//}
//
//extension Sequential: _Layer where Layer1: _Layer {
//  @differentiable(reverse)
//  public func callAsFunction(_ input: Layer1.Input) -> Layer2.Output {
//    layer2(layer1(input))
//  }
//}
