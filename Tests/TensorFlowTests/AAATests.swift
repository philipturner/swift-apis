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

import XCTest
@_spi(Reflection) import ReflectionMirror

struct ASimpleKPI {
  var w = 1
  
  var recursivelyAllKeyPaths: [PartialKeyPath<Self>] {
    var result: [PartialKeyPath<Self>] = []
    
    var out = [PartialKeyPath<Self>]()
    _forEachFieldWithKeyPath(of: Self.self, options: .ignoreUnknown) { name, kp in
      out.append(kp)
      return true
    }
    
    for kp in out {
      result.append(kp)
    }
    return result
  }
  
  var _recursivelyAllKeyPathsTypeErased: [AnyKeyPath] {
    recursivelyAllKeyPaths.map { $0 as AnyKeyPath }
  }
}

struct AMixedKPI {
  var string = "foo"
  
  var recursivelyAllKeyPaths: [PartialKeyPath<Self>] {
    var result: [PartialKeyPath<Self>] = []
    
    var out = [PartialKeyPath<Self>]()
    _forEachFieldWithKeyPath(of: Self.self, options: .ignoreUnknown) { name, kp in
      out.append(kp)
      return true
    }
    
    for kp in out {
      result.append(kp)
    }
    return result
  }
  
  var _recursivelyAllKeyPathsTypeErased: [AnyKeyPath] {
    recursivelyAllKeyPaths.map { $0 as AnyKeyPath }
  }
}

struct ANestedKPI {
  var simple = ASimpleKPI()
  var mixed = AMixedKPI()
  
  var recursivelyAllKeyPaths: [PartialKeyPath<Self>] {
    var result: [PartialKeyPath<Self>] = []
    
    var out = [PartialKeyPath<Self>]()
    _forEachFieldWithKeyPath(of: Self.self, options: .ignoreUnknown) { name, kp in
      out.append(kp)
      return true
    }
    
    for kp in out {
      result.append(kp)
      if let nested = self[keyPath: kp] as? ASimpleKPI {
        for nkp in nested._recursivelyAllKeyPathsTypeErased {
          result.append(kp.appending(path: nkp)!)
        }
      }
      else if let nested = self[keyPath: kp] as? AMixedKPI {
        for nkp in nested._recursivelyAllKeyPathsTypeErased {
          result.append(kp.appending(path: nkp)!)
        }
      }
    }
    return result
  }
}

final class AAATests: XCTestCase {
  func testMyCrasher() {
    var x = ANestedKPI()
    
    let arr = [\ANestedKPI.mixed.string]
    let xr1 = x.recursivelyAllKeyPaths
    let xr2 = x.recursivelyAllKeyPaths

    _ = arr == xr1
    _ = arr == xr2
  }
}
