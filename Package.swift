// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
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

import PackageDescription
import class Foundation.ProcessInfo

var conditionalPackageDependencies: [Package.Dependency] = []
var conditionalSwiftSettings: [SwiftSetting] = []
var conditionalTargetDependencies: [Target.Dependency] = []

if ProcessInfo.processInfo.environment["TENSORFLOW_USE_RELEASE_TOOLCHAIN"] != nil {
  conditionalPackageDependencies += [
    .package(url: "https://github.com/philipturner/differentiation", .branch("main")),
    .package(url: "https://github.com/philipturner/swift-reflection-mirror", .branch("main")),
  ]
  conditionalSwiftSettings += [
    .define("TENSORFLOW_USE_RELEASE_TOOLCHAIN"),
  ]
  conditionalTargetDependencies += [
    .product(name: "_Differentiation", package: "differentiation"),
    .product(name: "ReflectionMirror", package: "swift-reflection-mirror"),
  ]
}

let package = Package(
  name: "TensorFlow",
  platforms: [
    .macOS(.v10_13)
  ],
  products: [
    .library(
      name: "TensorFlow",
      type: .dynamic,
      targets: ["TensorFlow"]),
    .library(
      name: "x10_optimizers_optimizer",
      type: .dynamic,
      targets: ["x10_optimizers_optimizer"]),
    .library(
      name: "x10_optimizers_tensor_visitor_plan",
      type: .dynamic,
      targets: ["x10_optimizers_tensor_visitor_plan"]),
//    .library(
//      name: "x10_training_loop",
//      type: .dynamic,
//      targets: ["x10_training_loop"]),
  ],
  dependencies: [
    .package(url: "https://github.com/apple/swift-numerics", .branch("main")),
    .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
  ] + conditionalPackageDependencies,
  targets: [
    .target(
      name: "CTensorFlow",
      dependencies: []),
    .target(
      name: "CX10Modules",
      dependencies: []),
    .target(
      name: "TensorFlow",
      dependencies: [
        "PythonKit",
        "CTensorFlow",
        "CX10Modules",
        .product(name: "Numerics", package: "swift-numerics"),
      ] + conditionalTargetDependencies,
      swiftSettings: [
        .define("DEFAULT_BACKEND_EAGER"),
      ] + conditionalSwiftSettings),
    .target(
      name: "x10_optimizers_tensor_visitor_plan",
      dependencies: ["TensorFlow"],
      path: "Sources/x10",
      sources: [
        "swift_bindings/optimizers/TensorVisitorPlan.swift",
      ],
      swiftSettings: conditionalSwiftSettings),
    .target(
      name: "x10_optimizers_optimizer",
      dependencies: [
        "x10_optimizers_tensor_visitor_plan",
        "TensorFlow",
      ],
      path: "Sources/x10",
      sources: [
        "swift_bindings/optimizers/Optimizer.swift",
        "swift_bindings/optimizers/Optimizers.swift",
      ],
      swiftSettings: conditionalSwiftSettings),
//     .target(
//       name: "x10_training_loop",
//       dependencies: ["TensorFlow"],
//       path: "Sources/x10",
//       sources: [
//         "swift_bindings/training_loop.swift",
//       ],
//       swiftSettings: conditionalSwiftSettings),
    .target(
      name: "Experimental",
      dependencies: conditionalTargetDependencies,
      path: "Sources/third_party/Experimental",
      swiftSettings: conditionalSwiftSettings),
    .testTarget(
      name: "ExperimentalTests",
      dependencies: ["Experimental"],
      swiftSettings: conditionalSwiftSettings),
    .testTarget(
      name: "TensorFlowTests",
      dependencies: ["TensorFlow"],
      exclude: [],
      swiftSettings: conditionalSwiftSettings),
  ]
)
