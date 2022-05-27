// swift-tools-version:5.7
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "TensorFlow",
  products: [
    .executable(name: "TensorFlow", targets: ["TensorFlow"]),
  ],
  dependencies: [],
  targets: [
    .target(name: "TensorFlow"),
  ]
)
