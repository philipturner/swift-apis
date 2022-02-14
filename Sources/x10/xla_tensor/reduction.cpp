// Copyright 2020 TensorFlow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/compiler/tf2xla/xla_tensor/reduction.h"

#include <cmath>

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/helpers.h"
#include "tensorflow/compiler/tf2xla/xla_tensor/tensor_util.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/literal_util.h"

namespace swift_xla {
namespace {

struct ReductionInfo {
  std::vector<int64_t> new_dimensions;
  XlaHelpers::DynamicSize element_count;
};

struct SummationResult {
  ReductionInfo rinfo;
  xla::XlaOp result;
};

ReductionInfo GetReductionInfo(xla::XlaOp input, const xla::Shape& shape,
                               absl::Span<const int64_t> dimensions,
                               bool keep_reduced_dimensions) {
  ReductionInfo rinfo;
  size_t idim = 0;
  for (int64_t i = 0; i < shape.rank(); ++i) {
    if (idim < dimensions.size() && dimensions[idim] == i) {
      ++idim;
      if (keep_reduced_dimensions) {
        rinfo.new_dimensions.push_back(1);
      }
    } else {
      rinfo.new_dimensions.push_back(shape.dimensions(i));
    }
  }
  rinfo.element_count = XlaHelpers::GetDimensionsSize({input}, dimensions);
  return rinfo;
}

xla::XlaComputation CreateAllComputation(xla::PrimitiveType type) {
  xla::XlaBuilder builder("AllComputation");
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::XlaOp zero = xla::Zero(&builder, type);
  xla::XlaOp one = xla::One(&builder, type);
  xla::Select(xla::And(xla::Ne(x, zero), xla::Ne(y, zero)), one, zero);
  return ConsumeValue(builder.Build());
}

xla::XlaComputation CreateAnyComputation(xla::PrimitiveType type) {
  xla::XlaBuilder builder("AnyComputation");
  xla::XlaOp x =
      xla::Parameter(&builder, 0, xla::ShapeUtil::MakeShape(type, {}), "x");
  xla::XlaOp y =
      xla::Parameter(&builder, 1, xla::ShapeUtil::MakeShape(type, {}), "y");
  xla::XlaOp zero = xla::Zero(&builder, type);
  xla::XlaOp one = xla::One(&builder, type);
  xla::Select(xla::Or(xla::Ne(x, zero), xla::Ne(y, zero)), one, zero);
  return ConsumeValue(builder.Build());
}

xla::XlaOp GetScaleValue(xla::XlaOp input, xla::XlaOp count,
                         xla::PrimitiveType type) {
  xla::XlaOp zero = xla::Zero(input.builder(), XlaHelpers::TypeOfXlaOp(count));
  xla::XlaOp one = xla::One(input.builder(), type);
  xla::XlaOp scale = xla::Select(xla::Ne(count, zero),
                                 one / xla::ConvertElementType(count, type),
                                 xla::NanValue(input.builder(), type));
  return input * scale;
}

xla::XlaOp AverageValue(xla::XlaOp input, xla::XlaOp reduced) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp num_elements =
      XlaHelpers::GetDimensionsSize({input},
                                    XlaHelpers::GetAllDimensions(input_shape))
          .size;
  return GetScaleValue(reduced, num_elements, input_shape.element_type());
}

SummationResult CreateSummation(xla::XlaOp input,
                                absl::Span<const int64_t> dimensions,
                                bool keep_reduced_dimensions, bool scale) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init_value = xla::Zero(input.builder(), shape.element_type());
  SummationResult result;
  result.rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  result.result = xla::Reduce(
      input, init_value, XlaHelpers::CreateAddComputation(shape.element_type()),
      dimensions);
  if (scale) {
    result.result = GetScaleValue(
        result.result, result.rinfo.element_count.size, shape.element_type());
  }
  if (keep_reduced_dimensions) {
    result.result =
        XlaHelpers::DynamicReshape(result.result, result.rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp CreateProduct(xla::XlaOp input,
                         absl::Span<const int64_t> dimensions,
                         bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp init_value = xla::One(input.builder(), shape.element_type());
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMulComputation(shape.element_type()),
      dimensions);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

}  // namespace

xla::XlaOp BuildBinaryCrossEntropy(xla::XlaOp input, xla::XlaOp target,
                                   const absl::optional<xla::XlaOp>& weight,
                                   ReductionMode reduction) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp xweight;
  if (weight) {
    // Higher layers guarantee that weight and input have the same shape.
    xweight = *weight;
  } else {
    xweight =
        XlaHelpers::ScalarBroadcast<float>(1.0, input_shape, target.builder());
  }
  xla::XlaOp one = xla::One(input.builder(), input_shape.element_type());
  xla::XlaOp result = -xweight * (target * xla::Log(input) +
                                  (one - target) * xla::Log(one - input));
  if (reduction == ReductionMode::kNone) {
    return result;
  }
  xla::XlaOp reduced_result = xla::ReduceAll(
      result, xla::Zero(input.builder(), input_shape.element_type()),
      XlaHelpers::CreateAddComputation(input_shape.element_type()));
  if (reduction == ReductionMode::kMean) {
    reduced_result = AverageValue(result, reduced_result);
  }
  return reduced_result;
}

xla::XlaOp BuildBinaryCrossEntropyBackward(
    xla::XlaOp grad_output, xla::XlaOp input, xla::XlaOp target,
    const absl::optional<xla::XlaOp>& weight, ReductionMode reduction) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp xweight;
  if (weight) {
    // Higher layers guarantee that weight and input have the same shape.
    xweight = *weight;
  } else {
    xweight =
        XlaHelpers::ScalarBroadcast<float>(1.0, input_shape, target.builder());
  }
  xla::XlaOp one = xla::One(input.builder(), input_shape.element_type());
  xla::XlaOp result = xweight * (input - target) / input / (one - input);
  if (reduction == ReductionMode::kNone) {
    return result * grad_output;
  }
  result = result * grad_output;
  if (reduction == ReductionMode::kMean) {
    result = AverageValue(input, result);
  }
  return result;
}

xla::XlaOp BuildL1Loss(xla::XlaOp input, xla::XlaOp target,
                       ReductionMode reduction) {
  xla::XlaOp result = xla::Abs(input - target);
  if (reduction == ReductionMode::kNone) {
    return result;
  }
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  result = xla::ReduceAll(
      result, xla::Zero(input.builder(), input_shape.element_type()),
      XlaHelpers::CreateAddComputation(input_shape.element_type()));
  if (reduction == ReductionMode::kMean) {
    result = AverageValue(input, result);
  }
  return result;
}

xla::XlaOp BuildL1LossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                               xla::XlaOp target, ReductionMode reduction) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  if (reduction == ReductionMode::kNone) {
    xla::XlaOp one = xla::One(input.builder(), input_shape.element_type());
    xla::XlaOp mask = xla::Select(xla::Ge(input, target), one, -one);
    return mask * grad_output;
  }
  xla::XlaOp grad_value = grad_output;
  if (reduction == ReductionMode::kMean) {
    grad_value = AverageValue(input, grad_value);
  }
  return xla::Select(xla::Ge(input, target), grad_value, -grad_value);
}

xla::XlaOp BuildMseLoss(xla::XlaOp input, xla::XlaOp target,
                        ReductionMode reduction) {
  xla::XlaOp diff = input - target;
  xla::XlaOp result = diff * diff;
  if (reduction == ReductionMode::kNone) {
    return result;
  }
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  result = xla::ReduceAll(
      result, xla::Zero(input.builder(), input_shape.element_type()),
      XlaHelpers::CreateAddComputation(input_shape.element_type()));
  if (reduction == ReductionMode::kMean) {
    int64_t num_elements = xla::ShapeUtil::ElementsIn(input_shape);
    if (num_elements == 0) {
      return xla::NanValue(input.builder(), input_shape.element_type());
    } else {
      xla::XlaOp scale_value = XlaHelpers::ScalarValue<double>(
          1.0 / static_cast<double>(num_elements), input_shape.element_type(),
          input.builder());
      result = result * scale_value;
    }
  }
  return result;
}

xla::XlaOp BuildMseLossBackward(xla::XlaOp grad_output, xla::XlaOp input,
                                xla::XlaOp target, ReductionMode reduction) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp two = XlaHelpers::ScalarValue<double>(
      2, input_shape.element_type(), input.builder());
  xla::XlaOp d_input = two * (input - target);
  if (reduction == ReductionMode::kNone) {
    return d_input * grad_output;
  }
  xla::XlaOp grad_value = grad_output;
  if (reduction == ReductionMode::kMean) {
    int64_t num_elements = xla::ShapeUtil::ElementsIn(input_shape);
    xla::XlaOp scale_value = XlaHelpers::ScalarValue<double>(
        1.0 / static_cast<double>(num_elements), input_shape.element_type(),
        input.builder());
    grad_value = grad_output * scale_value;
  }
  return d_input * grad_value;
}

xla::XlaOp BuildCumulativeComputation(xla::XlaOp input, int64_t dim,
                                      const xla::XlaComputation& reducer,
                                      xla::XlaOp init, bool exclusive,
                                      bool reverse) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  std::vector<int64_t> window_strides(input_shape.rank(), 1);
  std::vector<int64_t> window_dims(input_shape.rank(), 1);
  window_dims[dim] = input_shape.dimensions(dim);
  std::vector<std::pair<int64_t, int64_t>> padding(input_shape.rank());
  padding[dim].first = input_shape.dimensions(dim) - (exclusive ? 0 : 1);
  if (reverse) std::swap(padding[dim].first, padding[dim].second);
  xla::XlaOp result = xla::ReduceWindowWithGeneralPadding(
      input, init, reducer, window_dims, window_strides,
      /*base_dilations=*/{}, /*window_dilations=*/{}, padding);
  if (exclusive) {
    int64_t offset = reverse ? 1 : 0;
    result = xla::SliceInDim(result, offset,
                             input_shape.dimensions(dim) + offset, 1, dim);
  }
  return result;
}

xla::XlaOp BuildMean(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                     bool keep_reduced_dimensions) {
  return CreateSummation(input, dimensions, keep_reduced_dimensions,
                         /*scale=*/true)
      .result;
}

xla::XlaOp BuildStdDeviation(xla::XlaOp input,
                             absl::Span<const int64_t> dimensions,
                             bool keep_reduced_dimensions, bool unbiased) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp mean =
      BuildMean(input, dimensions, /*keep_reduced_dimensions*/ true);
  xla::XlaOp bcast_mean =
      xla::BroadcastInDim(mean, input_shape.dimensions(),
                          xla::util::Iota<int64_t>(input_shape.rank()));
  xla::XlaOp input_mean_diff = input - bcast_mean;
  xla::XlaOp squared_var = input_mean_diff * input_mean_diff;
  xla::XlaOp squared_result;
  if (unbiased) {
    SummationResult sum_result = CreateSummation(
        squared_var, dimensions, keep_reduced_dimensions, /*scale=*/false);
    squared_result = GetScaleValue(
        sum_result.result,
        sum_result.rinfo.element_count.size -
            xla::One(input.builder(), XlaHelpers::TypeOfXlaOp(
                                          sum_result.rinfo.element_count.size)),
        input_shape.element_type());
  } else {
    SummationResult sum_result = CreateSummation(
        squared_var, dimensions, keep_reduced_dimensions, /*scale=*/true);
    squared_result = sum_result.result;
  }
  return xla::Sqrt(squared_result);
}

xla::XlaOp BuildSum(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions) {
  return CreateSummation(input, dimensions, keep_reduced_dimensions,
                         /*scale=*/false)
      .result;
}

xla::XlaOp BuildProd(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                     bool keep_reduced_dimensions) {
  return CreateProduct(input, dimensions, keep_reduced_dimensions);
}

xla::XlaOp BuildMaxInDim(xla::XlaOp input, int64_t dim,
                         bool keep_reduced_dimensions) {
  return BuildMaxInDims(input, {dim}, keep_reduced_dimensions);
}

xla::XlaOp BuildMaxInDims(xla::XlaOp input,
                          absl::Span<const int64_t> dimensions,
                          bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(shape.element_type());
  xla::XlaOp init_value = XlaHelpers::ScalarValue(
      min_max.min, shape.element_type(), input.builder());
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  if (rinfo.element_count.scalar_size) {
    // When can only assert this if dimensions are not dynamic.
    XLA_CHECK_GT(*rinfo.element_count.scalar_size, 0);
  }
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMaxComputation(shape.element_type()),
      dimensions);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildMinInDim(xla::XlaOp input, int64_t dim,
                         bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  XlaHelpers::MinMax min_max = XlaHelpers::MinMaxValues(shape.element_type());
  xla::XlaOp init_value = XlaHelpers::ScalarValue(
      min_max.max, shape.element_type(), input.builder());
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, {dim}, keep_reduced_dimensions);
  if (rinfo.element_count.scalar_size) {
    // When can only assert this if dimensions are not dynamic.
    XLA_CHECK_GT(*rinfo.element_count.scalar_size, 0);
  }
  xla::XlaOp result = xla::Reduce(
      input, init_value, XlaHelpers::CreateMinComputation(shape.element_type()),
      {dim});
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildArgMax(xla::XlaOp input, int64_t dim, bool keepdim) {
  const xla::Shape* shape = &XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = XlaHelpers::DynamicReshape(operand,
                                         {xla::ShapeUtil::ElementsIn(*shape)});
    shape = &XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMax(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim);
  if (keepdim) {
    auto dimensions = xla::util::ToVector<int64_t>(shape->dimensions());
    dimensions[dim] = 1;
    result = XlaHelpers::DynamicReshape(result, dimensions);
  }
  return result;
}

xla::XlaOp BuildArgMin(xla::XlaOp input, int64_t dim, bool keepdim) {
  const xla::Shape* shape = &XlaHelpers::ShapeOfXlaOp(input);
  xla::XlaOp operand = input;
  if (dim < 0) {
    dim = 0;
    operand = XlaHelpers::DynamicReshape(operand,
                                         {xla::ShapeUtil::ElementsIn(*shape)});
    shape = &XlaHelpers::ShapeOfXlaOp(operand);
  }
  xla::XlaOp result = xla::ArgMin(
      operand,
      GetDevicePrimitiveType(xla::PrimitiveType::S64, /*device=*/nullptr), dim);
  if (keepdim) {
    auto dimensions = xla::util::ToVector<int64_t>(shape->dimensions());
    dimensions[dim] = 1;
    result = XlaHelpers::DynamicReshape(result, dimensions);
  }
  return result;
}

xla::XlaOp BuildAll(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::One(shape.element_type()));
  xla::XlaOp result =
      xla::Reduce(input, init_value, CreateAllComputation(shape.element_type()),
                  dimensions);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildAny(xla::XlaOp input, absl::Span<const int64_t> dimensions,
                    bool keep_reduced_dimensions) {
  const xla::Shape& shape = XlaHelpers::ShapeOfXlaOp(input);
  ReductionInfo rinfo =
      GetReductionInfo(input, shape, dimensions, keep_reduced_dimensions);
  xla::XlaOp init_value = xla::ConstantLiteral(
      input.builder(), xla::LiteralUtil::Zero(shape.element_type()));
  xla::XlaOp result =
      xla::Reduce(input, init_value, CreateAnyComputation(shape.element_type()),
                  dimensions);
  if (keep_reduced_dimensions) {
    result = XlaHelpers::DynamicReshape(result, rinfo.new_dimensions);
  }
  return result;
}

xla::XlaOp BuildLogsumexp(xla::XlaOp input,
                          absl::Span<const int64_t> dimensions,
                          bool keep_reduced_dimensions) {
  // Use the log-sum-exp trick to avoid overflow.
  xla::XlaOp max_in_dim =
      BuildMaxInDims(input, dimensions, /*keep_reduced_dimensions=*/true);
  xla::XlaOp exps = xla::Exp(input - max_in_dim);
  xla::XlaOp sums = CreateSummation(exps, dimensions, keep_reduced_dimensions,
                                    /*scale=*/false)
                        .result;
  xla::XlaOp logs = xla::Log(sums);
  // If keep_reduced_dimensions is false, we need to reshape the max_in_dim to
  // the reduced shape before doing the add.
  if (!keep_reduced_dimensions) {
    max_in_dim =
        CreateSummation(max_in_dim, dimensions, keep_reduced_dimensions,
                        /*scale=*/false)
            .result;
  }
  return logs + max_in_dim;
}

}  // namespace swift_xla
