#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

#include <type_traits>

struct Val;

/*
 * The operations defined in this header is intended as user facing functions.
 * Generally users should not directly instantiate temporary TensorViews they
 * should instead use the functions below which will automatically create IR
 * nodes, and return a resulting TensorView of correctly tracked shapes.
 */

namespace torch {
namespace jit {
namespace fuser {

// Promotion logic between two values, returns a new val from resulting type
// promotion.
TORCH_CUDA_API Val* promoteNew(Val* v1, Val* v2);

// Insertion of casting op to dtype, returns new resulting val
TORCH_CUDA_API Val* castOp(DataType dtype, Val* v1);

// Perform unary op type and return the output
TORCH_CUDA_API Val* unaryOp(UnaryOpType type, Val* v1);

// Perform binary op type on v1 and v2 and return a type promoted output.
// Mod, CeilDiv, and LT are considered Int only output operations for now.
TORCH_CUDA_API Val* binaryOp(BinaryOpType type, Val* v1, Val* v2);
TORCH_CUDA_API TensorView* binaryOp(BinaryOpType type, TensorView* v1, Val* v2);
TORCH_CUDA_API TensorView* binaryOp(BinaryOpType type, Val* v1, TensorView* v2);
TORCH_CUDA_API TensorView* binaryOp(BinaryOpType type, TensorView* v1, TensorView* v2);

template <typename OpType1, typename OpType2,
          bool OpType1IsVal = std::is_base_of<Val, OpType1>::value,
          bool OpType2IsVal = std::is_base_of<Val, OpType2>::value,
          bool OpType1IsTensorView = std::is_base_of<TensorView, OpType1>::value,
          bool OpType2IsTensorView = std::is_base_of<TensorView, OpType2>::value>
struct BinaryOpRetType;

template <typename OpType1, typename OpType2>
struct BinaryOpRetType<OpType1, OpType2, true, true, false, false> {
  using Type = Val;
};

template <typename OpType1, typename OpType2>
struct BinaryOpRetType<OpType1, OpType2, true, true, true, false> {
  using Type = TensorView;
};

template <typename OpType1, typename OpType2>
struct BinaryOpRetType<OpType1, OpType2, true, true, false, true> {
  using Type = TensorView;
};

template <typename OpType1, typename OpType2>
struct BinaryOpRetType<OpType1, OpType2, true, true, true, true> {
  using Type = TensorView;
};

// Perform a reduction operation on v1, initial value for reduction is init,
// reduces across axes, and reduction operation defined by BinaryOp.
TORCH_CUDA_API Val* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    Val* v1);

// BINARY OPAERATIONS
template <typename OpType1, typename OpType2>
TORCH_CUDA_API typename BinaryOpRetType<OpType1, OpType2>::Type* add(
    OpType1* v1, OpType2* v2) {
  //return binaryOp2<OpType1, OpType2>(BinaryOpType::Add, v1, v2);
  return binaryOp(BinaryOpType::Add, v1, v2);
}
TORCH_CUDA_API Val* sub(Val* v1, Val* v2);
TORCH_CUDA_API Val* mul(Val* v1, Val* v2);
TORCH_CUDA_API Val* div(Val* v1, Val* v2);
TORCH_CUDA_API Val* mod(Val* v1, Val* v2);
TORCH_CUDA_API Val* lt(Val* v1, Val* v2);
TORCH_CUDA_API Val* ceilDiv(Val* v1, Val* v2);
TORCH_CUDA_API Val* andOp(Val* v1, Val* v2);

// REDUCTION OPERATIONS
TORCH_CUDA_API Val* sum(Val* v1, const std::vector<int>& reduction_axes);

// COMPOUND OPERATIONS
TORCH_CUDA_API Val* add_alpha(Val* v1, Val* v2, Val* s);
TORCH_CUDA_API Val* sub_alpha(Val* v1, Val* v2, Val* s);
TORCH_CUDA_API Val* lerp(Val* start, Val* end, Val* weight);
TORCH_CUDA_API Val* addcmul(Val* v1, Val* v2, Val* v3, Val* s);
TORCH_CUDA_API Val* where(Val* c, Val* v1, Val* v2);

// TERNARY OPERATIONS
TORCH_CUDA_API Val* threshold(Val* in, Val* thresh, Val* value);
TORCH_CUDA_API Val* clamp(Val* in, Val* min_val, Val* max_val);

} // namespace fuser
} // namespace jit
} // namespace torch
