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

namespace {

template <typename Head, typename... Tail>
struct IsValidArithOpType {
  static constexpr bool value = !std::is_base_of<Val, Head>::value ?
      false : IsValidArithOpType<Tail...>::value;
};

template <typename Type>
struct IsValidArithOpType<Type> {
  static constexpr bool value = std::is_base_of<Val, Type>::value;
};

template <typename Head, typename... Tail>
struct HasTensorView {
  static constexpr bool value = std::is_base_of<TensorView, Head>::value ?
      true : HasTensorView<Tail...>::value;
};

template <typename Type>
struct HasTensorView<Type> {
  static constexpr bool value = std::is_base_of<TensorView, Type>::value;
};

// Define return type of arithmetic operations. Currently, TensorView
// when any of operands is TensorView, and Val otherwise.
template <typename... OpTypes>
struct ArithOpRetType {
  using Type = typename std::conditional<HasTensorView<OpTypes...>::value,
                                         TensorView, Val>::type;
};

} // namespace

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

// Overload for the case when operands are not Val*. Template matching
// fails when their classes are not valid types as determined by IsValidArithOpType
template <typename OpType1, typename OpType2,
          std::enable_if_t<IsValidArithOpType<OpType1, OpType2>::value>* = nullptr>
TORCH_CUDA_API typename ArithOpRetType<OpType1, OpType2>::Type*
binaryOp(BinaryOpType type, OpType1* v1, OpType2* v2) {
  return static_cast<typename ArithOpRetType<OpType1, OpType2>::Type*>(
      binaryOp(type, static_cast<Val*>(v1), static_cast<Val*>(v2)));
}

// Perform a reduction operation on v1, initial value for reduction is init,
// reduces across axes, and reduction operation defined by BinaryOp.
TORCH_CUDA_API Val* reductionOp(
    BinaryOpType reduction_op_type,
    const std::vector<int>& axes,
    Val* init,
    Val* v1);

// BINARY OPAERATIONS
template <typename OpType1, typename OpType2>
TORCH_CUDA_API typename ArithOpRetType<OpType1, OpType2>::Type* add(
    OpType1* v1, OpType2* v2) {
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
