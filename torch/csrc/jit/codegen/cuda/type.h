#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <cstdint>
#include <iostream>
#include <string>

namespace torch {
namespace jit {
namespace fuser {

// Order of strength
enum class ValType {
  TensorIndex,
  TensorDomain,
  IterDomain,
  TensorView,
  Scalar,
  NamedScalar
};

enum class DataType { Float, Int, Null };

enum class ExprType {
  UnaryOp,
  BinaryOp,
  ForLoop,
  IfThenElse,
  Allocate,
  Split,
  Merge,
  Reorder
};

enum class UnaryOpType {
  Abs,
  Acos,
  Asin,
  Atan,
  Atanh,
  Cast,
  Ceil,
  Cos,
  Cosh,
  Exp,
  Expm1,
  Erf,
  Erfc,
  Floor,
  Frac,
  Gelu,
  Lgamma,
  Log,
  Log10,
  Log1p,
  Log2,
  Neg,
  //RandLike,
  Reciprocal,
  Relu,
  Rsqrt,
  Round,
  Sigmoid,
  Sin,
  Sinh,
  Sqrt,
  Tan,
  Tanh,
  Trunc
};

// TODO: Order of this list is important as it affects type promotion. it's not
// in the right order now.
enum class BinaryOpType {
  // Math Ops
  Add,
  Atan2,
  Div,
  Fmod,
  Max,
  Min,
  Mul,
  Pow,
  Remainder,
  Sub,
  //TypeAs,

  // Logical Ops
  // Int operations, leave position oif Mod we depend on its location of first
  Mod,
  CeilDiv,
  And,
  Eq,
  GE,
  GT,
  LE,
  LT,
  NE
};

enum class ParallelType {
  BIDz,
  BIDy,
  BIDx,
  TIDz,
  TIDy,
  TIDx,
  Vectorize,
  Unroll,
  Serial
};

ValType promote_type(const ValType& t1, const ValType& t2);
DataType promote_type(const DataType& t1, const DataType& t2);
bool is_cast_legal(const DataType& t1, const DataType& t2);

DataType aten_to_data_type(const at::ScalarType& scalar_type);

TORCH_CUDA_API std::ostream& operator<<(std::ostream&, const ValType);
TORCH_CUDA_API std::ostream& operator<<(std::ostream&, const DataType);
TORCH_CUDA_API std::ostream& operator<<(std::ostream&, const ExprType);
TORCH_CUDA_API std::ostream& operator<<(std::ostream&, const UnaryOpType);
TORCH_CUDA_API std::ostream& operator<<(std::ostream&, const BinaryOpType);
TORCH_CUDA_API std::ostream& operator<<(std::ostream&, const ParallelType);

std::string stringify(const ParallelType);
std::string stringifyThreadSize(const ParallelType);

TORCH_CUDA_API c10::optional<std::string> inline_op_str(const UnaryOpType);
TORCH_CUDA_API c10::optional<std::string> inline_op_str(const BinaryOpType);

} // namespace fuser
} // namespace jit
} // namespace torch
