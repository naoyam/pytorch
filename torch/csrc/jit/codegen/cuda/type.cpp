#include <torch/csrc/jit/codegen/cuda/type.h>

#include <stdexcept>
#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {

// Return highest on list (smallest enum val)
DataType promote_type(const DataType& t1, const DataType& t2) {
  TORCH_CHECK(
      DataType::Null != t1 && DataType::Null != t2,
      "Expected promotable DataTypes but got: ",
      t1,
      " and ",
      t2);
  return t1 < t2 ? t1 : t2;
}

// Return highest on list (smallest enum val)
ValType promote_type(const ValType& t1, const ValType& t2) {
  TORCH_CHECK(
      t1 >= ValType::TensorView && t2 >= ValType::TensorView,
      "Expected promotable ValTypes but got: ",
      t1,
      " and ",
      t2);
  // Check that it's a promotable type (with dtype)
  // static_assert??
  return t1 < t2 ? t1 : t2;
}

bool is_cast_legal(const DataType& t1, const DataType& t2) {
  if ((DataType::Null == t1) || (DataType::Null == t2))
    return false;
  // In theory there could be stronger real check here in the future
  return true;
}

template <typename T>
struct _enum_class_hash {
  size_t operator()(T v) const {
    return static_cast<size_t>(v);
  }
};
template <typename KeyType, typename ValType>
using _enum_unordered_map =
    std::unordered_map<KeyType, ValType, _enum_class_hash<KeyType>>;
static _enum_unordered_map<DataType, std::string> data_type_string_map{
    {DataType::Float, "float"},
    {DataType::Int, "size_t"}};
static _enum_unordered_map<ValType, std::string> val_type_string_map{
    {ValType::TensorIndex, "TensorIndex"},
    {ValType::TensorView, "TensorView"},
    {ValType::TensorDomain, "TensorDomain"},
    {ValType::IterDomain, "IterDomain"},
    {ValType::Scalar, "Scalar"},
    {ValType::NamedScalar, "NamedScalar"}};

static _enum_unordered_map<ExprType, std::string> expr_type_string_map{
    {ExprType::UnaryOp, "UnaryOp"},
    {ExprType::BinaryOp, "BinaryOp"},
    {ExprType::ForLoop, "ForLoop"},
    {ExprType::IfThenElse, "IfThenElse"},
    {ExprType::Allocate, "Allocate"},
    {ExprType::Split, "Split"},
    {ExprType::Merge, "Merge"},
    {ExprType::Reorder, "Reorder"}};
static _enum_unordered_map<UnaryOpType, std::string> unary_op_type_string_map{
    {UnaryOpType::Neg,        "neg"},
    {UnaryOpType::Cast,       "cast"},
    {UnaryOpType::Abs,        "fabs"},
    {UnaryOpType::Log,        "logf"},
    {UnaryOpType::Log10,      "log10f"},
    {UnaryOpType::Log1p,      "log1pf"},
    {UnaryOpType::Log2,       "log2f"},
    {UnaryOpType::Lgamma,     "lgammaf"},
    {UnaryOpType::Exp,        "expf"},
    {UnaryOpType::Expm1,      "expm1f"},
    {UnaryOpType::Erf,        "erff"},
    {UnaryOpType::Erfc,       "erfcf"},
    {UnaryOpType::Cos,        "cosf"},
    {UnaryOpType::Acos,       "acosf"},
    {UnaryOpType::Cosh,       "coshf"},
    {UnaryOpType::Sin,        "sinf"},
    {UnaryOpType::Asin,       "asinf"},
    {UnaryOpType::Sinh,       "sinhf"},
    {UnaryOpType::Tan,        "tanf"},
    {UnaryOpType::Atan,       "atanf"},
    {UnaryOpType::Atanh,      "atanhf"},
    {UnaryOpType::Sqrt,       "sqrtf"},
    {UnaryOpType::Rsqrt,      "rsqrtf"},
    {UnaryOpType::Ceil,       "ceilf"},
    {UnaryOpType::Floor,      "floorf"},
    {UnaryOpType::Round,      "roundf"},
    {UnaryOpType::Trunc,      "truncf"},
    {UnaryOpType::Frac,       "fracf"},
    {UnaryOpType::Reciprocal, "reciprocal"},
    {UnaryOpType::Relu,       "relu"},
    //{UnaryOpType::Threshold,  "threshold"}, // TODO: This is not really a Unary op.
    {UnaryOpType::Sigmoid,    "sigmoid"}};
static _enum_unordered_map<UnaryOpType, std::string>
    unary_op_type_inline_op_string_map{{UnaryOpType::Neg,        "-"}};
static _enum_unordered_map<UnaryOpType, std::function<void(std::ostream&, const std::string&)>>
    unary_op_type_func_string_map{
                                   {UnaryOpType::Reciprocal, [](std::ostream& out, const std::string& in) { out <<  "1.f / " << in; }},
                                   {UnaryOpType::Relu,       [](std::ostream& out, const std::string& in) { out << in <<  " < 0.f ? 0.f : " << in; }},
                                   {UnaryOpType::Sigmoid,    [](std::ostream& out, const std::string& in) { out << "1.f / (1.f + expf(-" << in << "))"; }},
 								 };
static _enum_unordered_map<BinaryOpType, std::string> binary_op_type_string_map{
    {BinaryOpType::Add,       "add"},
    {BinaryOpType::Sub,       "sub"},
    {BinaryOpType::Mul,       "mul"},
    {BinaryOpType::Div,       "div"},
    {BinaryOpType::Mod,       "mod"},
    {BinaryOpType::CeilDiv,   "ceilDiv"},
    {BinaryOpType::And,       "and"},
    {BinaryOpType::Atan2,     "atan2f"},
    {BinaryOpType::Min,       "fminf"},
    {BinaryOpType::Max,       "fmaxf"},
    {BinaryOpType::Pow,       "powf"},
    {BinaryOpType::Rem,       "remainderf"},
    {BinaryOpType::LT,        "lessThan"},
    {BinaryOpType::LE,        "lessThanOrEqual"},
    {BinaryOpType::GT,        "greaterThan"},
    {BinaryOpType::GE,        "greaterThanOrEqual"},
    {BinaryOpType::NE,        "notEqual"},
    {BinaryOpType::Eq,        "equal"}
   };
static _enum_unordered_map<BinaryOpType, std::string>
    binary_op_type_inline_op_string_map{{BinaryOpType::Add,    "+"},
                                        {BinaryOpType::Sub,    "-"},
                                        {BinaryOpType::Mul,    "*"},
                                        {BinaryOpType::Div,    "/"},
                                        {BinaryOpType::Mod,    "%"},
                                        {BinaryOpType::And,    "&&"},
                                        {BinaryOpType::LT,     "<"},
                                        {BinaryOpType::LE,     "<="},
                                        {BinaryOpType::GT,     ">"},
                                        {BinaryOpType::GE,     ">="},
                                        {BinaryOpType::NE,     "!="},
                                        {BinaryOpType::Eq,     "=="}
                                       };

static _enum_unordered_map<ParallelType, std::string> parallel_type_string_map{
    {ParallelType::BIDz, "blockIdx.z"},
    {ParallelType::BIDy, "blockIdx.y"},
    {ParallelType::BIDx, "blockIdx.x"},
    {ParallelType::TIDz, "threadIdx.z"},
    {ParallelType::TIDy, "threadIdx.y"},
    {ParallelType::TIDx, "threadIdx.x"},
    {ParallelType::Vectorize, "Vectorize"},
    {ParallelType::Unroll, "Unroll"},
    {ParallelType::Serial, "Serial"}};

static _enum_unordered_map<at::ScalarType, DataType> at_type_map{
    {at::ScalarType::Float, DataType::Float},
    {at::ScalarType::Int, DataType::Int}};

static _enum_unordered_map<ParallelType, std::string> thread_size_string_map{
    {ParallelType::BIDz, "gridDim.z"},
    {ParallelType::BIDy, "gridDim.y"},
    {ParallelType::BIDx, "gridDim.x"},
    {ParallelType::TIDz, "blockDim.z"},
    {ParallelType::TIDy, "blockDim.y"},
    {ParallelType::TIDx, "blockDim.x"}};

DataType aten_to_data_type(const at::ScalarType& scalar_type) {
  TORCH_INTERNAL_ASSERT(
      at_type_map.count(scalar_type) != 0, "No string found for scalar type.");
  return at_type_map[scalar_type];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ValType vtype) {
  TORCH_INTERNAL_ASSERT(
      val_type_string_map.count(vtype) != 0, "No string found for val type.");
  return out << val_type_string_map[vtype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const DataType dtype) {
  TORCH_INTERNAL_ASSERT(
      data_type_string_map.count(dtype) != 0, "No string found for data type.");
  return out << data_type_string_map[dtype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ExprType etype) {
  TORCH_INTERNAL_ASSERT(
      expr_type_string_map.count(etype) != 0, "No string found for expr type.");
  return out << expr_type_string_map[etype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const UnaryOpType uotype) {
  TORCH_INTERNAL_ASSERT(
      unary_op_type_string_map.count(uotype) != 0,
      "No string found for UnaryOp type.");
  return out << unary_op_type_string_map[uotype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const BinaryOpType botype) {
  TORCH_INTERNAL_ASSERT(
      binary_op_type_string_map.count(botype) != 0,
      "No string found for BinaryOp type.");
  return out << binary_op_type_string_map[botype];
}

std::string stringify(const ParallelType ptype) {
  TORCH_INTERNAL_ASSERT(
      parallel_type_string_map.count(ptype) != 0,
      "No string found for parallel type.");
  return parallel_type_string_map[ptype];
}

TORCH_CUDA_API std::ostream& operator<<(
    std::ostream& out,
    const ParallelType ptype) {
  return out << stringify(ptype);
}

TORCH_CUDA_API c10::optional<std::string> inline_op_str(
    const UnaryOpType uotype) {
  if (unary_op_type_inline_op_string_map.find(uotype) ==
      unary_op_type_inline_op_string_map.end()) {
    return c10::nullopt;
  } else {
    return unary_op_type_inline_op_string_map[uotype];
  }
}

TORCH_CUDA_API c10::optional<std::string> inline_op_str(
    const BinaryOpType botype) {
  if (binary_op_type_inline_op_string_map.find(botype) ==
      binary_op_type_inline_op_string_map.end()) {
    return c10::nullopt;
  } else {
    return binary_op_type_inline_op_string_map[botype];
  }
}

<<<<<<< HEAD
std::string stringifyThreadSize(const ParallelType ptype) {
  TORCH_INTERNAL_ASSERT(
      thread_size_string_map.find(ptype) != thread_size_string_map.end(),
      "Could not find size of the thread type ",
      ptype);
  return thread_size_string_map[ptype];
}
=======
TORCH_CUDA_API c10::optional<std::function<void(std::ostream&, const std::string&)>> func_str(
    const UnaryOpType uotype) {
  if (unary_op_type_func_string_map.find(uotype) ==
      unary_op_type_func_string_map.end()) {
    return c10::nullopt;
  } else {
    return unary_op_type_func_string_map[uotype];
  }
}

>>>>>>> Added a mapping of Unary Types to Functions that generate strings for operations like Relu and Sigmoid.
} // namespace fuser
} // namespace jit
} // namespace torch
