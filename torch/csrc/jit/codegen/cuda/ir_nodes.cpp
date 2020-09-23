#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_cloner.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/compute_at.h>

#include <sstream>

namespace torch {
namespace jit {
namespace fuser {

namespace {

class ScalarCheck : OptInDispatch {
 public:
  static bool sameAs(Val* v1, Val* v2) {
    if (v1 == v2)
      return true;

    if (v1->getValType() != v2->getValType())
      return false;

    if (v1->getDataType() != v2->getDataType())
      return false;

    ScalarCheck sc(v1, v2);
    return sc.same_;
  }

 private:
  void handle(Bool* b) override {
    same_ = v1_->as<Bool>()->sameAs(v2_->as<Bool>());
  }

  void handle(Float* f) override {
    same_ = v1_->as<Float>()->sameAs(v2_->as<Float>());
  }

  void handle(Half* h) override {
    same_ = v1_->as<Half>()->sameAs(v2_->as<Half>());
  }

  void handle(Int* i) override {
    same_ = v1_->as<Int>()->sameAs(v2_->as<Int>());
  }

  void handle(NamedScalar* ns) override {
    same_ = v1_->as<NamedScalar>()->sameAs(v2_->as<NamedScalar>());
  }

  ScalarCheck(Val* _v1, Val* _v2) : v1_(_v1), v2_(_v2) {
    OptInDispatch::handle(v1_);
  }

 private:
  Val* v1_ = nullptr;
  Val* v2_ = nullptr;
  bool same_ = false;
};

} // namespace

bool areEqualScalars(Val* v1, Val* v2) {
  return ScalarCheck::sameAs(v1, v2);
}

Bool::Bool(const Bool* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Bool::sameAs(const Bool* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

Float::Float(const Float* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Float::sameAs(const Float* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

Half::Half(const Half* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Half::sameAs(const Half* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

Int::Int(const Int* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), maybe_value_(src->maybe_value_) {}

bool Int::sameAs(const Int* const other) const {
  if (isConst() && other->isConst())
    return *value() == *(other->value());
  return this == other;
}

UnaryOp::UnaryOp(UnaryOpType _type, Val* _out, Val* _in)
    : Expr(ExprType::UnaryOp), unary_op_type_{_type}, out_{_out}, in_{_in} {
  addOutput(_out);
  addInput(_in);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

UnaryOp::UnaryOp(const UnaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      unary_op_type_(src->unary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

bool UnaryOp::sameAs(const UnaryOp* const other) const {
  if (type() != other->type())
    return false;
  return as<Expr>()->sameAs(other);
}

BinaryOp::BinaryOp(BinaryOpType _type, Val* _out, Val* _lhs, Val* _rhs)
    : Expr(ExprType::BinaryOp),
      binary_op_type_{_type},
      out_{_out},
      lhs_{_lhs},
      rhs_{_rhs} {
  addOutput(_out);
  addInput(_lhs);
  addInput(_rhs);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

BinaryOp::BinaryOp(const BinaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      binary_op_type_(src->binary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      lhs_(ir_cloner->clone(src->lhs_)),
      rhs_(ir_cloner->clone(src->rhs_)) {}

bool BinaryOp::sameAs(const BinaryOp* other) const {
  if (getBinaryOpType() != other->getBinaryOpType())
    return false;
  if (!(lhs()->sameAs(other->lhs()) && rhs()->sameAs(other->rhs())))
    return false;
  return true;
}

TernaryOp::TernaryOp(
    TernaryOpType _type,
    Val* _out,
    Val* _in1,
    Val* _in2,
    Val* _in3)
    : Expr(ExprType::TernaryOp),
      ternary_op_type_{_type},
      out_{_out},
      in1_{_in1},
      in2_{_in2},
      in3_{_in3} {
  addOutput(_out);
  addInput(_in1);
  addInput(_in2);
  addInput(_in3);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

TernaryOp::TernaryOp(const TernaryOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      ternary_op_type_(src->ternary_op_type_),
      out_(ir_cloner->clone(src->out_)),
      in1_(ir_cloner->clone(src->in1_)),
      in2_(ir_cloner->clone(src->in2_)),
      in3_(ir_cloner->clone(src->in3_)) {}

bool TernaryOp::sameAs(const TernaryOp* other) const {
  if (getTernaryOpType() != other->getTernaryOpType())
    return false;
  if (!(in1()->sameAs(other->in1()) && in2()->sameAs(other->in2()) &&
        in3()->sameAs(other->in3())))
    return false;
  return true;
}

BroadcastOp::BroadcastOp(Val* _out, Val* _in)
    : Expr(ExprType::BroadcastOp), out_(_out), in_(_in) {
  auto out_type = _out->getValType().value();
  auto in_type = _in->getValType().value();

  TORCH_INTERNAL_ASSERT(
      out_type == ValType::TensorView && in_type == ValType::TensorView,
      "Cannot braodcast a non-tensor object.");

  // This is a generic check that root dims of a consumer and producer match.
  // Maybe we shouldn't relegate it to this constructor.
  const auto c_tv = out()->as<TensorView>();
  const auto p_tv = in()->as<TensorView>();

  const auto& c_root = c_tv->getRootDomain();
  const auto& p_root = p_tv->getMaybeRFactorDomain();

  const auto root_p2c = TensorDomain::mapDomainPandC(p_root, c_root);

  std::vector<bool> c_mapped(c_root.size(), false);
  std::vector<bool> p_mapped(p_root.size(), false);

  for (auto pair_entry : root_p2c) {
    auto p_i = pair_entry.first;
    p_mapped[p_i] = true;
    auto c_i = pair_entry.second;
    c_mapped[c_i] = true;
  }

  bool bad_mismatch = false;

  for (size_t i = 0; i < c_root.size(); i++) {
    if (!c_mapped[i]) {
      if (!c_root[i]->isBroadcast()) {
        bad_mismatch = true;
      }
    }
  }

  for (size_t i = 0; i < p_root.size(); i++) {
    if (!p_mapped[i]) {
      if (!p_root[i]->isReduction()) {
        bad_mismatch = true;
      }
    }
  }

  TORCH_INTERNAL_ASSERT(
      !bad_mismatch,
      "Invalid broadcast op. Non-broadcasted dims don't match from input to output.");

  addOutput(_out);
  addInput(_in);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

BroadcastOp::BroadcastOp(const BroadcastOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

bool BroadcastOp::sameAs(const BroadcastOp* const other) const {
  return other->in() == in() && other->out() == out();
}

ReductionOp::ReductionOp(
    BinaryOpType _reduction_op_type,
    Val* _init,
    Val* _out,
    Val* _in)
    : Expr(ExprType::ReductionOp),
      reduction_op_type_(_reduction_op_type),
      init_(_init),
      out_(_out),
      in_(_in) {
  if (_out->getValType().value() == ValType::TensorView) {
    TORCH_INTERNAL_ASSERT(
        _in->getValType() == ValType::TensorView &&
            _out->getValType() == ValType::TensorView,
        "Reduction operation was created that does not have tensor inputs and outputs.");

    TORCH_INTERNAL_ASSERT(
        TensorDomain::noReductions(
            _in->as<TensorView>()->getMaybeRFactorDomain())
                .size() == _out->as<TensorView>()->getRootDomain().size(),
        "Reduction operation created with mismatched domains.");

  } else {
    TORCH_INTERNAL_ASSERT(
        _in->getValType() == ValType::TensorIndex &&
            _out->getValType() == ValType::TensorIndex,
        "Reduction operation was created that does not have tensor inputs and outputs.");
  }
  TORCH_INTERNAL_ASSERT(
      _init->isConstScalar(),
      "Tried to create a reduction operation whith an initial value that isn't a constant.");

  addOutput(_out);
  addInput(_in);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

ReductionOp::ReductionOp(const ReductionOp* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      reduction_op_type_(src->reduction_op_type_),
      init_(ir_cloner->clone(src->init_)),
      out_(ir_cloner->clone(src->out_)),
      in_(ir_cloner->clone(src->in_)) {}

bool ReductionOp::sameAs(const ReductionOp* other) const {
  return (
      in()->sameAs(other->in()) &&
      getReductionOpType() == other->getReductionOpType() &&
      init()->sameAs(other->init()));
}

IterDomain::IterDomain(
    Val* _start,
    Val* _extent,
    ParallelType _parallel_type,
    IterType _iter_type,
    bool _is_rfactor_domain)
    : Val(ValType::IterDomain, DataType::Int, false),
      start_(_start),
      extent_(_extent),
      parallel_type_(_parallel_type),
      iter_type_(_iter_type),
      is_rfactor_domain_(_is_rfactor_domain) {
  TORCH_CHECK(
      !(isRFactorProduct() && isBroadcast()),
      "IterDomain cannot be both a broadcast and rfactor domain.");

  TORCH_INTERNAL_ASSERT(
      _extent->isAnInt(),
      "Cannot create an iter domain over an extent that is not an int but recieved ",
      _extent,
      " .");

  TORCH_INTERNAL_ASSERT(
      _start->isAnInt(),
      "Cannot create an iter domain with a start that is not an int but recieved ",
      _extent,
      " .");

  // TORCH_INTERNAL_ASSERT(!kir::isLoweredVal(_extent));

  name_ = fusion_->registerVal(this);
}

IterDomain::IterDomain(const IterDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      start_(ir_cloner->clone(src->start_)),
      extent_(ir_cloner->clone(src->extent_)),
      parallel_type_(src->parallel_type_),
      iter_type_(src->iter_type_),
      is_rfactor_domain_(src->is_rfactor_domain_) {}

bool IterDomain::sameAs(const IterDomain* const other) const {
  if (other == this)
    return true;

  bool is_same = isReduction() == other->isReduction() &&
      getParallelType() == other->getParallelType();
  is_same = is_same && ScalarCheck::sameAs(extent(), other->extent());
  is_same = is_same && ScalarCheck::sameAs(start(), other->start());

  return is_same;
}

IterDomain* IterDomain::merge(IterDomain* outer, IterDomain* inner) {
  TORCH_CHECK(
      outer->start()->isZeroInt() && inner->start()->isZeroInt(),
      "Merging IterDomains with starting values that aren't 0 is not supported at this time.");
  TORCH_CHECK(
      outer->isReduction() == inner->isReduction(),
      "Merging IterDomains requires that their iteration types match.");
  TORCH_CHECK(
      outer->getParallelType() == inner->getParallelType(),
      "Merging IterDomains requires that their parallel types match.");

  Val* merged_id_size = mul(outer->extent(), inner->extent());

  IterType itype = outer->getIterType();

  if (outer->isBroadcast() && inner->isBroadcast()) {
    if (outer->getIterType() == IterType::BroadcastWithStride ||
        inner->getIterType() == IterType::BroadcastWithStride) {
      itype = IterType::BroadcastWithStride;
    } else {
      itype = IterType::BroadcastWithoutStride;
    }
  } else if (outer->isBroadcast() || inner->isBroadcast()) {
    itype = IterType::Iteration;
  }

  IterDomain* merged_id = new IterDomain(
      new Int(0),
      merged_id_size->as<Int>(),
      outer->getParallelType(),
      itype,
      outer->isRFactorProduct() || inner->isRFactorProduct());

  new Merge(merged_id, outer, inner);

  return merged_id;
}

std::pair<IterDomain*, IterDomain*> IterDomain::split(
    IterDomain* in,
    Val* factor) {
  TORCH_CHECK(
      in->start()->isZeroInt(),
      "Splitting IterDomains with starting values that aren't 0 is not supported at this time.");

  if (in->getParallelType() != ParallelType::Serial)
    TORCH_CHECK(
        false,
        "Splitting an axis of non-Serial iteration is not supported at this time."
        " Parallelization strategy must be set after calling split.");

  TORCH_CHECK(factor->isAnInt(), "Cannot split by non-integer value ", factor);

  if (factor->getValType() == ValType::Scalar) {
    TORCH_CHECK(
        factor->isConstScalar() ||
            FusionGuard::getCurFusion()->hasInput(factor),
        factor,
        " is not a constant nor an input. It must be one or the other to be used in a split.",
        " If you want a symbolic split based on a thread dimension please use IterDomain::split(IterDomain*, ParallelType);");
  } else if (factor->getValType() == ValType::NamedScalar) {
    TORCH_CHECK(
        factor->as<NamedScalar>()->getParallelDim() != c10::nullopt,
        "Splitting a dimension by a named scalar is only supported on block or grid dimensions but received ",
        factor);
  }

  // outer loop size
  Val* vo = ceilDiv(in->extent(), factor);

  // outer loop IterDomain
  IterDomain* ido = new IterDomain(
      new Int(0),
      vo->as<Int>(),
      in->getParallelType(),
      in->getIterType(),
      in->isRFactorProduct());

  // inner loop IterDomain
  IterDomain* idi = new IterDomain(
      new Int(0),
      factor,
      in->getParallelType(),
      in->getIterType(),
      in->isRFactorProduct());

  new Split(ido, idi, in, factor);
  return {ido, idi};
}

// TODO(kir): review if this is still needed in the Fusion IR
Val* IterDomain::extent() const {
  if (isThread()) {
    if (extent_->getValType() == ValType::Scalar)
      if (extent_->as<Int>()->isConst())
        return extent_;

    return NamedScalar::getParallelDim(getParallelType());
  }
  return extent_;
}

TensorDomain::TensorDomain(
    std::vector<IterDomain*> _domain,
    std::vector<bool> _contiguity)
    : Val(ValType::TensorDomain),
      root_domain_(std::move(_domain)),
      contiguity_(
          _contiguity.empty() ? std::vector<bool>(root_domain_.size(), false)
                              : std::move(_contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == root_domain_.size(),
      "Invalid contiguity information provided, incorrect size. Recieved vector of size ",
      contiguity_.size(),
      " but needed one of size ",
      root_domain_.size());

  domain_ = root_domain_;
  resetDomains();
}

TensorDomain::TensorDomain(
    std::vector<IterDomain*> _root_domain,
    std::vector<IterDomain*> _domain,
    std::vector<bool> _contiguity)
    : Val(ValType::TensorDomain, DataType::Null, false),
      root_domain_(std::move(_root_domain)),
      domain_(std::move(_domain)),
      contiguity_(
          _contiguity.empty() ? std::vector<bool>(root_domain_.size(), false)
                              : std::move(_contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == root_domain_.size(),
      "Invalid contiguity information provided, incorrect size. Recieved vector of size ",
      contiguity_.size(),
      " but needed one of size ",
      root_domain_.size());

  std::vector<Val*> domain_vals(domain_.begin(), domain_.end());
  auto inps = IterVisitor::getInputsTo(domain_vals);

  // Validate that the root domain consists of all inputs to _domain
  // Uncertain if this will hold for RFactor

  std::unordered_set<Val*> root_vals(root_domain_.begin(), root_domain_.end());
  std::for_each(inps.begin(), inps.end(), [root_vals](Val* inp) {
    TORCH_INTERNAL_ASSERT(
        root_vals.find(inp) != root_vals.end(),
        "Invalid tensor domain, ",
        inp,
        " is an input of domain, but it is not found in the root domain.");
  });

  resetDomains();

  name_ = fusion_->registerVal(this);
}

TensorDomain::TensorDomain(
    std::vector<IterDomain*> _root_domain,
    std::vector<IterDomain*> _rfactor_domain,
    std::vector<IterDomain*> _domain,
    std::vector<bool> _contiguity)
    : Val(ValType::TensorDomain, DataType::Null, false),
      root_domain_(std::move(_root_domain)),
      domain_(std::move(_domain)),
      rfactor_domain_(std::move(_rfactor_domain)),
      contiguity_(
          _contiguity.empty() ? std::vector<bool>(root_domain_.size(), false)
                              : std::move(_contiguity)) {
  TORCH_CHECK(
      contiguity_.size() == root_domain_.size(),
      "Invalid contiguity information provided, incorrect size. Recieved vector of size ",
      contiguity_.size(),
      " but needed one of size ",
      root_domain_.size());

  auto inps = IterVisitor::getInputsTo(
      std::vector<Val*>(domain_.begin(), domain_.end()));

  // Validate that the root domain consists of all inputs to _domain
  // Uncertain if this will hold for RFactor

  std::unordered_set<Val*> root_vals(root_domain_.begin(), root_domain_.end());
  std::for_each(inps.begin(), inps.end(), [root_vals](Val* inp) {
    TORCH_INTERNAL_ASSERT(
        root_vals.find(inp) != root_vals.end(),
        "Invalid tensor domain, ",
        inp,
        " is an input of domain, but it is not found in the root domain.");
  });

  inps = IterVisitor::getInputsTo(
      std::vector<Val*>(rfactor_domain_.begin(), rfactor_domain_.end()));
  std::for_each(inps.begin(), inps.end(), [root_vals](Val* inp) {
    TORCH_INTERNAL_ASSERT(
        root_vals.find(inp) != root_vals.end(),
        "Invalid tensor domain, ",
        inp,
        " is an input of the rfactor domain, but it is not found in the root domain.");
  });

  resetDomains();
  name_ = fusion_->registerVal(this);
}

TensorDomain::TensorDomain(const TensorDomain* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner),
      root_domain_(ir_cloner->clone(src->root_domain_)),
      domain_(ir_cloner->clone(src->domain_)),
      no_bcast_domain_(ir_cloner->clone(src->no_bcast_domain_)),
      no_reduction_domain_(ir_cloner->clone(src->no_reduction_domain_)),
      rfactor_domain_(ir_cloner->clone(src->rfactor_domain_)),
      contiguity_(src->contiguity()) {}

bool TensorDomain::operator==(const TensorDomain& other) const {
  // Checks equality of each class field. Should not be necessary to
  // check no_bcast_domain_ and no_reduction_domain_ as they are just
  // derived from domain_.
  return root_domain_ == other.root_domain_ && domain_ == other.domain_ &&
      rfactor_domain_ == other.rfactor_domain_ &&
      contiguity_ == other.contiguity_;
}

bool TensorDomain::sameAs(const TensorDomain* const other) const {
  if (nDims() != other->nDims())
    return false;
  if (getRootDomain().size() != other->getRootDomain().size())
    return false;
  if (getRFactorDomain().size() != other->getRFactorDomain().size())
    return false;

  for (size_t i = 0; i < nDims(); i++)
    if (!(axis(i)->sameAs(other->axis(i))))
      return false;

  for (size_t i = 0; i < getRootDomain().size(); i++)
    if (!(getRootDomain()[i]->sameAs(other->getRootDomain()[i])))
      return false;

  for (size_t i = 0; i < getRFactorDomain().size(); i++)
    if (!(getRFactorDomain()[i]->sameAs(other->getRFactorDomain()[i])))
      return false;

  return true;
}

bool TensorDomain::sameAs(
    const std::vector<IterDomain*>& lhs,
    const std::vector<IterDomain*>& rhs) {
  if (lhs.size() != rhs.size())
    return false;
  size_t i = 0;
  for (auto td_lhs : lhs) {
    if (!td_lhs->sameAs(rhs[i++]))
      return false;
  }
  return true;
}

bool TensorDomain::hasReduction() const {
  return no_reduction_domain_.size() != domain_.size();
}

bool TensorDomain::hasBlockReduction() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->isReduction() && id->isThreadDim();
  });
}

bool TensorDomain::hasGridReduction() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->isReduction() && id->isBlockDim();
  });
}

bool TensorDomain::hasBlockBroadcast() const {
  return std::any_of(domain_.begin(), domain_.end(), [](IterDomain* id) {
    return id->isBroadcast() && id->isThreadDim();
  });
}

bool TensorDomain::hasBroadcast() const {
  return no_bcast_domain_.size() != domain_.size();
}

bool TensorDomain::hasRFactor() const {
  return !rfactor_domain_.empty();
}

c10::optional<unsigned int> TensorDomain::getReductionAxis() const {
  auto it = std::find_if(domain_.begin(), domain_.end(), [](const auto& id) {
    return id->isReduction();
  });
  if (it == domain_.end()) {
    return c10::optional<unsigned int>();
  } else {
    return c10::optional<unsigned int>(std::distance(domain_.begin(), it));
  }
}

// i here is int, as we want to accept negative value and ::size_type can be a
// uint.
IterDomain* TensorDomain::axis(int i) const {
  TORCH_INTERNAL_ASSERT(
      nDims() > 0, "Tried to access an axis in a 0-dim domain");
  if (i < 0)
    i += nDims();
  TORCH_CHECK(
      i >= 0 && (unsigned int)i < nDims(),
      "Tried to access axis ",
      i,
      " in domain ",
      this);
  return domain_[i];
}

size_t TensorDomain::posOf(IterDomain* id) const {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to find an axis in a 0-dim domain");
  size_t i = 0;
  while (i < domain_.size()) {
    if (domain_[i] == id)
      return i;
    i++;
  }
  TORCH_CHECK(false, "Provided id is not part of this domain.");
}

void TensorDomain::split(int axis_, Val* factor) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do split on a 0-dim domain");
  if (axis_ < 0)
    axis_ += nDims();

  TORCH_INTERNAL_ASSERT(
      axis_ >= 0 && (unsigned int)axis_ < nDims(),
      "Tried to split on axis outside TensorDomain's range.");

  IterDomain* id = axis(axis_);
  auto split_ids = IterDomain::split(id, factor);
  domain_.erase(domain_.begin() + axis_);
  domain_.insert(domain_.begin() + axis_, split_ids.second);
  domain_.insert(domain_.begin() + axis_, split_ids.first);
  resetDomains();
}

// Merge "axis" and "axis+1" into 1 dimension
void TensorDomain::merge(int axis_o, int axis_i) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to do merge on a 0-dim domain");
  if (axis_o < 0)
    axis_o += nDims();

  if (axis_i < 0)
    axis_i += nDims();

  TORCH_CHECK(
      axis_o >= 0 && (unsigned int)axis_o < nDims() && axis_i >= 0 &&
          (unsigned int)axis_i < nDims(),
      "Invalid merge detected, either one or both axes are outside of TensorView's range.");

  TORCH_CHECK(
      axis_o != axis_i,
      "Invalid merge detected, axes provided are the same axis.");

  if (axis_o > axis_i) {
    auto tmp = axis_i;
    axis_i = axis_o;
    axis_o = tmp;
  }

  IterDomain* first = axis(axis_o);
  IterDomain* second = axis(axis_i);

  IterDomain* merged_id = IterDomain::merge(first, second);

  domain_.erase(domain_.begin() + axis_i);
  domain_.erase(domain_.begin() + axis_o);
  domain_.insert(domain_.begin() + axis_o, merged_id);
  resetDomains();
}

// Reorder axes according to map[old_pos] = new_pos
void TensorDomain::reorder(const std::unordered_map<int, int>& old2new_) {
  TORCH_INTERNAL_ASSERT(
      !(nDims() == 0 && old2new_.size() > 0),
      "Tried to reorder a 0-dim domain");
  domain_ = orderedAs(domain_, old2new_);
  resetDomains();
}

std::vector<IterDomain*> TensorDomain::orderedAs(
    const std::vector<IterDomain*>& dom,
    const std::unordered_map<int, int>& old2new_) {
  TORCH_INTERNAL_ASSERT(
      !(dom.size() == 0 && old2new_.size() > 0),
      "Tried to reorder a 0-dim domain");

  // Eventhough these checks are already in TensorView, we want to redo them as
  // we can enter this function from other places, not through TensorView

  // adjust based on negative values (any negative values gets nDims added to
  // it)
  std::unordered_map<int, int> old2new;
  auto ndims = dom.size();
  std::transform(
      old2new_.begin(),
      old2new_.end(),
      std::inserter(old2new, old2new.begin()),
      [ndims](std::unordered_map<int, int>::value_type entry) {
        return std::unordered_map<int, int>::value_type({
            entry.first < 0 ? entry.first + ndims : entry.first,
            entry.second < 0 ? entry.second + ndims : entry.second,
        });
      });

  // Check if any adjusted values are < 0, or >= nDims, which are invalid

  TORCH_CHECK(
      std::none_of(
          old2new.begin(),
          old2new.end(),
          [ndims](std::unordered_map<int, int>::value_type entry) {
            return entry.first < 0 || (unsigned int)entry.first >= ndims ||
                entry.second < 0 || (unsigned int)entry.second >= ndims;
          }),
      "Reorder axes are not within the number of dimensions of the provided domain.");

  // Going to use sets, to see if any duplicate values are in the map.

  std::set<int> old_pos_set;
  std::transform(
      old2new.begin(),
      old2new.end(),
      std::inserter(old_pos_set, old_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.first;
      });

  std::set<int> new_pos_set;
  std::transform(
      old2new.begin(),
      old2new.end(),
      std::inserter(new_pos_set, new_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.second;
      });

  // Error out if duplicate values are found.
  TORCH_CHECK(
      old_pos_set.size() == old2new.size() &&
          new_pos_set.size() == old2new.size(),
      "Duplicate entries in transformation map sent to TensorView reorder.");

  // END VALIDATION CHECKS

  std::vector<int> new2old(ndims, -1);

  // Go through each old and new position, make sure they're within [0, ndims)
  for (std::pair<int, int> elem : old2new) {
    int old_pos = elem.first;
    int new_pos = elem.second;
    new2old[new_pos] = old_pos;
  }

  // old_positions that already have a new position
  std::set<int> old_positions(new2old.begin(), new2old.end());
  old_positions.erase(-1);

  // All available new positions
  std::set<int> all_positions;
  for (decltype(ndims) i{0}; i < ndims; i++)
    all_positions.insert(i);

  // Check what positions haven't been specified.
  std::set<int> positions_left;
  std::set_difference(
      all_positions.begin(),
      all_positions.end(),
      old_positions.begin(),
      old_positions.end(),
      std::inserter(positions_left, positions_left.end()));

  // Fill in positions that weren't specified, in relative order,
  // in empty spots in the set of new positions.
  // new2old[new_position] = old_position
  auto it = positions_left.begin(); // old positions left
  std::transform(
      new2old.begin(), new2old.end(), new2old.begin(), [&it](int i) -> int {
        return i == -1 ? *it++ : i;
      });

  std::vector<IterDomain*> reordered_domain;
  std::transform(
      new2old.begin(),
      new2old.end(),
      std::back_inserter(reordered_domain),
      [dom](int i) -> IterDomain* { return dom[i]; });

  return reordered_domain;
}

std::vector<IterDomain*> TensorDomain::noReductions(
    const std::vector<IterDomain*>& td) {
  size_t size_out = 0;
  for (auto id : td)
    if (!id->isReduction())
      size_out++;
  std::vector<IterDomain*> noReductionDomain(size_out);

  int it = 0;
  for (auto id : td)
    if (!id->isReduction())
      noReductionDomain[it++] = id;

  return noReductionDomain;
}

std::vector<IterDomain*> TensorDomain::noBroadcasts(
    const std::vector<IterDomain*>& td) {
  size_t size_out = 0;
  for (auto id : td)
    if (!id->isBroadcast())
      size_out++;
  std::vector<IterDomain*> noBroadcastDomain(size_out);

  int it = 0;
  for (auto id : td)
    if (!id->isBroadcast())
      noBroadcastDomain[it++] = id;

  return noBroadcastDomain;
}

bool TensorDomain::hasBroadcast(const std::vector<IterDomain*>& td) {
  for (auto id : td)
    if (id->isBroadcast())
      return true;
  return false;
}
bool TensorDomain::hasReduction(const std::vector<IterDomain*>& td) {
  for (auto id : td)
    if (id->isReduction())
      return true;
  return false;
}

std::vector<std::pair<int, int>> TensorDomain::mapDomainPandC(
    const std::vector<IterDomain*>& producer,
    const std::vector<IterDomain*>& consumer) {
  std::vector<std::pair<int, int>> dom_map;

  size_t itc = 0, itp = 0;
  while (itc < consumer.size() && itp < producer.size()) {
    if (consumer[itc]->isBroadcast() && !producer[itp]->isBroadcast()) {
      itc++;
      continue;
    }
    if (producer[itp]->isReduction()) {
      itp++;
      continue;
    }

    dom_map.emplace_back(std::make_pair(itp, itc));
    itc++;
    itp++;
  }
  return dom_map;
}

std::vector<std::pair<IterDomain*, IterDomain*>> TensorDomain::mapRootPandC(
    const TensorDomain* producer,
    const TensorDomain* consumer) {
  auto consumer_root = consumer->getRootDomain();
  auto producer_root = producer->getMaybeRFactorDomain();
  std::vector<std::pair<IterDomain*, IterDomain*>> root_id_map;
  for (const auto& m : mapDomainPandC(producer_root, consumer_root)) {
    auto producer_axis = producer_root[m.first];
    auto consumer_axis = consumer_root[m.second];
    root_id_map.emplace_back(std::make_pair(producer_axis, consumer_axis));
  }
  return root_id_map;
}

std::unordered_map<IterDomain*, IterDomain*> TensorDomain::mapRootCtoP(
    const TensorDomain* consumer,
    const TensorDomain* producer,
    const std::unordered_set<IterDomain*>& consumer_root_dims_to_map) {
  std::unordered_map<IterDomain*, IterDomain*> root_id_map;
  for (const auto& kv : mapRootPandC(producer, consumer)) {
    auto producer_axis = kv.first;
    auto consumer_axis = kv.second;
    if (consumer_root_dims_to_map.find(consumer_axis) !=
        consumer_root_dims_to_map.end()) {
      root_id_map[consumer_axis] = producer_axis;
    }
  }
  return root_id_map;
}

std::unordered_map<IterDomain*, IterDomain*> TensorDomain::mapRootPtoC(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const std::unordered_set<IterDomain*>& producer_maybe_rfactor_dims_to_map) {
  std::unordered_map<IterDomain*, IterDomain*> root_id_map;
  for (const auto& kv : mapRootPandC(producer, consumer)) {
    auto producer_axis = kv.first;
    auto consumer_axis = kv.second;
    if (producer_maybe_rfactor_dims_to_map.find(producer_axis) !=
        producer_maybe_rfactor_dims_to_map.end()) {
      root_id_map[producer_axis] = consumer_axis;
    }
  }
  return root_id_map;
}

// pair is in order where second is the consumer of first
std::pair<TensorDomain*, TensorDomain*> TensorDomain::rFactor(
    const std::vector<int>& axes_) {
  TORCH_INTERNAL_ASSERT(nDims() > 0, "Tried to rFactor a 0-dim domain");

  std::vector<int> axes(axes_.size());

  auto ndims = nDims();
  std::transform(axes_.begin(), axes_.end(), axes.begin(), [ndims](int i) {
    return i < 0 ? i + ndims : i;
  });

  TORCH_CHECK(
      std::none_of(
          axes.begin(),
          axes.end(),
          [ndims](int i) { return i < 0 || (unsigned int)i >= ndims; }),
      "RFactor axes less than 0 or >= ndims.");

  // We might be able to lift this constraint in some instances, but needs more
  // investigation.
  TORCH_CHECK(
      !hasRFactor(), "Cannot call rfactor on the same tensor domain twice.");

  std::unordered_set<int> axes_set(axes.begin(), axes.end());

  bool rfactor_found = false;
  bool reduction_found = false;
  for (decltype(nDims()) i{0}; i < nDims(); i++) {
    if (axis(i)->isReduction()) {
      if (axes_set.find(i) != axes_set.end()) {
        rfactor_found = true;
      } else {
        reduction_found = true;
      }
    }
  }

  TORCH_CHECK(
      rfactor_found && reduction_found,
      "Invalid rfactor found, rfactor must be provided at least one reduction axis, but not all reduction axes.");

  return std::pair<TensorDomain*, TensorDomain*>{
      TransformRFactor::runReplay(this, axes),
      TransformRFactor::runReplay2(this, axes)};
}

namespace {
class BroadcastMapping: public BackwardVisitor {
 public:
  using BackwardVisitor::handle;

  BroadcastMapping(Fusion* fusion) {
    traverseFrom(fusion, fusion->outputs(), false);
  }

  void handle(BinaryOp* bop) override {
    if (!ir_utils::isTVOp(bop)) return;
    for (const auto input: bop->inputs()) {
      if (input->getValType().value() != ValType::TensorView) continue;
      const TensorView* input_tv = input->as<TensorView>();
      const auto& input_root = input_tv->getRootDomain();
      const auto& output_root = bop->output(0)->as<TensorView>()->getRootDomain();
      for (size_t i = 0; i < input_root.size(); ++i) {
        if (!input_root.at(i)->isBroadcast()) continue;
        if (output_root.at(i)->isBroadcast()) {
          continue;
        }
        std::cerr << "Concrete ID found: "
                  << input_root.at(i) << " -> " << output_root.at(i) << std::endl;
        map_.insert({input_root.at(i), output_root.at(i)});
      }
    }
  }

  // Mapping from a broadcast domain to its concrete domain
  std::unordered_map<const IterDomain*, const IterDomain*> map_;

  static const IterDomain* getConcreteDomain(const IterDomain* bcast_dom) {
    BroadcastMapping bcast_mapping(bcast_dom->fusion());
    auto& m = bcast_mapping.map_;

    std::vector<Val*> from_vals;
    from_vals.push_back(const_cast<IterDomain*>(bcast_dom));
    auto dom_exprs = ExprSort::getExprs(bcast_dom->fusion(), from_vals);

    for (auto expr: dom_exprs) {
      if (expr->getExprType() == ExprType::Split) {
        Split* split = expr->as<Split>();
        if (!split->in()->isBroadcast()) {
          continue;
        }
        TORCH_INTERNAL_ASSERT(m.find(split->in()) != bcast_mapping.map_.end());
        IterDomain* concrete_id = const_cast<IterDomain*>(m.find(split->in())->second);
        auto split_ids = IterDomain::split(concrete_id, split->factor());
        m.insert({split->outer(), split_ids.first});
        m.insert({split->inner(), split_ids.second});
      } else if (expr->getExprType() == ExprType::Merge) {
        TORCH_INTERNAL_ASSERT(false, "TODO");
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unexpected expr: ", expr);
      }
    }

    auto it = bcast_mapping.map_.find(bcast_dom);
    std::stringstream ss;
    ss << "Concrete domain not found for "
       << bcast_dom;
    TORCH_INTERNAL_ASSERT(it != bcast_mapping.map_.end(), ss.str());
    std::cerr << "Concrete domain for " << bcast_dom
              << ": " << it->second << std::endl;
    return it->second;
  }
};

class MatchingIterDomainSearch: public IterVisitor {
 public:
  using IterVisitor::handle;

  void handle(BinaryOp* bop) override {
    if (!ir_utils::isTVOp(bop)) return;
    const TensorView* base_tv = nullptr;
    for (const auto input: bop->inputs()) {
      if (input->getValType().value() != ValType::TensorView) continue;
      if (base_tv == nullptr) {
        base_tv = input->as<TensorView>();
        continue;
      }
      const TensorView* tv = input->as<TensorView>();
      addToEquivalentSets(base_tv, tv);
    }
    const TensorView* out_tv = bop->output(0)->as<TensorView>();
    addToEquivalentSets(base_tv, out_tv);
  }

  void addToEquivalentSets(const TensorView* tv_x,
                           const TensorView* tv_y) {
    TORCH_INTERNAL_ASSERT(tv_x != nullptr);
    TORCH_INTERNAL_ASSERT(tv_y != nullptr);
    if (tv_x == tv_y) return;
    const auto& root_x = tv_x->getRootDomain();
    const auto& root_y = tv_y->getRootDomain();
    TORCH_INTERNAL_ASSERT(root_x.size() == root_y.size());
    for (size_t i = 0; i < root_x.size(); ++i) {
      addToEquivalentSets(root_x[i], root_y[i]);
    }
  }

  void addToEquivalentSets(const IterDomain *id_x,
                           const IterDomain *id_y) {
    bool dbg = false;
    const auto x_key = id_x->extent();
    const auto y_key = id_y->extent();
    if (x_key == y_key) {
      // same pointer; nothing to do
      return;
    }
    auto it_x = equivalent_sets_.find(x_key);
    auto it_y = equivalent_sets_.find(y_key);
    if (dbg) std::cerr << "Equivalent IDs: " << id_x << " == " << id_y << std::endl;
    if (it_x != equivalent_sets_.end() &&
        it_y != equivalent_sets_.end()) {
      if (dbg) std::cerr << "joining two sets\n";
      // both already exist; join them
      if (it_x->second == it_y->second) {
        if (dbg) std::cerr << "already pointing to the same set" << std::endl;
        return;
      }
      auto x_set = it_x->second;
      auto y_set = it_y->second;
      for (const auto id_in_y: *y_set) {
        if (dbg) std::cerr << "Adding " << id_in_y << " to set for " << id_x << std::endl;
        x_set->insert(id_in_y);
        equivalent_sets_[id_in_y] = x_set;
      }
    } else if (it_x != equivalent_sets_.end()) {
      if (dbg) std::cerr << "Adding " << id_y << " to the existing set for " << id_x << std::endl;
      // id_y is a new ID
      auto x_set = it_x->second;
      x_set->insert(y_key);
      equivalent_sets_.insert({y_key, x_set});
    } else if (it_y != equivalent_sets_.end()) {
      if (dbg) std::cerr << "Adding " << id_x << " to the existing set for " << id_y << std::endl;
      // id_x is a new ID
      auto y_set = it_y->second;
      y_set->insert(x_key);
      equivalent_sets_.insert({x_key, y_set});
    } else {
      if (dbg) std::cerr << "Creating a new equiv set\n";
      // both are new
      auto id_set = std::make_shared<std::unordered_set<const Val*>>();
      id_set->insert(x_key);
      id_set->insert(y_key);
      equivalent_sets_.insert({x_key, id_set});
      equivalent_sets_.insert({y_key, id_set});
    }
  }

  // Mapping from a broadcast domain to its concrete domain
  std::unordered_map<const Val*,
                     std::shared_ptr<std::unordered_set<const Val*>>> equivalent_sets_;

  static bool isEquivalent(const Val* id_x,
                           const Val* id_y) {
    bool dbg = false;
    if (dbg) std::cerr << "isEquivalent? " << id_x << " and " << id_y << std::endl;
    Fusion* fusion = id_x->fusion();
    MatchingIterDomainSearch search;
    search.traverseFrom(fusion, fusion->outputs(), false);

    auto it = search.equivalent_sets_.find(id_x);
    if (it == search.equivalent_sets_.end()) {
      // id_x not detected at all
      if (dbg) std::cerr << id_x << " not found\n";
      return false;
    }
    auto equivalent_set = it->second;
    bool result = equivalent_set->find(id_y) != equivalent_set->end();
    if (!result) {
      if (dbg) std::cerr << id_y << " not found in the equivalent set\n";
    }
    return result;
  }
};

bool sameAs(Val* v1, Val* v2);

Val* omitMul1(Expr* e) {
  if (e->getExprType() == ExprType::BinaryOp) {
    auto bop = e->as<BinaryOp>();
    if (bop->getBinaryOpType() == BinaryOpType::Mul) {
      if (bop->lhs()->isOneInt()) {
        return bop->rhs();
      } else if (bop->rhs()->isOneInt()) {
        return bop->lhs();
      }
    }
  }
  return nullptr;
}

bool sameAs(const Expr* e1, const Expr* e2) {
  //std::cerr << "Checking expr equivalence of " << e1 << " and " << e2 << std::endl;
  if (e1 == nullptr || e2 == nullptr) return false;

  if (e1->inputs().size() != e2->inputs().size() ||
      e1->outputs().size() != e2->outputs().size() ||
      e1->getExprType() != e2->getExprType()) {
    return false;
  }
  for (size_t i = 0; i < e1->inputs().size(); ++i) {
    if (!sameAs(e1->input(i), e2->input(i))) {
      return false;
    }
  }
  return true;
}

bool sameAs(Val* v1, Val* v2) {
  if (v1 == nullptr || v2 == nullptr) return false;

  // TODO (CD): This is a temporary unsafe workaround. If a value is 1,
  // assume it originates from a broadcast dimension and matches with
  // any other given dimnsion. This is cheating and must be fixed.
  if (v1->isOneInt() || v2->isOneInt()) {
    //TORCH_INTERNAL_ASSERT(false, "should never happen");
    return true;
  }

  if (v1->getOrigin() && v2->getOrigin()) {
    //TORCH_INTERNAL_ASSERT(false, "should never happen");
    return sameAs(v1->getOrigin(), v2->getOrigin());
  } else if (v1->getOrigin()) {
    //TORCH_INTERNAL_ASSERT(false, "should never happen",
    //v1, ", ", v2);
    auto v = omitMul1(v1->getOrigin());
    if (v) {
      return sameAs(v, v2);
    }
  } else if (v2->getOrigin()) {
    //TORCH_INTERNAL_ASSERT(false, "should never happen");
    auto v = omitMul1(v2->getOrigin());
    if (v) {
      return sameAs(v1, v);
    }
  }

  if (ScalarCheck::sameAs(v1, v2)) {
    return true;
  }

  return MatchingIterDomainSearch::isEquivalent(v1, v2);
}

} // namespace

bool ComputeDomain::sameAxes(const IterDomain* id1, const IterDomain* id2) {
  bool dbg = false;
  std::stringstream debug_msg;
  debug_msg << "Checking ID equivalence of " << id1 << " and " << id2;

  if (id1 == id2) {
    debug_msg << " -> true";
    if (dbg) std::cerr << debug_msg.str() << std::endl;
    return true;
  }

  if (id1->isBroadcast()) {
    id1 = BroadcastMapping::getConcreteDomain(id1);
  }
  if (id2->isBroadcast()) {
    id2 = BroadcastMapping::getConcreteDomain(id2);
  }

  bool result = sameAs(id1->start(), id2->start()) &&
      sameAs(id1->rawExtent(), id2->rawExtent());

  if (dbg) {
    std::cerr << debug_msg.str()
              << " -> " << result << std::endl;
  }
  return result;
}

ComputeDomain::ComputeDomain(const TensorDomain* td):
    td_(td->domain()),
    axes_(td->domain().begin(),
          td->domain().end()),
    td_map_(td->nDims()) {
  std::iota(td_map_.begin(), td_map_.end(), 0);
}

std::unordered_map<IterDomain*, IterDomain*> ComputeDomain::mapRootDomain(
    const std::vector<IterDomain*>& root_domain,
    const std::unordered_set<IterDomain*>& compute_root_ids) const {
  auto root_ids = compute_root_ids;
  std::unordered_map<IterDomain*, IterDomain*> root_id_map;
  for (const auto& id: root_domain) {
    auto it = std::find_if(
        root_ids.begin(), root_ids.end(),
        [id](const IterDomain* d) {
          return ComputeDomain::sameAxes(id, d);
        });
    if (it == root_ids.end()) {
      continue;
    }
    root_id_map.emplace(*it, id);
    root_ids.erase(it);
  }
  return root_id_map;
}

void ComputeDomain::split(const TensorDomain* new_td, int axis_idx) {
  std::cerr << "Splitting: " << *this
            << " at " << axis_idx
            << std::endl;
  auto cd_axis_idx = getComputeDomainAxisIndex(axis_idx);
  TORCH_INTERNAL_ASSERT(cd_axis_idx >= getComputeAtPos());
  td_ = new_td->domain();
  auto new_id_left = td_.at(axis_idx);
  auto new_id_right = td_.at(axis_idx+1);
  setAxis(cd_axis_idx, new_id_left);
  insertAxis(cd_axis_idx + 1, new_id_right, axis_idx + 1);
  std::cerr << "Split completed: " << *this
            << std::endl;
}

void ComputeDomain::merge(const TensorDomain* new_td, int axis_o, int axis_i) {
  std::cerr << "Merging " << axis_o << " and " << axis_i
            << " of " << *this << std::endl;
  if (axis_o > axis_i) {
    std::swap(axis_o, axis_i);
  }
  auto cd_axis_o = getComputeDomainAxisIndex(axis_o);
  auto cd_axis_i = getComputeDomainAxisIndex(axis_i);
  TORCH_INTERNAL_ASSERT(cd_axis_o >= getComputeAtPos());
  TORCH_INTERNAL_ASSERT(cd_axis_i >= getComputeAtPos());
  td_ = new_td->domain();
  setAxis(cd_axis_o, new_td->domain().at(axis_o));
  eraseAxis(cd_axis_i);
  std::cerr << "Merge completed: " << *this
            << std::endl;
}

// Transform this compute domain so that it is computed under the
// target domain. TensorDomain is assumed to be already transformed by
// replayPasC or replayCasP.
void ComputeDomain::computeAt(const TensorDomain* td,
                              int this_pos,
                              const ComputeDomain* target,
                              int target_pos,
                              const std::vector<size_t>& td2cd_map) {
  std::cerr << "computeAt: " << *target
            << " at " << target_pos
            << ", td: " << td << " at " << this_pos << std::endl;
  // reset the current status
  td_ = td->domain();
  axes_.clear();
  td_map_.clear();
  td_map_.resize(td_.size());
  pos_ = 0;

  normalizeComputeAtPos(target_pos, target->nDims());
  normalizeComputeAtPos(this_pos, td->nDims());

  TORCH_INTERNAL_ASSERT(this_pos <= target_pos);
  TORCH_INTERNAL_ASSERT((size_t)this_pos <= td->nDims());
  TORCH_INTERNAL_ASSERT((size_t)target_pos <= target->nDims());

#if 0
  std::vector<IterDomain*> target_axes{target->axes().begin(),
                                       target->axes().begin() + target_pos};
  auto target_axes_it = target_axes.begin();
  for (int i = 0; i < this_pos; ++i) {
    IterDomain* this_axis = td_.at(i);
    target_axes_it = std::find_if(
        target_axes_it, target_axes.end(),
        [this_axis](const IterDomain* ca) {
          return ComputeDomain::sameAxes(this_axis, ca);
        });
    if (target_axes_it == target_axes.end()) {
      std::cerr << "Axis not found: " << this_axis
                << " of " << td << std::endl;
      std::cerr << "Target axes: " << target_axes
                << ", target domain: " << *target << std::endl;
      TORCH_INTERNAL_ASSERT(false);
    }
    td_map_.at(i) = std::distance(target_axes.begin(), target_axes_it);
    // Search the rest of td in the remaining target axes
    ++target_axes_it;
  }
#else
  TORCH_INTERNAL_ASSERT(td2cd_map.size() == this_pos);
  std::copy(td2cd_map.begin(), td2cd_map.end(),
            td_map_.begin());
#endif

  //auto num_shared_axes = std::distance(target_axes.begin(),
  //target_axes_it);
  auto num_shared_axes = target_pos;

  // Copy from target domain
  std::copy(target->axes().begin(),
            target->axes().begin() + num_shared_axes,
            std::back_inserter(axes_));

  pos_ = num_shared_axes;

  // Set up own compute axes
  std::copy(td_.begin() + this_pos,
            td_.end(),
            std::back_inserter(axes_));

  std::iota(td_map_.begin() + this_pos, td_map_.end(),
            num_shared_axes);

  updateDependents();

  fixupPosition();
  std::cerr << "computeAt done: " << *this << std::endl;
}

void ComputeDomain::fixupPosition() {
  for (size_t i = pos_; i > 0; --i) {
    IterDomain* cd_left = axis(i-1);
    if (!isComputeDomainAxisUsed(i-1)) break;
    IterDomain* td_left = td_.at(getTensorDomainAxisIndex(i-1));
    if (cd_left != td_left) break;
    --pos_;
  }
}

void ComputeDomain::setAxis(size_t cd_axis, IterDomain* id) {
  TORCH_INTERNAL_ASSERT(cd_axis < axes_.size(),
                        "Out of range error. Attempting to access axis at offset ",
                        cd_axis, " of size-", axes_.size(),
                        " compute domain.");
  axes_[cd_axis] = id;
}

void ComputeDomain::insertAxis(size_t cd_axis, IterDomain* cd_id, size_t td_axis) {
  TORCH_INTERNAL_ASSERT(cd_axis <= axes_.size(),
                        "Out of range error. Attempting to insert axis at offset ",
                        cd_axis, " of size-", axes_.size(),
                        " compute domain.");
  axes_.insert(axes_.begin() + cd_axis, cd_id);
  // shift right and increment
  td_map_.resize(td_map_.size() + 1);
  for (auto i = td_axis; i < td_map_.size() - 1; ++i) {
    td_map_.at(i + 1) = td_map_.at(i) + 1;
  }
  td_map_.at(td_axis) = cd_axis;
  sanityCheck();
}

void ComputeDomain::eraseAxis(size_t cd_axis) {
  TORCH_INTERNAL_ASSERT(cd_axis < axes_.size(),
                        "Out of range error. Attempting to erase axis at offset ",
                        cd_axis, " of size-", axes_.size(),
                        " compute domain.");
  auto td_axis = getTensorDomainAxisIndex(cd_axis);
  axes_.erase(axes_.begin() + cd_axis);
  // shift and decrement
  for (auto i = td_axis; i < td_map_.size() - 1; ++i) {
    td_map_[i] = td_map_[i+1] - 1;
  }
  td_map_.resize(td_map_.size() - 1);
  sanityCheck();
}

void ComputeDomain::sanityCheck() const {
  auto td_ndims = td_.size();
  TORCH_INTERNAL_ASSERT(td_ndims == td_map_.size());
  TORCH_INTERNAL_ASSERT(td_ndims <= nDims());
  if (td_ndims > 0) {
    // Make sure td_map is in an increasing order
    auto prev_cd_pos = td_map_[0];
    for (size_t i = 1; i < td_map_.size(); ++i) {
      auto cd_pos = td_map_[i];
      TORCH_INTERNAL_ASSERT(cd_pos > prev_cd_pos);
      prev_cd_pos = cd_pos;
    }
  }
}

void ComputeDomain::registerAsDependent(ComputeDomain* target) {
  target->registerDependent(this, getComputeAtPos());
}

bool ComputeDomain::isDependent(const ComputeDomain* cd) const {
  for (const auto& dep: dependents_) {
    if (dep.second == cd || dep.second->isDependent(cd)) {
      return true;
    }
  }
  return false;
}

void ComputeDomain::updateDependents() {
  for (auto& dep: dependents_) {
    const auto num_axes = dep.first;
    auto affected_cd = dep.second;
    std::copy(axes().begin(), axes().begin() + num_axes,
              affected_cd->axes_.begin());
    affected_cd->updateDependents();
  }
}

void ComputeDomain::registerDependent(ComputeDomain* dependent, size_t pos) {
  // don't create a dependency loop as it should not make any sense
  if (dependent->isDependent(this)) {
    return;
  }
  dependents_.push_back({pos, dependent});
}

std::unordered_set<IterDomain*> ComputeDomain::getRootDomain() const {
  auto root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(axes().begin(), axes().end()));
  std::unordered_set<IterDomain*> root_domain =
      {ir_utils::filterByType<IterDomain>(root_vals).begin(),
       ir_utils::filterByType<IterDomain>(root_vals).end()};
  return root_domain;
}

std::ostream& ComputeDomain::print(std::ostream& os) const {
  os << "compute_domain(";
  auto map_it = td_map_.begin();
  for (size_t i = 0; i < pos_; ++i) {
    os << " " << axis(i);
    if (map_it != td_map_.end() && *map_it == i) {
      os << "@" << std::distance(td_map_.begin(), map_it);
      ++map_it;
    }
  }
  os << " |";
  for (size_t i = pos_; i < nDims(); ++i) {
    os << " " << axis(i);
    if (map_it != td_map_.end() && *map_it == i) {
      os << "@" << std::distance(td_map_.begin(), map_it);
      ++map_it;
    }
  }
  os << ", " << td_map_;
  os << " )";
  return os;
}

std::ostream& operator<<(std::ostream& os, const ComputeDomain& cd) {
  return cd.print(os);
}

Split::Split(
    IterDomain* _outer,
    IterDomain* _inner,
    IterDomain* _in,
    Val* _factor)
    : Expr(ExprType::Split),
      outer_{_outer},
      inner_{_inner},
      in_{_in},
      factor_{_factor} {
  TORCH_INTERNAL_ASSERT(
      factor_->isAnInt(),
      "Attempted to create a Split node with a non-integer factor.");
  addOutput(_outer);
  addOutput(_inner);
  addInput(_in);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Split::Split(const Split* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      outer_(ir_cloner->clone(src->outer_)),
      inner_(ir_cloner->clone(src->inner_)),
      in_(ir_cloner->clone(src->in_)),
      factor_(ir_cloner->clone(src->factor_)) {}

bool Split::sameAs(const Split* const other) const {
  return (
      outer()->sameAs(other->outer()) && inner()->sameAs(other->inner()) &&
      in()->sameAs(other->in()) && factor()->sameAs(other->factor()));
}

Merge::Merge(IterDomain* _out, IterDomain* _outer, IterDomain* _inner)
    : Expr(ExprType::Merge), out_{_out}, outer_{_outer}, inner_{_inner} {
  addOutput(_out);
  addInput(_outer);
  addInput(_inner);
  name_ = FusionGuard::getCurFusion()->registerExpr(this);
}

Merge::Merge(const Merge* src, IrCloner* ir_cloner)
    : Expr(src, ir_cloner),
      out_(ir_cloner->clone(src->out_)),
      outer_(ir_cloner->clone(src->outer_)),
      inner_(ir_cloner->clone(src->inner_)) {}

bool Merge::sameAs(const Merge* const other) const {
  return (
      out()->sameAs(other->out()) && outer()->sameAs(other->outer()) &&
      inner()->sameAs(other->inner()));
}

NamedScalar::NamedScalar(const NamedScalar* src, IrCloner* ir_cloner)
    : Val(src, ir_cloner), name_(src->name_) {}

NamedScalar* NamedScalar::getParallelDim(ParallelType p_type) {
  std::string parallel_dim = stringifyThreadSize(p_type);
  return new NamedScalar(parallel_dim, DataType::Int);
}

NamedScalar* NamedScalar::getParallelIndex(ParallelType p_type) {
  std::string parallel_ind = stringifyThread(p_type);
  return new NamedScalar(parallel_ind, DataType::Int);
}

c10::optional<ParallelType> NamedScalar::getParallelDim() const {
  if (stringifyThreadSize(ParallelType::TIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDx);
  } else if (stringifyThreadSize(ParallelType::TIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDy);
  } else if (stringifyThreadSize(ParallelType::TIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDz);
  } else if (stringifyThreadSize(ParallelType::BIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDx);
  } else if (stringifyThreadSize(ParallelType::BIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDy);
  } else if (stringifyThreadSize(ParallelType::BIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDz);
  }
  return c10::nullopt;
}

c10::optional<ParallelType> NamedScalar::getParallelIndex() const {
  if (stringifyThread(ParallelType::TIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDx);
  } else if (stringifyThread(ParallelType::TIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDy);
  } else if (stringifyThread(ParallelType::TIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::TIDz);
  } else if (stringifyThread(ParallelType::BIDx).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDx);
  } else if (stringifyThread(ParallelType::BIDy).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDy);
  } else if (stringifyThread(ParallelType::BIDz).compare(name()) == 0) {
    return c10::optional<ParallelType>(ParallelType::BIDz);
  }
  return c10::nullopt;
}

} // namespace fuser
} // namespace jit
} // namespace torch
