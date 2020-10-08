#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_base_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>

/*
 * Nodes in here should generally not be used by users. They should be behind
 * the scenes and users shouldn't have to be aware of what they do to use the
 * code generator.
 */

namespace torch {
namespace jit {
namespace fuser {

// Returns true if both v1 and v2 are scalars, are the same type of scalars, and
// dispatches to the inherited Val type's `->sameAs` call. e.g. if both vals are
// `Int` will dispatch to v1->as<Int>()->sameAs(v2.as<Int>())
bool areEqualScalars(Val* v1, Val* v2);

/*
 * TODO: improve implementation bool IterDomain::sameAs(const IterDomain*) const
 * TODO: Add testing of sameAs functions for these nodes
 */

/*
 * A specialization for Unary operations. Unary operations take in a single
 * input and produce a single output. Examples include:
 *   1) Casting operation i.e. float(a_val)
 *   2) Negation i.e. val * -1
 *   3) Reduction across a dimension i.e. val.sum(axis=2)
 *   4) split/merge
 */
class TORCH_CUDA_API UnaryOp : public Expr {
 public:
  ~UnaryOp() = default;
  UnaryOp(UnaryOpType _type, Val* _out, Val* _in);

  UnaryOp(const UnaryOp* src, IrCloner* ir_cloner);

  UnaryOp(const UnaryOp& other) = delete;
  UnaryOp& operator=(const UnaryOp& other) = delete;

  UnaryOp(UnaryOp&& other) = delete;
  UnaryOp& operator=(UnaryOp&& other) = delete;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  UnaryOpType getUnaryOpType() const {
    return unary_op_type_;
  }

  bool sameAs(const UnaryOp* const other) const;

 private:
  const UnaryOpType unary_op_type_;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

/*
 * A specialization for Binary operations. Binary operations take in two inputs
 * and produce a single output. Examples include:
 *  1) Add/mul/div/mod/sub (A * B)
 *  2) LT (A < B)
 */
class TORCH_CUDA_API BinaryOp : public Expr {
 public:
  ~BinaryOp() = default;
  BinaryOp(BinaryOpType _type, Val* _out, Val* _lhs, Val* _rhs);

  BinaryOp(const BinaryOp* src, IrCloner* ir_cloner);

  BinaryOp(const BinaryOp& other) = delete;
  BinaryOp& operator=(const BinaryOp& other) = delete;

  BinaryOp(BinaryOp&& other) = delete;
  BinaryOp& operator=(BinaryOp&& other) = delete;

  Val* out() const {
    return out_;
  }
  Val* lhs() const {
    return lhs_;
  }
  Val* rhs() const {
    return rhs_;
  }

  BinaryOpType getBinaryOpType() const {
    return binary_op_type_;
  }

  bool sameAs(const BinaryOp* other) const;

 private:
  const BinaryOpType binary_op_type_;
  Val* const out_ = nullptr;
  Val* const lhs_ = nullptr;
  Val* const rhs_ = nullptr;
};

/*
 * Broadcast _in to match _out. broadcast_dims are relative to out. Where
 * broadcast_dims.size() + _in->nDims() == _out->nDims().
 */
class TORCH_CUDA_API BroadcastOp : public Expr {
 public:
  ~BroadcastOp() = default;
  BroadcastOp(Val* _out, Val* _in);

  BroadcastOp(const BroadcastOp* src, IrCloner* ir_cloner);

  BroadcastOp(const BroadcastOp& other) = delete;
  BroadcastOp& operator=(const BroadcastOp& other) = delete;

  BroadcastOp(BroadcastOp&& other) = delete;
  BroadcastOp& operator=(BroadcastOp&& other) = delete;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }

  bool sameAs(const BroadcastOp* const other) const;

 private:
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

/*
 * Reduction operation. Out is first initialized to _init. Then
 * _reduction_op_type is used to update out as out = reductionOp(out, in).
 * Output's axes marked as reduction will be reduced to produce an output
 * tensor. The output tensors size will be the size of all
 * non-reduction/non-broadcast dimensions.
 */
class TORCH_CUDA_API ReductionOp : public Expr {
 public:
  ~ReductionOp() = default;
  ReductionOp(BinaryOpType _reduction_op_type, Val* _init, Val* _out, Val* _in);

  ReductionOp(const ReductionOp* src, IrCloner* ir_cloner);

  ReductionOp(const ReductionOp& other) = delete;
  ReductionOp& operator=(const ReductionOp& other) = delete;

  ReductionOp(ReductionOp&& other) = delete;
  ReductionOp& operator=(ReductionOp&& other) = delete;

  Val* out() const {
    return out_;
  }
  Val* in() const {
    return in_;
  }
  Val* init() const {
    return init_;
  }

  BinaryOpType getReductionOpType() const {
    return reduction_op_type_;
  }

  bool sameAs(const ReductionOp* const other) const;

 private:
  const BinaryOpType reduction_op_type_;
  Val* const init_ = nullptr;
  Val* const out_ = nullptr;
  Val* const in_ = nullptr;
};

class TORCH_CUDA_API TernaryOp : public Expr {
 public:
  ~TernaryOp() = default;
  TernaryOp(TernaryOpType _type, Val* _out, Val* _in1, Val* _in2, Val* _in3);

  TernaryOp(const TernaryOp* src, IrCloner* ir_cloner);

  TernaryOp(const TernaryOp& other) = delete;
  TernaryOp& operator=(const TernaryOp& other) = delete;

  TernaryOp(TernaryOp&& other) = delete;
  TernaryOp& operator=(TernaryOp&& other) = delete;

  Val* out() const {
    return out_;
  }

  Val* in1() const {
    return in1_;
  }
  Val* in2() const {
    return in2_;
  }
  Val* in3() const {
    return in3_;
  }

  TernaryOpType getTernaryOpType() const {
    return ternary_op_type_;
  }

  bool sameAs(const TernaryOp* other) const;

 private:
  const TernaryOpType ternary_op_type_;
  Val* const out_ = nullptr;
  Val* const in1_ = nullptr;
  Val* const in2_ = nullptr;
  Val* const in3_ = nullptr;
};

// Simply a representation of an annotated 1D iterable from start to extent.
// TensorDomains which represent how to iterate over a tensor is made up of
// IterDomains to form an ND iterable. We directly set parallization strategies
// on IterDomains.
class TORCH_CUDA_API IterDomain : public Val {
 public:
  IterDomain(
      Val* _start,
      Val* _extent,
      ParallelType _parallel_type = ParallelType::Serial,
      IterType _iter_type = IterType::Iteration,
      bool _is_rfactor_domain = false);

  IterDomain(const IterDomain* src, IrCloner* ir_cloner);

  bool sameAs(const IterDomain* const other) const;

  // Returns a new IterDomain matching properties of this
  // TODO: parallel_method->getParallelType
  IterDomain* clone() const {
    return new IterDomain(
        start(),
        extent(),
        getParallelType(),
        getIterType(),
        isRFactorProduct());
  }

  static IterDomain* merge(IterDomain* outer, IterDomain* inner);

  // TODO: Make protected and friend TensorDomain so only it can call into this
  // directly, users should not be able to use this call
  static std::pair<IterDomain*, IterDomain*> split(IterDomain* in, Val* factor);

  bool isReduction() const {
    return getIterType() == IterType::Reduction;
  }

  bool isRFactorProduct() const {
    return is_rfactor_domain_;
  }

  bool isBroadcast() const {
    return getIterType() == IterType::BroadcastWithStride ||
        getIterType() == IterType::BroadcastWithoutStride;
  }

  bool isParallelized() const {
    return getParallelType() != ParallelType::Serial;
  }

  // Return if this iter domain is mapped to a grid dimension
  bool isBlockDim() const {
    return (
        getParallelType() == ParallelType::BIDz ||
        getParallelType() == ParallelType::BIDy ||
        getParallelType() == ParallelType::BIDx);
  }

  // Return if this iter domain is mapped to a block dimension
  bool isThreadDim() const {
    return (
        getParallelType() == ParallelType::TIDz ||
        getParallelType() == ParallelType::TIDy ||
        getParallelType() == ParallelType::TIDx);
  }

  // Return if this iter domain is either mapped to a block or grid dimension
  bool isThread() const {
    return (isBlockDim() || isThreadDim());
  }

  void parallelize(ParallelType t) {
    parallel_type_ = t;

    TORCH_CHECK(
        t != ParallelType::Vectorize, "Vectorization not yet supported.");

    if (t == ParallelType::Unroll)
      TORCH_CHECK(
          start()->isZeroInt() && extent()->isConstScalar(),
          "Unrolling only supported with start = 0 and extent as a const int, but got ",
          "a start of ",
          start(),
          " and extent ",
          extent(),
          " .");
  }

  ParallelType getParallelType() const {
    return parallel_type_;
  }

  IterType getIterType() const {
    return iter_type_;
  }

  Val* start() const {
    return start_;
  }
  Val* extent() const;

  Val* rawExtent() const {
    return extent_;
  }

  IterDomain(const IterDomain& other) = delete;
  IterDomain& operator=(const IterDomain& other) = delete;

  IterDomain(IterDomain&& other) = delete;
  IterDomain& operator=(IterDomain&& other) = delete;

 private:
  Val* const start_ = nullptr;
  Val* const extent_ = nullptr;
  ParallelType parallel_type_ = ParallelType::Serial;
  IterType iter_type_ = IterType::Iteration;
  bool is_rfactor_domain_ = false;
};

/*
 * TensorDomain holds a vector of IterDomains. It holds an IterDomain for every
 * logical axis in its associated tensor. TensorDomain does not directly hold
 * the Tensor it is associated with, and in theory could be associated with
 * multiple tensors. TensorDomain's primary responsibility is to provide a
 * mechanism to access history of transformations that were used to generate it.
 * This is done through the normal interaction of Expr/Val in Fusion. i.e. if we
 * want to know the previous operation generating a particular TensorDomain we
 * can simply call FusionGuard::getCurFusion()->origin(a_tensor_domain) which
 * should give us an operation in the list [split, merge] or similar
 * operations that take in a TensorDomain, applies a transformation and outputs
 * a tensor domain.
 */
class TORCH_CUDA_API TensorDomain : public Val {
 public:
  TensorDomain() = delete;
  ~TensorDomain() = default;

  TensorDomain(const TensorDomain& other) = delete;
  TensorDomain& operator=(const TensorDomain& other) = delete;

  TensorDomain(TensorDomain&& other) = delete;
  TensorDomain& operator=(TensorDomain&& other) = delete;

  explicit TensorDomain(
      std::vector<IterDomain*> _domain,
      std::vector<bool> _contiguity = std::vector<bool>());

  TensorDomain(
      std::vector<IterDomain*> _root_domain,
      std::vector<IterDomain*> _domain,
      std::vector<bool> _contiguity = std::vector<bool>());

  TensorDomain(
      std::vector<IterDomain*> _root_domain,
      std::vector<IterDomain*> _rfactor_domain,
      std::vector<IterDomain*> _domain,
      std::vector<bool> _contiguity = std::vector<bool>(),
      std::vector<bool> _placeholder = std::vector<bool>(),
      std::unordered_map<IterDomain*, IterDomain*> _crossover_map = {});

  TensorDomain(const TensorDomain* src, IrCloner* ir_cloner);

  bool operator==(const TensorDomain& other) const;
  bool operator!=(const TensorDomain& other) const {
    return !(*this == other);
  }

  std::vector<IterDomain*>::size_type nDims() const {
    return domain_.size();
  }

  bool sameAs(const TensorDomain* const other) const;

  static bool sameAs(
      const std::vector<IterDomain*>& lhs,
      const std::vector<IterDomain*>& rhs);

  const std::vector<IterDomain*>& domain() const {
    return domain_;
  }

  const std::vector<bool>& contiguity() const {
    return contiguity_;
  }

  const std::vector<bool>& placeholder() const {
    return placeholder_;
  }

  const std::unordered_map<const IterDomain*, bool>& placeholder_map() const {
    return placeholder_map_;
  }

  ir_utils::FilterView<std::vector<IterDomain*>::const_iterator> noPlaceholder() const;

  std::string getContiguityString() const {
    std::stringstream ss;
    for (auto b : contiguity()) {
      ss << (b ? "t" : "f");
    }
    return ss.str();
  }

  bool hasReduction() const;
  bool hasBlockReduction() const;
  bool hasGridReduction() const;
  bool hasBlockBroadcast() const;
  bool hasBroadcast() const;
  bool hasRFactor() const;

  c10::optional<unsigned int> getReductionAxis() const;

  const std::vector<IterDomain*>& noReductions() const {
    return no_reduction_domain_;
  }

  const std::vector<IterDomain*>& noBroadcasts() const {
    return no_bcast_domain_;
  }

  const std::vector<IterDomain*>& getRootDomain() const {
    return root_domain_;
  };

  const std::vector<IterDomain*>& getRFactorDomain() const {
    return rfactor_domain_;
  };

  // If rfactor domain exists in domain() return it, otherwise return root
  // domain.
  const std::vector<IterDomain*>& getMaybeRFactorDomain() const {
    return hasRFactor() ? getRFactorDomain() : getRootDomain();
  }

  void resetDomains() {
    no_reduction_domain_ = noReductions(domain_);
    no_bcast_domain_ = noBroadcasts(domain_);
  }

  // i here is int, as we want to accept negative value and ::size_type can be a
  // uint.
  IterDomain* axis(int i) const;

  size_t posOf(IterDomain* id) const;

  // Split "axis" into 2 axes where the inner axes is size of "factor"
  // and outer axis is size axis.size() / factor. Allow factor to be symbolic
  // value instead of constant.
  // TODO: Make protected and friend TensorDomain so only it can call into this
  // directly, users should not be able to use this call
  void split(int axis_, Val* factor);

  // Merge axis_o and axis_i. axis_i is the fast changing dimension. Resulting
  // axis is by default placed at original position axis_o
  void merge(int axis_o, int axis_i);

  // Reorder axes according to map[old_pos] = new_pos
  void reorder(const std::unordered_map<int, int>& old2new);

  static std::vector<IterDomain*> orderedAs(
      const std::vector<IterDomain*>& td,
      const std::unordered_map<int, int>& old2new);

  static std::vector<IterDomain*> noReductions(const std::vector<IterDomain*>&);
  static std::vector<IterDomain*> noBroadcasts(const std::vector<IterDomain*>&);

  static bool hasBroadcast(const std::vector<IterDomain*>&);
  static bool hasReduction(const std::vector<IterDomain*>&);

  // return std::pair<producer_id, consumer_id> representing
  // the mapping between corresponding axes. Not all axes have
  // corresponding mapping, e.g., broadcast axis in consumer
  // does not have any corresponding axis in producer.
  static std::vector<std::pair<int, int>> mapDomainPandC(
      const std::vector<IterDomain*>& producer,
      const std::vector<IterDomain*>& consumer);

  // Create a map between producer root IterDomains and consumer root
  // IterDomains.
  static std::vector<std::pair<IterDomain*, IterDomain*>> mapRootPandC(
      const TensorDomain* producer,
      const TensorDomain* consumer);

  // Create a map from consumer root IterDomains -> producer root IterDomains.
  // Only those root consumer IDs present in consumer_root_dims_to_map
  // will be attempted to map to their corresponding producer IDs.
  static std::unordered_map<IterDomain*, IterDomain*> mapRootCtoP(
      const TensorDomain* consumer,
      const TensorDomain* producer,
      const std::unordered_set<IterDomain*>& consumer_root_dims_to_map);

  static std::unordered_map<IterDomain*, IterDomain*> mapRootCtoP(
      const TensorDomain* consumer,
      const TensorDomain* producer) {
    return mapRootCtoP(
        consumer,
        producer,
        std::unordered_set<IterDomain*>(
            consumer->getRootDomain().begin(),
            consumer->getRootDomain().end()));
  }

  // Create a map from producer root IterDomains -> consumer root IterDomains.
  // Only those root producer IDs present in producer_maybe_rfactor_dims_to_map
  // will be attempted to map to their corresponding consumer IDs.
  static std::unordered_map<IterDomain*, IterDomain*> mapRootPtoC(
      const TensorDomain* producer,
      const TensorDomain* consumer,
      const std::unordered_set<IterDomain*>&
          producer_maybe_rfactor_dims_to_map);

  static std::unordered_map<IterDomain*, IterDomain*> mapRootPtoC(
      const TensorDomain* producer,
      const TensorDomain* consumer) {
    auto p_root = producer->getMaybeRFactorDomain();
    return mapRootPtoC(
        producer,
        consumer,
        std::unordered_set<IterDomain*>(p_root.begin(), p_root.end()));
  }

  // pair is in order where second is the consumer of first
  std::pair<TensorDomain*, TensorDomain*> rFactor(const std::vector<int>& axes);

 private:
  void updatePlaceholderMap();

 private:
  const std::vector<IterDomain*> root_domain_;
  std::vector<IterDomain*> domain_;
  std::vector<IterDomain*> no_bcast_domain_;
  std::vector<IterDomain*> no_reduction_domain_;
  const std::vector<IterDomain*> rfactor_domain_;
  const std::vector<bool> contiguity_;
  const std::vector<bool> placeholder_;
  std::unordered_map<const IterDomain*, bool> placeholder_map_;
};

class Merge;

//#ifndef INCOMPLETE_MERGE_EXPR
//#define INCOMPLETE_MERGE_EXPR
//#endif

class TORCH_CUDA_API ComputeDomain {
 public:
  ComputeDomain() = default;
  explicit ComputeDomain(const TensorDomain* td);

  const TensorDomain* td() const {
    return td_;
  }

  size_t nDims() const {
    return axes().size();
  }

  IterDomain* axis(size_t idx) const {
    TORCH_INTERNAL_ASSERT(idx < nDims(),
                          "Out of range error. Attempting to access axis at offset ",
                          idx, " of size-", axes_.size(),
                          " compute domain.");
    return axes_[idx];
  }

  IterDomain* getAxisForReplay(IterDomain* id) const;
  IterDomain* getAxisForReplay(size_t idx) const;

  const std::deque<IterDomain*>& axes() const {
    return axes_;
  }

#ifdef INCOMPLETE_MERGE_EXPR
  using IncompleteMergeType = std::unordered_map<Merge*, bool>;
#else
  using IncompleteMergeType = std::unordered_map<IterDomain*, bool>;
#endif

  void computeAt(const TensorDomain* td,
                 int this_pos,
                 const ComputeDomain* target,
                 int target_pos,
                 const std::vector<size_t>& td2cd_map,
                 const std::unordered_map<IterDomain*, IterDomain*>& crossover_map,
                 const IncompleteMergeType& incomplete_merge);

  std::unordered_set<IterDomain*> getRootDomain() const;
  std::unordered_set<IterDomain*> getRFactorDomain() const;
  std::unordered_set<IterDomain*> getMaybeRFactorDomain() const;
  std::unordered_set<IterDomain*> getCompleteRootDomain() const;
  std::unordered_set<IterDomain*> getInputsTo(const std::vector<IterDomain*>& axes) const;

  // Return a map from IterDomains in ComputeDomain to IterDomains in a given domain
  static std::unordered_map<IterDomain*, IterDomain*> mapRootDomain(
      const std::vector<IterDomain*>& root_domain,
      const std::unordered_set<IterDomain*>& compute_root_ids);

  void split(const TensorDomain* new_td, int axis_idx);

  void merge(const TensorDomain* new_td, int axis_o, int axis_i);

  void reorder();

  size_t getComputeAtPos() const {
    return pos_;
  }

  size_t getTensorDomainPos(size_t cd_pos) const {
    while (cd_pos > 0) {
      auto cd_axis = cd_pos - 1;
      if (isComputeDomainAxisUsed(cd_axis)) {
        auto td_axis = getTensorDomainAxisIndex(cd_axis);
        return td_axis + 1;
      }
      --cd_pos;
    }
    return 0;
  }

  // Returns the ComputeDomain position that corresponds to the
  // given TensorDomain position
  size_t getComputeDomainPos(size_t td_pos) const {
    if (td_pos == 0) return 0;
    auto td_axis = td_pos - 1;
    auto cd_axis = getComputeDomainAxisIndex(td_axis);
    return cd_axis + 1;
  }

  size_t getComputeDomainAxisIndex(size_t td_axis) const {
    TORCH_INTERNAL_ASSERT(td_axis < td_map_.size());
    return td_map_.at(td_axis);
  }

  bool isComputeDomainAxisUsed(size_t cd_axis) const {
    return std::find(td_map_.begin(), td_map_.end(), cd_axis) != td_map_.end();
  }

  size_t getTensorDomainAxisIndex(size_t cd_axis_idx) const {
    auto it = std::find(td_map_.begin(), td_map_.end(), cd_axis_idx);
    TORCH_INTERNAL_ASSERT(it != td_map_.end());
    return std::distance(td_map_.begin(), it);
  }

  IterDomain* getTensorDomainAxis(size_t cd_axis_idx) const {
    return td_->axis(getTensorDomainAxisIndex(cd_axis_idx));
  }

  std::ostream& print(std::ostream& os) const;

  static const IterDomain* getConcreteDomain(const IterDomain* id);
  static bool sameAxes(const IterDomain* id1, const IterDomain* id2);

  void registerAsDependent(ComputeDomain* target);
  void registerDependent(ComputeDomain* dependent, size_t pos);

  const auto& crossoverMap() const {
    return crossover_map_;
  }

  const auto& incompleteMerge() const {
    return incomplete_merge_;
  }

  const std::vector<Expr*>& getExprsToRoot() const;
  IterDomain* getTensorDomainAxisForDependentAxis(IterDomain* cd_axis) const;

  std::unordered_map<IterDomain*, IterDomain*> mapFromProducer(
      const TensorDomain* producer) const;
  std::unordered_map<IterDomain*, IterDomain*> mapToProducer(
      const TensorDomain* producer) const;
  std::unordered_map<IterDomain*, IterDomain*> mapFromConsumer(
      const TensorDomain* consumer) const;
  std::unordered_map<IterDomain*, IterDomain*> mapToConsumer(
      const TensorDomain* consumer) const;
  IterDomain* getCorrespondingComputeDomainID(IterDomain* td_id) const;
  IterDomain* getCorrespondingTensorDomainID(IterDomain* cd_id) const;

#if 0
  void cacheBefore();
#endif
 private:
  void setAxis(size_t cd_axis, IterDomain* id);
  void insertAxis(size_t cd_axis, IterDomain* cd_id, size_t td_axis);
  void eraseAxis(size_t cd_axis);
  void sanityCheck() const;
  void fixupPosition();
  void updateDependents(size_t first_changed_axis);
  bool isDependent(const ComputeDomain* cd) const;
  void buildExprListToRoot() const;
  void invalidateExprList() {
    exprs_to_root_valid_ = false;
  }

  const std::unordered_map<IterDomain*, IterDomain*>& getCD2TDMap() const;
  std::unordered_map<IterDomain*, IterDomain*> mapProducerAndComputeDomain(
      const TensorDomain* producer, bool from_producer) const;
  std::unordered_map<IterDomain*, IterDomain*> mapConsumerAndComputeDomain(
      const TensorDomain* consumer, bool from_consumer) const;

 private:
  const TensorDomain* td_ = nullptr;
  std::deque<IterDomain*> axes_;
  bool computed_at_ = false;
  // Mapping from TD IterDomain index to CD IterDomain index
  std::vector<size_t> td_map_;
  size_t pos_ = 0;
  std::vector<std::pair<size_t, ComputeDomain*>> dependents_;

  std::unordered_map<IterDomain*, IterDomain*> crossover_map_;
#ifdef INCOMPLETE_MERGE_EXPR
  std::unordered_map<Merge*, bool> incomplete_merge_;
#else
  std::unordered_map<IterDomain*, bool> incomplete_merge_;
#endif

  mutable bool exprs_to_root_valid_ = false;
  mutable std::vector<Expr*> exprs_to_root_;
  mutable std::unordered_map<IterDomain*, IterDomain*> cd2td_map_;
};

std::ostream& operator<<(std::ostream& os, const ComputeDomain& cd);

/*
 * Representation a split on an IterDomain by "factor"
 * TODO: Implement split by nparts
 */
class TORCH_CUDA_API Split : public Expr {
 public:
  ~Split() = default;

  Split(const Split& other) = delete;
  Split& operator=(const Split& other) = delete;

  Split(Split&& other) = delete;
  Split& operator=(Split&& other) = delete;

  Split(IterDomain* _outer, IterDomain* _inner, IterDomain* _in, Val* _factor);

  Split(const Split* src, IrCloner* ir_cloner);

  IterDomain* outer() const {
    return outer_;
  }
  IterDomain* inner() const {
    return inner_;
  }
  IterDomain* in() const {
    return in_;
  }
  Val* factor() const {
    return factor_;
  }
  bool sameAs(const Split* const other) const;

 private:
  IterDomain* const outer_ = nullptr;
  IterDomain* const inner_ = nullptr;
  IterDomain* const in_ = nullptr;
  Val* const factor_ = nullptr;
};

/*
 * Merge the IterDomains outer and inner into one domain, outer and inner
 * dictate which will be traversed first (inner). Both IterDomains must be of
 * the same iter or reduction type, as well as the same parallelization strategy
 * if there is one.
 * TODO: Should this be a unary op type?
 */
class TORCH_CUDA_API Merge : public Expr {
 public:
  ~Merge() = default;
  Merge(IterDomain* _out, IterDomain* _outer, IterDomain* _inner);

  Merge(const Merge* src, IrCloner* ir_cloner);

  Merge(const Merge& other) = delete;
  Merge& operator=(const Merge& other) = delete;

  Merge(Merge&& other) = delete;
  Merge& operator=(Merge&& other) = delete;

  IterDomain* out() const {
    return out_;
  }
  IterDomain* outer() const {
    return outer_;
  }
  IterDomain* inner() const {
    return inner_;
  }

  bool sameAs(const Merge* const other) const;

 private:
  IterDomain* const out_ = nullptr;
  IterDomain* const outer_ = nullptr;
  IterDomain* const inner_ = nullptr;
};

/*
 * Integer value which has a special name. These could be:
 * - threadIdx.x
 * - blockIdx.y
 * - blockDim.z
 * - T3.stride[2]
 */
class TORCH_CUDA_API NamedScalar : public Val {
 public:
  ~NamedScalar() = default;
  NamedScalar() = delete;

  NamedScalar(std::string _name, DataType dtype)
      : Val(ValType::NamedScalar, dtype), name_(_name) {}

  NamedScalar(const NamedScalar* src, IrCloner* ir_cloner);

  NamedScalar(const NamedScalar& other) = delete;
  NamedScalar& operator=(const NamedScalar& other) = delete;

  NamedScalar(NamedScalar&& other) = delete;
  NamedScalar& operator=(NamedScalar&& other) = delete;

  const std::string& name() const {
    return name_;
  }

  bool sameAs(const NamedScalar* const other) const {
    return other->name().compare(name()) == 0;
  }

  // Return the named scalar extent of a parallel dimension (e.g. blockDim.x)
  static NamedScalar* getParallelDim(ParallelType p_type);

  // Return the named scalar index of a parallel dimension (e.g. threadIdx.x)
  static NamedScalar* getParallelIndex(ParallelType p_type);

  // Return the parallel type of this NamedScalar if it is an extent of a
  // parallel dimension
  c10::optional<ParallelType> getParallelDim() const;

  // Return the parallel type of this NamedScalar if it is an index of a
  // parallel dimension
  c10::optional<ParallelType> getParallelIndex() const;

 private:
  std::string name_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
