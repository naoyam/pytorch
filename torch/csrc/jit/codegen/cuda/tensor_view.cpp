#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {
DataType aten_opt_type_map(const c10::optional<at::ScalarType>& scalar_type) {
  return scalar_type.has_value() ? aten_to_data_type(scalar_type.value())
                                 : DataType::Null;
}

c10::optional<TensorContiguity> infer_contiguity_from_tensor_type(
    const std::shared_ptr<c10::TensorType>& tensor_type) {
  if (!tensor_type->isComplete()) {
    return c10::nullopt;
  } else {
    return TensorContiguity(
        *(tensor_type->sizes().concrete_sizes()),
        *(tensor_type->strides().concrete_sizes()));
  }
}

} // namespace

TensorView::TensorView(TensorDomain* _domain, DataType dtype)
    : Val(ValType::TensorView, dtype), domain_(_domain) {}

TensorView::TensorView(const std::shared_ptr<c10::TensorType>& tensor_type)
    : Val(ValType::TensorView, aten_opt_type_map(tensor_type->scalarType())) {
  std::vector<IterDomain*> sizes;
  TORCH_CHECK(
      tensor_type->dim().has_value(), "Requires static rank for Tensor");
  for (int i = 0; i < tensor_type->dim().value(); i++) {
    sizes.push_back(new IterDomain(new Int(0), new Int()));
  }
  domain_ = new TensorDomain(sizes);
}

TensorView* TensorView::clone() const {
  TensorView* new_view = new TensorView(domain_, getDataType().value());
  new_view->compute_at_view_ = compute_at_view_;
  new_view->compute_at_axis_ = compute_at_axis_;
  return new_view;
}

bool TensorView::hasReduction() const {
  return domain()->hasReduction();
}

TensorView* TensorView::newForOutput(DataType dtype) const {
  std::vector<IterDomain*> domain_copy;
  for (decltype(this->nDims()) i = 0; i < this->nDims(); i++) {
    // If reduction axis, don't copy it over. Reduction axes are owned by
    // consumers and we're copying over a producer.
    if (this->axis(i)->isReduction())
      continue;
    domain_copy.push_back(
        new IterDomain(this->axis(i)->start(), this->axis(i)->extent()));
  }
  TensorDomain* td = new TensorDomain(domain_copy);
  return new TensorView(td, dtype);
};

// TODO: How do we adjust this so we can reduce to a single scalar value?
TensorView* TensorView::newForReduction(std::vector<unsigned int> axes) const {
  TensorDomain* orig_domain = this->getRootDomain()->noReductions();
  std::set<unsigned int> axes_set(axes.begin(), axes.end());

  std::vector<IterDomain*> new_domain;

  TORCH_INTERNAL_ASSERT(
      (*axes_set.end()) < orig_domain->nDims(),
      "Error setting up reduction, reduction axis is outside nDims. Keep in mind reductions are relative to root domains, not modified views.");

  for (decltype(orig_domain->nDims()) dim = 0; dim < orig_domain->nDims();
       dim++) {
    IterDomain* orig_dom = orig_domain->axis(dim);

    bool isReduction = false;
    if ((*axes_set.begin()) == dim) {
      isReduction = true;
      axes_set.erase(axes_set.begin());
    }

    new_domain.push_back(new IterDomain(
        orig_dom->start(),
        orig_dom->extent(),
        ParallelType::Serial,
        isReduction));
  }

  TensorDomain* td = new TensorDomain(new_domain);
  return new TensorView(td, this->getDataType().value());
};

TensorDomain* TensorView::getRootDomain() const {
  return TransformIter::getRoot(this->domain());
};

void TensorView::resetView() {
  setDomain(getRootDomain());
  compute_at_view_ = nullptr;
  compute_at_axis_ = 0;
}

std::vector<IterDomain*>::size_type TensorView::nDims() const {
  return domain()->nDims();
}

IterDomain* TensorView::axis(int pos) const {
  if (pos < 0)
    pos += domain()->nDims();
  TORCH_CHECK(
      pos >= 0 && pos < domain()->nDims(),
      "Tried to access position ",
      pos,
      " in domain: ",
      domain());
  return domain()->axis(pos);
}

TensorView* TensorView::unsafeClone() const {
  TensorView* new_view = new TensorView(domain_, getDataType().value());
  new_view->compute_at_view_ = compute_at_view_;
  new_view->compute_at_axis_ = compute_at_axis_;
  new_view->setMemoryType(memory_type_);
  new_view->name_ = name();
  MemoryType memory_type_ = MemoryType::Global;

  return new_view;
}

void TensorView::copyDomain(const TensorDomain* td) {
  std::vector<IterDomain*> idv;
  for (decltype(td->nDims()) i = 0; i < td->nDims(); i++)
    idv.push_back(td->axis(i));
  setDomain(new TensorDomain(idv));
}

void TensorView::computeAt_impl(TensorView* consumer, int axis) {
  // Reset view otherwise will conflict with replay.
  this->compute_at_view_ = nullptr;
  this->compute_at_axis_ = 0;
  // replay this as consumer / producer as consumer
  TransformReplay::replayPasC(this, consumer, axis);
  this->compute_at_view_ = consumer;
  this->compute_at_axis_ = (unsigned int)axis;
}

void TensorView::forwardComputeAt_impl(TensorView* producer, int axis) {
  // Reset view otherwise will conflict with replay.
  producer->compute_at_view_ = nullptr;
  producer->compute_at_axis_ = 0;
  TransformReplay::replayCasP(this, producer, axis);
  producer->compute_at_view_ = this;
  producer->compute_at_axis_ = (unsigned int)axis;
}

namespace {
// Wrapper around set_intersection
template <typename T>
std::set<T> set_intersection(const std::set<T>& set1, const std::set<T>& set2) {
  std::set<T> intersection;
  std::set_intersection(
      set1.begin(),
      set1.end(),
      set2.begin(),
      set2.end(),
      std::inserter(intersection, intersection.begin()));
  return intersection;
}

// convert an iterable of Val* to be an iterable of TensorView*
template <typename T1, typename T2>
T1 tv_iterable(const T2& val_iterable) {
  T1 tv_iterable = T1();
  std::transform(
      val_iterable.begin(),
      val_iterable.end(),
      std::back_inserter(tv_iterable),
      [](Val* v) {
        TORCH_INTERNAL_ASSERT(
            v->getValType().value() == ValType::TensorView,
            "When following the computeAt dependency chain, a non TensorView value was found.");
        return static_cast<TensorView*>(v);
      });
  return tv_iterable;
}
} // namespace

TensorView* TensorView::computeAt(TensorView* consumer, int axis) {
  TORCH_CHECK(
      this->fusion() == consumer->fusion(),
      this,
      " and ",
      consumer,
      " are not in the same fusion.");

  FusionGuard fg(this->fusion());

  TORCH_CHECK(
      !this->sameAs(consumer), "Cannot call this->computeAt(this, ...)");

  if (axis < 0)
    // Compute at is a bit strange where size is the maximum acceptable value
    // instead of size-1
    axis += int(consumer->nDims()) + 1;

  TORCH_CHECK(
      axis >= 0 && axis < consumer->nDims() + 1,
      "Compute at called on an axis outside valid range.");

  // If not direct relationship follow dependency chain.
  auto dep_chains = DependencyCheck::getAllDependencyChains(this, consumer);

  std::deque<Val*> dep_chain;
  if (!dep_chains.empty())
    dep_chain = dep_chains.front();

  TORCH_CHECK(
      !dep_chain.empty(),
      "Compute At expects ",
      this,
      " is a dependency of ",
      consumer,
      ", however it is not.");

  TORCH_INTERNAL_ASSERT(
      dep_chain.back() == consumer && dep_chain[0] == this,
      "Error computing dependency chain.");

  // Replay from consumer to producer
  while (dep_chain.size() > 1) {
    Val* consumer_val = dep_chain.back();
    dep_chain.pop_back();
    Val* producer_val = dep_chain.back();

    TORCH_INTERNAL_ASSERT(
        consumer_val->getValType().value() == ValType::TensorView &&
            producer_val->getValType().value() == ValType::TensorView,
        "When following the computeAt dependency chain, a non TensorView value was found.");

    TensorView* running_consumer = static_cast<TensorView*>(consumer_val);
    TensorView* running_producer = static_cast<TensorView*>(producer_val);
    running_producer->computeAt_impl(running_consumer, axis);
  }

  /*
   * Compute At has now worked from consumer to producer, transforming producer
   * to match computeAt selected in consumer We now need to work from producer
   * up to its consumers (including indirect consumption) so their use also
   * matches. If we can find a TV that contains all uses of producer, we can
   * terminate this propagation there. If not, we need to propagate all the way
   * to outputs.
   *
   * First we'll look for that terminating point.
   */

  // Grab all uses of producer
  auto val_all_consumer_chains =
      DependencyCheck::getAllDependencyChainsTo(this);

  // Convert dep chains to tensor view chains
  std::deque<std::deque<TensorView*>> all_consumer_chains;
  for (auto val_dep_chain : val_all_consumer_chains)
    all_consumer_chains.push_back(
        tv_iterable<std::deque<TensorView*>>(val_dep_chain));

  std::set<TensorView*> common_consumers(
      all_consumer_chains.front().begin(), all_consumer_chains.front().end());
  for (auto dep_chain : all_consumer_chains) {
    std::set<TensorView*> chain_consumers(dep_chain.begin(), dep_chain.end());
    common_consumers = set_intersection(
        common_consumers,
        std::set<TensorView*>(dep_chain.begin(), dep_chain.end()));
  }

  // Remove all TVs between producer and consumer
  for (auto dep_chain : dep_chains) {
    auto tv_chain = tv_iterable<std::deque<TensorView*>>(dep_chain);
    for (auto tv : tv_chain) {
      if (tv != consumer)
        common_consumers.erase(tv);
    }
  }

  // Grab the first (topologically) common consumer
  TensorView* common_consumer = nullptr;
  if (!common_consumers.empty()) {
    for (TensorView* tv : all_consumer_chains.front())
      if (common_consumers.find(tv) != common_consumers.end()) {
        common_consumer = tv;
        break;
      }
  }

  // Forward compute at through all consumers until common_consumer if there is
  // one
  std::set<TensorView*> output_set;
  std::vector<TensorView*> ordered_outputs;
  for (auto dep_chain : all_consumer_chains) {
    while (dep_chain.size() > 1) {
      TensorView* running_producer = dep_chain.front();
      dep_chain.pop_front();
      TensorView* running_consumer = dep_chain.front();

      if (running_producer == common_consumer)
        break;

      running_consumer->forwardComputeAt_impl(running_producer, axis);
      if (dep_chain.size() == 1) { // last one
        if (output_set.find(running_consumer) == output_set.end()) {
          output_set.emplace(running_consumer);
          ordered_outputs.push_back(running_consumer);
        }
      }
    }
  }

  for (decltype(ordered_outputs.size()) i{0};
       i < ordered_outputs.size() - 1 && ordered_outputs.size() > 0;
       i++) {
    ordered_outputs[i]->computeAt_impl(ordered_outputs[i + 1], axis);
  }

  return this;
}

TensorView* TensorView::split(int axis, int factor) {
  if (axis < 0)
    axis += domain()->nDims();

  TORCH_CHECK(
      axis >= 0 && axis < domain()->nDims(),
      "Trying to split axis outside of TensorView's range.");

  TORCH_CHECK(factor >= 0, "Cannot split by a factor less than 1.");

  if (getComputeAtView() != nullptr)
    if (axis < getComputeAtAxis())
      TORCH_CHECK(false, "Cannot split axis within compute at range.");

  setDomain(domain()->split(axis, factor));
  return this;
}

// Merge "axis" and "axis+1" into 1 dimension
TensorView* TensorView::merge(int axis) {
  if (axis < 0)
    axis += domain()->nDims();

  TORCH_CHECK(
      axis >= 0 && axis + 1 < domain()->nDims(),
      "Trying to merge axis outside of TensorView's range.");

  if (getComputeAtView() != nullptr)
    if (axis + 1 < getComputeAtAxis())
      TORCH_CHECK(false, "Cannot merge axis within compute at range.");

  setDomain(domain()->merge(axis));
  return this;
}

// Reorder axes according to map[old_pos] = new_pos
TensorView* TensorView::reorder(const std::unordered_map<int, int>& old2new_) {
  // START VALIDATION CHECKS
  // adjust based on negative values (any negative values gets nDims added to
  // it)
  std::unordered_map<int, int> old2new;
  auto ndims = nDims();
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
  bool out_of_range = std::any_of(
      old2new.begin(),
      old2new.end(),
      [ndims](std::unordered_map<int, int>::value_type entry) {
        return entry.first < 0 || entry.first >= ndims || entry.second < 0 ||
            entry.second >= ndims;
      });

  TORCH_CHECK(
      !out_of_range,
      "TensorView reorder axes are outside the number of dimensions in the TensorView.")

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
        return entry.first;
      });

  // Error out if duplicate values are found.
  TORCH_CHECK(
      old_pos_set.size() == old2new.size() &&
          new_pos_set.size() == old2new.size(),
      "Duplicate entries in transformation map sent to TensorView reorder.");

  // Check if we're trying to reorder any values outside of the computeAt axis

  if (hasComputeAt()) {
    auto compute_at_axis = getComputeAtAxis();
    bool outside_computeat = std::any_of(
        old2new.begin(),
        old2new.end(),
        [compute_at_axis](std::unordered_map<int, int>::value_type entry) {
          return entry.first < compute_at_axis ||
              entry.second < compute_at_axis;
        });
    TORCH_CHECK(
        !outside_computeat,
        "Cannot reorder dimensions that are outside computeAt axis.");
  }
  // END VALIDATION CHECKS
  setDomain(domain()->reorder(old2new_));

  return this;
}

/*
 * Take reduction axes out of this domain, and create a new domain. New domain
 * will be used to create this domain. For example: TV1[I0, I1] = TV0[I0, R0,
 * R1, I1] TV0->rfactor({1}) TV0 is transformed to -> TV0[I0, R1, I1] The
 * TensorView returned is: TV2[I0, R0, I3, I1] The reduction will now beset
 * as: TV1[I0, R1, I1] = TV2[I0, R0, I3, I1] TV0[I0, I1] = TV1[I0, R1, I1]
 */
TensorView* TensorView::rFactor(const std::vector<int> axes) {
  auto domain_pair = domain()->rFactor(axes);
  auto producer_domain = domain_pair.first;
  auto consumer_domain = domain_pair.second;
  FusionGuard fg(this->fusion());
  Expr* origin_expr = this->fusion()->origin(this);
  TORCH_CHECK(
      origin_expr != nullptr && origin_expr->getExprType() == ExprType::ReductionOp,
      "Error rfactoring ",
      this,
      " its origin is either a nullptr or not a reduction.");
  ReductionOp* this_origin = static_cast<ReductionOp*>(origin_expr);

  TensorView* producer = new TensorView(producer_domain, this->getDataType().value());

  this->setDomain(consumer_domain);
  TensorView* consumer = this;
  
  Expr* producer_origin = new ReductionOp(
      this_origin->getReductionOpType(), this_origin->init(), producer, this_origin->in());
  Expr* consumer_origin = new ReductionOp(
      this_origin->getReductionOpType(), this_origin->init(), consumer, producer);
    
  return producer;
}

} // namespace fuser
} // namespace jit
} // namespace torch
