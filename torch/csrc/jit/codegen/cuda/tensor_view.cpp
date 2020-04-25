#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
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


bool TensorView::hasReduction() const { return domain()->hasReduction(); }

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

//TODO: How do we adjust this so we can reduce to a single scalar value?
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

TensorView* TensorView::computeAt(TensorView* consumer, int axis) {
  /*
   * Recursive compute_at:
   * Recurse backward from consumer, to this, make sure there's a dependency
   * chain there. Call ComputeAt for all tensors between this and consumer.
   *
   * Compute at modifies this, not consumer.
   */
  TORCH_CHECK(this->fusion() == consumer->fusion(), this, " and ", consumer, " are not in the same fusion.");
  TORCH_CHECK(
      !this->sameAs(consumer), "Cannot call this->computeAt(this, ...)");
  
  if (axis < 0)
    // Compute at is a bit strange where size is the maximum acceptable value
    // instead of size-1
    axis += int(consumer->nDims()) + 1;

  TORCH_CHECK(
      axis >= 0 && axis < consumer->nDims() + 1,
      "Compute at called on an axis outside valid range.");

  // Direct dependency relationship, run replay.
  if(FusionGuard::getCurFusion()->origin(consumer) != nullptr
     && FusionGuard::getCurFusion()->origin(consumer)->hasInput(this)){
    // Reset view otherwise will conflict with replay.
    this->compute_at_view_ = nullptr;
    this->compute_at_axis_ = -1;
    TransformReplay::replay(consumer, this, axis);
    this->compute_at_view_ = consumer;
    this->compute_at_axis_ = (unsigned int) axis;

    // If another view consumes this, we may be computing this at a position
    // that doesn't match that consumer. Likely producing too little for that
    // next consumer
    for (Expr* other_use : FusionGuard::getCurFusion()->uses(this)) {
      for (Val* maybe_other_consumer : other_use->outputs()) {
        if (*(maybe_other_consumer->getValType()) != ValType::TensorView)
          continue;

        TensorView* other_consumer =
            static_cast<TensorView*>(maybe_other_consumer);
        if (consumer->sameAs(other_consumer))
          continue;

        if (DependencyCheck::isDependencyOf(consumer, other_consumer)) {
          // There seem to be two choices here, either consumer or
          // consumer I believe they end up being equivelent, but uncertain they
          // actually are.
          consumer->computeAt(other_consumer, axis);
        } else {
          other_consumer->computeAt(consumer, axis);
        }
      }
    }
    return this;
  }

  // If not direct relationship follow dependency chain.
  auto dep_chain = DependencyCheck::getDependencyChain(this, consumer);
  if(dep_chain.size() == 0)
    TORCH_CHECK(DependencyCheck::getDependencyChain(consumer, this).size() == 0,
      "Expected ", this, " computeAt ", consumer, " but got the reverse."
    );

  // forward apply to uses of this.
  // Recursively apply replay.
  TensorView* running_consumer = nullptr;

  while (dep_chain.size() > 1) {

    TORCH_INTERNAL_ASSERT(
        dep_chain.top()->getValType() == ValType::TensorView,
        "When following the transform dependency chain, an invalid value was found.");


    running_consumer = static_cast<TensorView*>(dep_chain.top());
    dep_chain.pop();
    
    TORCH_INTERNAL_ASSERT(
        dep_chain.top()->getValType() == ValType::TensorView,
        "When following the transform dependency chain, an invalid value was found.");

    TensorView* running_producer = static_cast<TensorView*>(dep_chain.top());

    running_producer->computeAt(running_consumer, axis);
  }

  return this;
}

TensorView* TensorView::split(int axis, int factor) {
  if (axis < 0)
    axis += domain()->nDims();

  TORCH_CHECK(
      axis >= 0 && axis < domain()->nDims(),
      "Trying to split axis outside of TensorView's range.");

  TORCH_CHECK(
      factor >= 0,
      "Cannot split by a factor less than 1.");


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
TensorView* TensorView::reorder(const std::unordered_map<int, int>& axis2pos_) {
  // START VALIDATION CHECKS
  // adjust based on negative values (any negative values gets nDims added to
  // it)
  std::unordered_map<int, int> axis2pos;
  auto ndims = nDims();
  std::transform(
      axis2pos_.begin(),
      axis2pos_.end(),
      std::inserter(axis2pos, axis2pos.begin()),
      [ndims](std::unordered_map<int, int>::value_type entry) {
        return std::unordered_map<int, int>::value_type({
            entry.first < 0 ? entry.first + ndims : entry.first,
            entry.second < 0 ? entry.second + ndims : entry.second,
        });
      });

  // Check if any adjusted values are < 0, or >= nDims, which are invalid
  bool out_of_range = std::any_of(
      axis2pos.begin(),
      axis2pos.end(),
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
      axis2pos.begin(),
      axis2pos.end(),
      std::inserter(old_pos_set, old_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.first;
      });

  std::set<int> new_pos_set;
  std::transform(
      axis2pos.begin(),
      axis2pos.end(),
      std::inserter(new_pos_set, new_pos_set.begin()),
      [](std::unordered_map<int, int>::value_type entry) {
        return entry.first;
      });

  // Error out if duplicate values are found.
  TORCH_CHECK(
      old_pos_set.size() == axis2pos.size() &&
          new_pos_set.size() == axis2pos.size(),
      "Duplicate entries in transformation map sent to TensorView reorder.");

  // Check if we're trying to reorder any values outside of the computeAt axis

  if (hasComputeAt()) {
    auto compute_at_axis = getComputeAtAxis();
    bool outside_computeat = std::any_of(
        axis2pos.begin(),
        axis2pos.end(),
        [compute_at_axis](std::unordered_map<int, int>::value_type entry) {
          return entry.first < compute_at_axis ||
              entry.second < compute_at_axis;
        });
    TORCH_CHECK(
        !outside_computeat,
        "Cannot reorder dimensions that are outside computeAt axis.");
  }
  // END VALIDATION CHECKS
  setDomain(domain()->reorder(axis2pos_));

  return this;
}


/*
  * Take reduction axes out of this domain, and create a new domain. New domain will be
  * used to create this domain. For example:
  * TV1[I0, I1] = TV0[I0, R0, R1, I1]
  * TV0->rfactor({1})
  * TV0 is transformed to -> TV0[I0, R1, I1]
  * The TensorView returned is:
  * TV2[I0, R0, I3, I1]
  * The reduction will now beset as:
  * TV1[I0, R1, I1] = TV2[I0, R0, I3, I1]
  * TV0[I0, I1] = TV1[I0, R1, I1]
  */
TensorView* TensorView::rfactor(std::vector<int> axes){
  /*
  for(decltype(axes.size()) i{0}; i<axes.size(); i++){
    if(axes[i] < nDims())
      axes[i]+=nDims();
  }
  for(decltype(axes.size()) i{0}; i<axes.size(); i++){
    TORCH_INTERNAL_ASSERT(
        axes[i] >= 0 && axes[i] < nDims(),
        "Expected rfactor axis to be >= 0 and < ",
        nDims(),
        " but got axis value of ",
        axes[i]);
    TORCH_INTERNAL_ASSERT(
        axis(axes[i])->isReduction(),
        "Tried to rfactor our iteration domain ",
        axis(axes[i]),
        " but it is not a reduction axis.");
  }

  std::set<int> axes_set(axes.begin(), axes.end());
  // orig [I0, R0, R1, I1] (factor 1)
  std::vector<IterDomain*> factored; // [I0, R0, I3, I1]
  std::vector<IterDomain*> factored_out; // [I0, R1, I1]

  for(decltype(nDims()) i{0}; i<nDims(); i++){
    if(!axis(i)->isReduction()){
      factored.push_back(axis(i));
      factored_out.push_back(axis(i)->clone());
    }else{
      if(axes_set.find(i) == axes_set.end()){
        // not factoring out i
        factored_out.push_back(axis(i));
        factored.push_back(new IterDomain(
            axis(i)->start(),
            axis(i)->extent(),
            axis(i)->parallel_method(),
            false));
      } else {
        factored.push_back(axis(i));
      }
    }
  }
*/
  return this;
}

} // namespace fuser
} // namespace jit
} // namespace torch
