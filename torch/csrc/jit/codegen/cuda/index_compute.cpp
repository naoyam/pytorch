#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <c10/util/Exception.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>

namespace torch {
namespace jit {
namespace fuser {

namespace {

// A merge is contiguous if:
//   Inputs of outer are to the left in the root domain of the inputs of RHS.
//   All inputs are contiguous in the root domain:
//     - All marked as contiguous
//     - Only gaps between inputs are broadcast or reductoin dims
//   There are no split transformations performed on outer or inner
//   All transformations on outer or inner are contiguous merges
// If this criteria holds, then we can index the input root domains of this
// merge with the indexing provided to the output of the merge in the backward
// index pass

class ContigIDs : public OptInDispatch {
 private:
  using OptInDispatch::handle;

  // Mark if ids are result of contigous merges
  std::unordered_set<kir::IterDomain*> contig_ids;
  // Given contiguous domain, return all iter domains within its history.
  std::unordered_map<kir::IterDomain*, std::unordered_set<kir::IterDomain*>>
      within_contig_ids;
  const std::vector<IterDomain*>& root_domain_;
  const std::vector<bool>& root_contiguity_;
  std::unordered_map<IterDomain*, bool> is_contig_root;

  bool inRoot(const std::vector<IterDomain*>& ids) {
    return std::all_of(ids.begin(), ids.end(), [this](IterDomain* id) {
      return is_contig_root.find(id) != is_contig_root.end();
    });
  }

  bool isContig(kir::IterDomain* id) {
    return contig_ids.find(id) != contig_ids.end();
  }

  // Split outputs are not conitguous, don't need to do anything.
  void handle(Split*) override {}

  void handle(Merge* merge) override {
    // If either input is non-contiguous so is output.
    auto inner = merge->inner();
    auto outer = merge->outer();
    if (!isContig(kir::lowerValue(inner)->as<kir::IterDomain>()) ||
        !isContig(kir::lowerValue(outer)->as<kir::IterDomain>())) {
      return;
    }

    // Grab inputs, make sure they're in root domain, check if they're
    // contiguous.

    auto lhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({outer}, root_domain_);
    auto rhs_inputs =
        ir_utils::iterDomainInputsOfOrderedAs({inner}, root_domain_);

    TORCH_INTERNAL_ASSERT(
        inRoot(lhs_inputs) && inRoot(rhs_inputs),
        "Found an invalid merge operation, inputs of its arguments are not in the root domain.");

    std::deque<IterDomain*> ordered_inputs(
        lhs_inputs.begin(), lhs_inputs.end());
    ordered_inputs.insert(
        ordered_inputs.end(), rhs_inputs.begin(), rhs_inputs.end());

    // If any root input is not contig, output is not contig
    if (!(std::all_of(
            ordered_inputs.begin(),
            ordered_inputs.end(),
            [this](IterDomain* id) {
              return is_contig_root.at(id) && !id->isBroadcast() &&
                  !id->isReduction();
            }))) {
      return;
    }

    std::deque<IterDomain*> root_copy(root_domain_.begin(), root_domain_.end());

    // Forward to first matching argument
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() != ordered_inputs.front()) {
        root_copy.pop_front();
      } else {
        break;
      }
    }

    // Forward through all matching arguments
    while (!root_copy.empty() && !ordered_inputs.empty()) {
      if (root_copy.front() == ordered_inputs.front()) {
        root_copy.pop_front();
        ordered_inputs.pop_front();
        // We probably should be able to make access contiguous through
        // reduction domains, however, for now it's causing issues in predicate
        // generation. See test: ReductionSchedulerMultiDimNonFastest
        //  } else if (
        //     root_copy.front()->isReduction() ||
        //     root_copy.front()->isBroadcast()) {
        //   root_copy.pop_front();
      } else {
        break;
      }
    }

    // If we matched all inputs, the output is contiguous. Only want to keep the
    // top contig ID, lower ids should be placed in the "within_contig_ids" map
    // of top id.
    auto kir_inner = kir::lowerValue(merge->inner())->as<kir::IterDomain>();
    auto kir_outer = kir::lowerValue(merge->outer())->as<kir::IterDomain>();
    auto kir_out = kir::lowerValue(merge->out())->as<kir::IterDomain>();
    if (ordered_inputs.empty()) {
      if (contig_ids.find(kir_inner) != contig_ids.end()) {
        contig_ids.erase(kir_inner);
      }

      if (contig_ids.find(kir_outer) != contig_ids.end()) {
        contig_ids.erase(kir_outer);
      }

      contig_ids.emplace(kir_out);

      std::unordered_set<kir::IterDomain*> within_out;
      within_out.emplace(kir_inner);
      if (within_contig_ids.find(kir_inner) != within_contig_ids.end()) {
        auto in_inner = within_contig_ids.at(kir_inner);
        within_out.insert(in_inner.begin(), in_inner.end());
        within_contig_ids.erase(kir_inner);
      }

      within_out.emplace(kir_outer);
      if (within_contig_ids.find(kir_outer) != within_contig_ids.end()) {
        auto in_outer = within_contig_ids.at(kir_outer);
        within_out.insert(in_outer.begin(), in_outer.end());
        within_contig_ids.erase(kir_outer);
      }

      within_contig_ids[kir_out] = within_out;
    }
  }

 public:
  ContigIDs() = delete;

  // Check through thie history of ids whose inputs map to root_domain with
  // contiguity root_contiguity. Return unordered_set of all merges that are
  // contiguous.
  ContigIDs(
      const std::vector<IterDomain*>& ids,
      const std::vector<IterDomain*>& _root_domain,
      const std::vector<bool>& _root_contiguity)
      : root_domain_(_root_domain), root_contiguity_(_root_contiguity) {
    if (ids.empty()) {
      return;
    }

    TORCH_INTERNAL_ASSERT(
        root_domain_.size() == root_contiguity_.size(),
        "Arguments don't match ",
        root_domain_.size(),
        " != ",
        root_contiguity_.size());

    for (size_t i = 0; i < root_domain_.size(); i++) {
      if (root_contiguity_[i]) {
        auto kir_root_domain_i =
            kir::lowerValue(root_domain_[i])->as<kir::IterDomain>();
        contig_ids.emplace(kir_root_domain_i);
        within_contig_ids[kir_root_domain_i] =
            std::unordered_set<kir::IterDomain*>();
      }
      is_contig_root[root_domain_[i]] = root_contiguity_[i];
    }

    auto exprs = ExprSort::getExprs(ids[0]->fusion(), {ids.begin(), ids.end()});

    for (auto expr : exprs) {
      handle(expr);
    }
  }

  const std::unordered_set<kir::IterDomain*> contigIDs() const {
    return contig_ids;
  }

  const std::
      unordered_map<kir::IterDomain*, std::unordered_set<kir::IterDomain*>>
      withinContigIDs() const {
    return within_contig_ids;
  }
};

} // namespace

void IndexCompute::handle(Split* split) {
  auto in_id = kir::lowerValue(split->in())->as<kir::IterDomain>();
  auto outer_id = kir::lowerValue(split->outer())->as<kir::IterDomain>();
  auto inner_id = kir::lowerValue(split->inner())->as<kir::IterDomain>();

  auto outer_it = index_map_.find(outer_id);
  auto inner_it = index_map_.find(inner_id);
  if (outer_it == index_map_.end() || inner_it == index_map_.end())
    return;

  auto outer_ind = outer_it->second;
  auto inner_ind = inner_it->second;

  bool outer_zero = outer_ind->isZeroInt();
  bool inner_zero = inner_ind->isZeroInt();

  bool outer_bcast = outer_id->isBroadcast();
  bool inner_bcast = inner_id->isBroadcast();

  // Zero inds because a dim is bcast is part of normal traversal, if it's not
  // bcast but is zero ind then it's from local or smem. In the latter case we
  // want to propagate this property.
  if ((outer_zero && !outer_bcast) || (inner_zero && !inner_bcast) ||
      hasZeroMerged(inner_id) || hasZeroMerged(outer_id)) {
    zero_merged_in_.emplace(in_id);
  } else {
    // Maybe clear in_id as it could have been mapped over from another
    // IndexCompute. Uncertain if this is needed but seems to be safe.
    if (hasZeroMerged(in_id)) {
      zero_merged_in_.erase(in_id);
    }
  }

  if (outer_zero && inner_zero) {
    index_map_[in_id] = new kir::Int(0);
  } else if (outer_zero) {
    index_map_[in_id] = inner_ind;
    zero_merged_in_.emplace(in_id);
    extent_map_[in_id] = getExtent(inner_id);
  } else if (inner_zero) {
    index_map_[in_id] = outer_ind;
    zero_merged_in_.emplace(in_id);
    extent_map_[in_id] = getExtent(outer_id);
  } else {
    index_map_[in_id] =
        kir::addExpr(kir::mulExpr(outer_ind, getExtent(inner_id)), inner_ind);
  }
}

void IndexCompute::handle(Merge* merge) {
  auto out_id = kir::lowerValue(merge->out())->as<kir::IterDomain>();
  auto outer_id = kir::lowerValue(merge->outer())->as<kir::IterDomain>();
  auto inner_id = kir::lowerValue(merge->inner())->as<kir::IterDomain>();

  auto out_it = index_map_.find(out_id);
  if (out_it == index_map_.end())
    return;

  auto out_ind = out_it->second;

  auto zero = new kir::Int(0);

  if (out_ind->isZeroInt()) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = zero;
    extent_map_[outer_id] = zero;
    extent_map_[inner_id] = zero;
    return;
  }

  if (!hasZeroMerged(out_id) && contig_ids.find(out_id) != contig_ids.end()) {
    auto input_ids = ir_utils::iterDomainInputsOfOrderedAs(
        {merge->out()}, td_->getRootDomain());

    // Shouldn't hit this, but don't want to segfault if somehow we do.
    TORCH_INTERNAL_ASSERT(!input_ids.empty());

    for (auto root_id : input_ids) {
      index_map_[kir::lowerValue(root_id)->as<kir::IterDomain>()] = zero;
    }

    index_map_[kir::lowerValue(*(input_ids.end() - 1))->as<kir::IterDomain>()] =
        out_ind;
    return;
  }

  Val* inner_extent = getExtent(inner_id);
  Val* outer_extent = getExtent(outer_id);

  if (inner_id->isBroadcast() && inner_extent->isOneInt()) {
    index_map_[outer_id] = out_ind;
    index_map_[inner_id] = zero;

    extent_map_[outer_id] = getExtent(out_id);
  } else if (outer_id->isBroadcast() && outer_extent->isOneInt()) {
    index_map_[outer_id] = zero;
    index_map_[inner_id] = out_ind;

    extent_map_[inner_id] = getExtent(out_id);
  } else if (hasZeroMerged(out_id)) {
    index_map_[inner_id] = out_ind;
    extent_map_[inner_id] = getExtent(out_id);

    index_map_[outer_id] = zero;
    extent_map_[outer_id] = zero;

    zero_merged_in_.emplace(inner_id);
    zero_merged_in_.emplace(outer_id);
  } else {
    Val* I = inner_extent;

    Val* outer_ind = kir::divExpr(out_ind, I);
    Val* inner_ind = kir::modExpr(out_ind, I);

    index_map_[outer_id] = outer_ind;
    index_map_[inner_id] = inner_ind;
  }
}

void IndexCompute::handle(Expr* e) {
  switch (e->getExprType().value()) {
    case (ExprType::Split):
    case (ExprType::Merge):
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Invalid expr type found in transform traversal.");
  }
  BackwardVisitor::handle(e);
}

// Otherwise warning on runBackward as it hides an overloaded virtual
// using TransformIter::runBackward;
IndexCompute::IndexCompute(
    const TensorDomain* _td,
    std::unordered_map<kir::IterDomain*, Val*> initial_index_map,
    std::unordered_map<kir::IterDomain*, Val*> _extent_map,
    std::unordered_set<kir::IterDomain*> _zero_merged_in,
    const std::vector<bool>& root_contiguity)
    : td_(_td),
      index_map_(std::move(initial_index_map)),
      extent_map_(std::move(_extent_map)),
      zero_merged_in_(std::move(_zero_merged_in)) {
  // Make sure we recompute any indices we can that map to a contiguous access
  // in physical memory.
  if (std::any_of(root_contiguity.begin(), root_contiguity.end(), [](bool b) {
        return b;
      })) {
    ContigIDs contig_finder(
        td_->domain(), td_->getRootDomain(), root_contiguity);
    contig_ids = contig_finder.contigIDs();
    auto within_contig = contig_finder.withinContigIDs();
    for (auto contig_id : contig_ids) {
      if (index_map_.find(contig_id) != index_map_.end()) {
        TORCH_INTERNAL_ASSERT(
            within_contig.find(contig_id) != within_contig.end());
        for (auto id : within_contig.at(contig_id)) {
          index_map_.erase(id);
        }
      }
    }
  }

  const std::vector<Val*> domain_vals(
      td_->domain().begin(), td_->domain().end());

  traverseFrom(td_->fusion(), domain_vals, false);
}

Val* IndexCompute::getExtent(kir::IterDomain* id) {
  if (extent_map_.find(id) != extent_map_.end()) {
    return extent_map_.at(id);
  } else {
    return id->extent();
  }
}

bool IndexCompute::hasZeroMerged(kir::IterDomain* id) {
  return zero_merged_in_.find(id) != zero_merged_in_.end();
}

IndexCompute IndexCompute::updateIndexCompute(
    const TensorDomain* new_td,
    const std::unordered_map<IterDomain*, IterDomain*>& id_map,
    std::unordered_map<kir::IterDomain*, Val*> new_index_entries,
    const std::vector<bool>& root_contiguity) {
  std::unordered_map<kir::IterDomain*, Val*> updated_index_map =
      std::move(new_index_entries);
  std::unordered_map<kir::IterDomain*, Val*> updated_extent_map;
  std::unordered_set<kir::IterDomain*> updated_zero_merged_in;

  for (auto id_entry : id_map) {
    kir::IterDomain* prev_id =
        kir::lowerValue(id_entry.first)->as<kir::IterDomain>();
    kir::IterDomain* new_id =
        kir::lowerValue(id_entry.second)->as<kir::IterDomain>();

    if (index_map_.find(prev_id) != index_map_.end()) {
      updated_index_map[new_id] = index_map_.at(prev_id);
    }

    if (extent_map_.find(prev_id) != extent_map_.end()) {
      updated_extent_map[new_id] = extent_map_.at(prev_id);
    }

    if (zero_merged_in_.find(prev_id) != zero_merged_in_.end()) {
      updated_zero_merged_in.emplace(new_id);
    }
  }

  return IndexCompute(
      new_td,
      updated_index_map,
      updated_extent_map,
      updated_zero_merged_in,
      root_contiguity);
}

std::vector<bool> IndexCompute::contiguityAnd(
    const std::vector<bool>& contig1,
    const std::vector<bool>& contig2) {
  TORCH_INTERNAL_ASSERT(
      contig1.size() == contig2.size(),
      "Called contiguityAnd with mismatched vectors.");

  std::vector<bool> contig_result;
  std::transform(
      contig1.begin(),
      contig1.end(),
      contig2.begin(),
      std::back_inserter(contig_result),
      std::logical_and<>());
  return contig_result;
}

// TODO: use new mapping functions
// This mapping might need to go through rfactor, unclear
std::vector<bool> IndexCompute::contiguityPasC(
    TensorDomain* producer,
    TensorDomain* consumer) {
  const std::vector<bool>& producer_contiguity = producer->contiguity();
  std::vector<bool> as_consumer_contiguity;

  auto c_root = consumer->getRootDomain();
  auto p_root = producer->getRootDomain();

  size_t p_ind = 0;
  size_t c_ind = 0;
  while (p_ind < p_root.size()) {
    if (p_root[p_ind]->isReduction()) {
      p_ind++;
    } else if (
        c_root[c_ind]->isBroadcast() &&
        p_root[p_ind]->getIterType() != c_root[c_ind]->getIterType()) {
      c_ind++;
      as_consumer_contiguity.push_back(false);
    } else {
      as_consumer_contiguity.push_back(producer_contiguity[p_ind]);
      c_ind++;
      p_ind++;
    }
  }

  while (c_ind < c_root.size()) {
    as_consumer_contiguity.push_back(false);
    c_ind++;
  }

  return as_consumer_contiguity;
}

namespace {

std::deque<TensorView*> getComputeAtTVStackFrom(TensorView* from_tv) {
  // What's the computeAt root tensor view in this operation
  // This tensor is the terminating tensor in the computeAT dag from consumer
  auto end_tv = from_tv->getComputeAtAxis(0).second;

  // grab all tensor views from producer_tv -> computeAtRoot
  std::deque<TensorView*> tv_stack;

  // Then immediate consumer
  auto running_tv = from_tv;

  // Follow computeAt path until we hit end_tv
  while (running_tv != end_tv) {
    TORCH_INTERNAL_ASSERT(running_tv->hasComputeAt());
    tv_stack.push_front(running_tv);
    running_tv = running_tv->getComputeAtView();
  }

  tv_stack.push_front(end_tv);

  return tv_stack;
}

std::pair<
    std::unordered_map<kir::IterDomain*, Val*>,
    std::unordered_map<kir::IterDomain*, Val*>>
generateIndexAndExtentMap(
    std::deque<TensorView*> c2p_tv_stack,
    std::deque<kir::ForLoop*> loops,
    const std::unordered_map<kir::ForLoop*, Val*>& loop_to_ind_map,
    const std::vector<bool>& last_tv_root_contiguity) {
  if (c2p_tv_stack.empty())
    return std::make_pair(
        std::unordered_map<kir::IterDomain*, Val*>(),
        std::unordered_map<kir::IterDomain*, Val*>());

  // Go through our stack, and map the intermediate IterDomains from common
  // transformations from consumer to producer
  std::deque<std::unordered_map<IterDomain*, IterDomain*>> c2p_ID_maps;
  std::deque<std::unordered_map<IterDomain*, IterDomain*>> p2c_ID_maps;

  // c2p_tv_stack comes in as consumer -> producer
  // Realized we may want to actually do a pass from producer->consumer first to
  // propagate iterators outside the compute at position back into consumers, so
  // we can repropagate back to producer. The need for this was exposed in
  // https://github.com/csarofeen/pytorch/issues/286

  for (size_t i = 0; i + 1 < c2p_tv_stack.size(); i++) {
    auto c_tv = c2p_tv_stack[i];
    auto p_tv = c2p_tv_stack[i + 1];

    // Map root ID's from consumer to producer
    auto c2p_root_map =
        TensorDomain::mapRootCtoP(c_tv->domain(), p_tv->domain());

    // Look for matching ID transformations in producer and consumer...
    BestEffortReplay replay(
        p_tv->domain()->domain(), c_tv->domain()->domain(), c2p_root_map);

    // and grab the intermediate IterDomain map.
    c2p_ID_maps.push_back(replay.getReplay());

    // Something wasn't symmetric when using:
    //
    // auto p2c_root_map = TensorDomain::mapRootPtoC(p_tv->domain(),
    // c_tv->domain());
    //
    // replay = BestEffortReplay(
    //     c_tv->domain()->domain(), p_tv->domain()->domain(), p2c_root_map,
    //     true);

    BestEffortReplay replay_p2c(
        p_tv->domain()->domain(), c_tv->domain()->domain(), c2p_root_map, true);

    std::unordered_map<IterDomain*, IterDomain*> p2c_id_map;

    for (auto ent : replay_p2c.getReplay()) {
      p2c_id_map[ent.second] = ent.first;
    }

    // and grab the intermediate IterDomain map.
    p2c_ID_maps.push_front(p2c_id_map);
  }

  // Maps to be used in the c2p propagation
  std::unordered_map<TensorView*, std::unordered_map<kir::IterDomain*, Val*>>
      p2c_index_maps;

  // PROPAGATE PRODUCER -> CONSUMER START

  std::deque<TensorView*> p2c_tv_stack(
      c2p_tv_stack.rbegin(), c2p_tv_stack.rend());

  // Setup initial IndexCompute:
  auto tv = p2c_tv_stack.front();
  p2c_tv_stack.pop_front();
  auto td = tv->domain()->domain();

  std::vector<kir::IterDomain*> kir_td;

  std::transform(
      td.begin(), td.end(), std::back_inserter(kir_td), [](IterDomain* id) {
        return kir::lowerValue(id)->as<kir::IterDomain>();
      });

  // Map from all IterDomain's to corresponding index as we process each tv in
  // the stack
  std::unordered_map<kir::IterDomain*, Val*> initial_index_map;

  // Match loops to this TV if the loop matchis this TV's ID (could reduce
  // complexity here)

  while (
      !loops.empty() &&
      std::find(kir_td.rbegin(), kir_td.rend(), loops.back()->iter_domain()) !=
          kir_td.rend()) {
    TORCH_INTERNAL_ASSERT(
        loop_to_ind_map.find(loops.back()) != loop_to_ind_map.end());
    initial_index_map[loops.back()->iter_domain()] =
        loop_to_ind_map.at(loops.back());
    loops.pop_back();
  }

  IndexCompute index_compute(
      tv->domain(),
      initial_index_map,
      std::unordered_map<kir::IterDomain*, Val*>(),
      std::unordered_set<kir::IterDomain*>(),
      std::vector<bool>(tv->getRootDomain().size(), false));

  p2c_index_maps[tv] = index_compute.indexMap();

  // Go through the tv entire stack
  while (!p2c_tv_stack.empty()) {
    // Grab the TV
    tv = p2c_tv_stack.front();
    p2c_tv_stack.pop_front();
    td = tv->domain()->domain();
    kir_td.clear();
    std::transform(
        td.begin(), td.end(), std::back_inserter(kir_td), [](IterDomain* id) {
          return kir::lowerValue(id)->as<kir::IterDomain>();
        });

    // Match loops to this TV if the loop matchis this TV's ID (could reduce
    // complexity here)

    // Map from all IterDomain's to corresponding index as we process each tv in
    // the stack
    std::unordered_map<kir::IterDomain*, Val*> new_indices;

    while (!loops.empty() &&
           std::find(
               kir_td.rbegin(), kir_td.rend(), loops.back()->iter_domain()) !=
               kir_td.rend()) {
      TORCH_INTERNAL_ASSERT(
          loop_to_ind_map.find(loops.back()) != loop_to_ind_map.end());
      new_indices[loops.back()->iter_domain()] =
          loop_to_ind_map.at(loops.back());
      loops.pop_back();
    }

    if (!p2c_ID_maps.empty()) {
      index_compute = index_compute.updateIndexCompute(
          tv->domain(),
          p2c_ID_maps.front(),
          new_indices,
          std::vector<bool>(tv->getRootDomain().size(), false));

      p2c_index_maps[tv] = index_compute.indexMap();

      p2c_ID_maps.pop_front();
    }
  }

  // PROPAGATE PRODUCER -> CONSUMER END

  // PROPAGATE CONSUMER -> PRODUCER START

  // Setup initial IndexCompute:
  tv = c2p_tv_stack.front();
  c2p_tv_stack.pop_front();

  // Map from all IterDomain's to corresponding index as we process each tv in
  // the stack
  initial_index_map = p2c_index_maps.at(tv);

  std::unordered_map<kir::IterDomain*, Val*> initial_extent_map;
  if (!c2p_ID_maps.empty()) {
    auto first_id_map = c2p_ID_maps.front();
    for (auto id_entry : first_id_map) {
      kir::IterDomain* this_id =
          kir::lowerValue(id_entry.first)->as<kir::IterDomain>();
      if (initial_extent_map.find(this_id) == initial_extent_map.end()) {
        initial_extent_map[this_id] = this_id->extent();
      }
    }
  }

  index_compute = IndexCompute(
      tv->domain(),
      initial_index_map,
      initial_extent_map,
      std::unordered_set<kir::IterDomain*>(),
      c2p_tv_stack.empty()
          ? last_tv_root_contiguity
          : std::vector<bool>(tv->getRootDomain().size(), false));

  // Go through the tv entire stack
  while (!c2p_tv_stack.empty()) {
    // Grab the TV
    tv = c2p_tv_stack.front();
    c2p_tv_stack.pop_front();

    if (!c2p_ID_maps.empty()) {
      index_compute = index_compute.updateIndexCompute(
          tv->domain(),
          c2p_ID_maps.front(),
          p2c_index_maps.at(tv),
          c2p_tv_stack.empty()
              ? last_tv_root_contiguity
              : std::vector<bool>(tv->getRootDomain().size(), false));

      c2p_ID_maps.pop_front();
    }
  }

  // PROPAGATE CONSUMER -> PRODUCER END

  // Fill in extent map as some mapped indices may not have their extent filled
  // in it, but consumers of this function expect it to be there

  std::unordered_map<kir::IterDomain*, Val*> extent_map(
      index_compute.extentMap());
  for (auto ind_entry : index_compute.indexMap()) {
    auto id = ind_entry.first;
    if (extent_map.find(id) == extent_map.end()) {
      extent_map[id] = id->extent();
    }
  }

  return std::make_pair(index_compute.indexMap(), extent_map);
}

} // namespace

kir::TensorIndex* Index::getGlobalProducerIndex(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  // Replay producer to look like consumer so we can index on producer since our
  // loop nests look like consumer
  auto producerAsC = std::get<0>(TransformReplay::replayPasC(
      producer_tv->domain(), consumer_tv->domain(), consumer_tv->getComputeDomain(), -1));

  // Make the actual producer_tv look like consumer while we do the indexing
  // math in this function
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  std::cerr << "getGlobalProducerIndex: producer: " << producer_tv << std::endl;

  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);
  tv_stack.push_back(producer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;
  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  auto index_map = generateIndexAndExtentMap(
                       tv_stack,
                       std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
                       loop_to_ind_map,
                       producer_tv->domain()->contiguity())
                       .first;

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();

  bool inner_most_dim_contig =
      root_dom[root_dom.size() - 1]->getIterType() == IterType::Iteration &&
      producer_tv->domain()->contiguity()[root_dom.size() - 1];

  // Global striding
  int64_t stride_i = 0;
  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (root_dom[i]->getIterType() == IterType::BroadcastWithStride) {
      stride_i++;
      continue;
    }

    auto kir_root_dom_i = kir::lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        producer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir_root_dom_i,
        " IR root id: ",
        root_dom[i],
        " producer: " ,
        producer_tv);

    auto root_ind = index_map.at(kir_root_dom_i);
    TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ind));

    if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(root_ind);
    } else if (root_ind->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(kir::mulExpr(
          root_ind, new kir::NamedScalar(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new kir::Int(0));

  return new kir::TensorIndex(producer_tv, strided_inds);
}

namespace {

std::unordered_map<kir::ForLoop*, Val*> indexMapFromTV(
    TensorView* tv,
    const std::vector<kir::ForLoop*>& loops) {
  auto alloc_point = loop_utils::getAllocPoint(tv, loops);
  auto alloc_loop = alloc_point.first;

  bool within_alloc = false;
  if (alloc_loop == nullptr) {
    within_alloc = true;
  }

  Val* zero = new kir::Int(0);

  bool is_shared = tv->getMemoryType() == MemoryType::Shared;
  bool is_local = tv->getMemoryType() == MemoryType::Local;

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;

  for (auto loop : loops) {
    if (!within_alloc) {
      loop_to_ind_map[loop] = zero;
    } else if (loop->iter_domain()->isBlockDim() && is_shared) {
      loop_to_ind_map[loop] = zero;
    } else if (loop->iter_domain()->isThread() && is_local) {
      loop_to_ind_map[loop] = zero;
    } else {
      loop_to_ind_map[loop] = loop->index();
    }

    if (!within_alloc && loop == alloc_loop) {
      within_alloc = true;
    }
  }
  return loop_to_ind_map;
}

Val* mulx(Val* v1, Val* v2) {
  if (v1 == nullptr) {
    return v2;
  } else if (v2 == nullptr) {
    return v1;
  } else if (v1->isZeroInt() || v2->isZeroInt()) {
    return new kir::Int(0);
  } else if (v1->isOneInt()) {
    return v2;
  } else if (v2->isOneInt()) {
    return v1;
  } else {
    return kir::mulExpr(v1, v2);
  }
}

Val* divx(Val* v1, Val* v2) {
  if (v1->isZeroInt()) {
    return new kir::Int(0);
  } else {
    return kir::divExpr(v1, v2);
  }
}

Val* modx(Val* v1, Val* v2) {
  if (v1->isZeroInt()) {
    return new kir::Int(0);
  } else {
    return kir::modExpr(v1, v2);
  }
}

Val* addx(Val* v1, Val* v2) {
  if (v1 == nullptr) {
    return v2;
  } else if (v2 == nullptr) {
    return v1;
  } else if (v1->isZeroInt()) {
    return v2;
  } else if (v2->isZeroInt()) {
    return v1;
  } else {
    return kir::addExpr(v1, v2);
  }
}

} // namespace

// Producer index for either shared or local memory
kir::TensorIndex* Index::getProducerIndex_impl(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {

  std::cerr << "getProducerIndex_impl: " << producer_tv << ", "
            << consumer_tv << std::endl;


  // producer_tv->domain() is not replayed as the loop strucutre we were
  // provided, so replay it to match consumer_tv which is.
  auto producerAsC = std::get<0>(TransformReplay::replayPasC(
      producer_tv->domain(), consumer_tv->domain(), consumer_tv->getComputeDomain(), -1));

  // Set producer_tv with the domain replayed as consumer to grab the right
  // indices. The guard will reset the domain when this scope ends.
  ir_utils::TVDomainGuard domain_guard(producer_tv, producerAsC);

  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);
  tv_stack.push_back(producer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map =
      indexMapFromTV(producer_tv, loops);

  auto index_and_extent_map = generateIndexAndExtentMap(
      tv_stack,
      std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
      loop_to_ind_map,
      std::vector<bool>(producer_tv->getRootDomain().size(), false));
  auto index_map = index_and_extent_map.first;
  auto extent_map = index_and_extent_map.second;

  // Indices should now be mapped onto IterDomains in producer, so just grab
  // and use them.
  auto root_dom = producer_tv->getMaybeRFactorDomain();

  std::vector<Val*> strided_inds;

  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast()) {
      continue;
    }

    auto kir_root_dom_i = kir::lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        producer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir_root_dom_i);

    auto root_ind_i = index_map.at(kir_root_dom_i);
    TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ind_i));

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction()) {
        continue;
      }

      auto kir_root_dom_j = kir::lowerValue(root_dom[j])->as<kir::IterDomain>();

      TORCH_INTERNAL_ASSERT(
          index_map.find(kir_root_dom_j) != index_map.end() &&
              extent_map.find(kir_root_dom_j) != extent_map.end(),
          "Couldn't find root mapping for TV",
          consumer_tv->name(),
          " dim: ",
          i,
          " id: ",
          root_dom[i]);

      auto root_ind_j = index_map.at(kir_root_dom_j);
      auto root_ext_j = extent_map.at(kir_root_dom_j);

      TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ext_j));

      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = kir::mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds.push_back(kir::mulExpr(root_ind_i, stride));
    } else {
      strided_inds.push_back(root_ind_i);
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new kir::Int(0));

  return new kir::TensorIndex(producer_tv, strided_inds);
}

namespace {
class IdxGraphNode {
 public:
  explicit IdxGraphNode(Val* idx): idx_(idx) {}
  explicit IdxGraphNode(std::shared_ptr<IdxGraphNode> outer,
                        std::shared_ptr<IdxGraphNode> inner):
      children_({outer, inner}) {}
  //IdxGraphNode(const std::vector<std::shared_ptr<IdxGraphNode>>& children): children_(children) {}

  Val* idx() const {
    //TORCH_INTERNAL_ASSERT(isLeaf());
    if (!isLeaf()) {
      std::stringstream ss;
      print(ss);
      TORCH_INTERNAL_ASSERT(false, "Invalid access to IdxGraphNode: ", ss.str());
    }
    return idx_;
  }

  std::shared_ptr<IdxGraphNode> getInner() const {
    TORCH_INTERNAL_ASSERT(children_.size() == 2);
    return children_[1];
  }
  std::shared_ptr<IdxGraphNode> getOuter() const {
    TORCH_INTERNAL_ASSERT(children_.size() == 2);
    return children_[0];
  }

  bool isLeaf() const {
    bool is_leaf = idx_ != nullptr;
    if (is_leaf) {
      TORCH_INTERNAL_ASSERT(children_.size() == 0);
    }
    return is_leaf;
  }

  std::ostream& print(std::ostream& os) const {
    if (idx_) {
      os << "IdxGraphNode {" << idx_;
      if (idx_->getOrigin()) {
        os << " (" << idx_->getOrigin() << ")";
      }
      os << "}";
    } else {
      os << "IdxGraphNode (";
      for (auto child: children_) {
        os << " ";
        child->print(os);
      }
      os << ")";
    }
    return os;
  }
 private:
  Val* idx_ = nullptr;
  std::vector<std::shared_ptr<IdxGraphNode>> children_;
};

std::ostream& operator<<(std::ostream& os, const IdxGraphNode& info) {
  return info.print(os);
}

class IterDomainInfo {
 public:
  IterDomainInfo(std::shared_ptr<IdxGraphNode> idx, bool is_ca=false):
      idx_(idx), is_ca_(is_ca) {
    validate();
  }
  void validate() const {
    TORCH_INTERNAL_ASSERT(idx_ != nullptr);
    //TORCH_INTERNAL_ASSERT(kir::isLoweredVal(idx_));
    //TORCH_INTERNAL_ASSERT(extent_ != nullptr);
    //TORCH_INTERNAL_ASSERT(kir::isLoweredVal(extent_));
  }
  std::ostream& print(std::ostream& os) const {
    os << "{idx: " << *idx_;
#if 0
    if (extent_) {
      os << ", extent: " << extent_;
      if (extent_->getOrigin()) {
        os << " (" << extent_->getOrigin() << ")";
      }
    }
#endif
    os << ", is CA?: " << (is_ca_ ? "true" : "false");
    os << "}";
    return os;
  }
  std::shared_ptr<IdxGraphNode> idx() const {
    return idx_;
  }
#if 0
  Val* extent() const {
    return extent_;
  }
#endif
  bool isCA() const {
    return is_ca_;
  }
 private:
  std::shared_ptr<IdxGraphNode> idx_ = nullptr;
  //Val* extent_ = nullptr;
  bool is_ca_ = false;
};

std::ostream& operator<<(std::ostream& os, const IterDomainInfo& info) {
  return info.print(os);
}

std::unordered_set<IterDomain*> getMaybeRFactorCAIDs(TensorView* tv) {
  std::vector<IterDomain*> root = tv->getMaybeRFactorDomain();
  std::vector<Val*> ca_ids{tv->domain()->domain().begin(),
                           tv->domain()->domain().begin() + tv->getThisComputeAtAxis()};
  std::unordered_set<Val*> all_CA_id_deps = DependencyCheck::getAllValsBetween(
      {root.begin(), root.end()},
      {ca_ids.begin(), ca_ids.end()});
  std::unordered_set<IterDomain*> root_ca_ids;
  for (IterDomain* id : root) {
    if (all_CA_id_deps.find(id) != all_CA_id_deps.end()) {
      root_ca_ids.emplace(id);
    }
  }
  return root_ca_ids;
}

std::vector<Expr*> getExprsFromRFactorRoot(TensorView* tv,
                                           const std::vector<Val*>& from) {
  // There should be more efficient way to do this.
  std::vector<Expr*> all_exprs = ExprSort::getExprs(tv->fusion(), from);

  std::vector<IterDomain*> rfactor_root = tv->getMaybeRFactorDomain();
  std::unordered_set<Val*> all_deps = DependencyCheck::getAllValsBetween(
      {rfactor_root.begin(), rfactor_root.end()}, from);
  {
    // Just sanity check
    for (auto id: rfactor_root) {
      TORCH_INTERNAL_ASSERT(all_deps.find(id) != all_deps.end());
    }
    for (auto id: from) {
      TORCH_INTERNAL_ASSERT(all_deps.find(id) != all_deps.end());
    }
  }

  std::vector<Expr*> exprs_from_rfactor_root;
  for (auto expr: all_exprs) {
    if (!std::all_of(expr->inputs().begin(), expr->inputs().end(),
                     [&all_deps](Val* val) {
                       return all_deps.find(val) != all_deps.end();
                     })) {
      // some input not found
      continue;
    }
    if (!std::all_of(expr->outputs().begin(), expr->outputs().end(),
                     [&all_deps](Val* val) {
                       return all_deps.find(val) != all_deps.end();
                     })) {
      // some output not found
      continue;
    }
    exprs_from_rfactor_root.push_back(expr);
  }

  return exprs_from_rfactor_root;
}
#if 0
bool isRootTDDomain(const TensorView* tv, const IterDomain* id) {
  const auto cd = tv->getComputeDomain();
  const auto& root = tv->getRootDomain();
  auto td_id = cd->getTensorDomainAxisForDependentAxis(id);
  TORCH_INTERNAL_ASSERT(td_id != nullptr);
  return std::find(root.begin(), root.end(), td_id) != root.end();
}
#endif
class MarkCAIDs: public BackwardVisitor {
 public:
  MarkCAIDs(TensorView* tv, size_t pos): tv_(tv) {
    std::vector<Val*> domain_vals;
    const auto& domain = tv_->domain()->domain();
    for (size_t i = 0; i < domain.size(); ++i) {
      IterDomain* id = domain[i];
      domain_vals.push_back(id->as<Val>());
      ca_ids_.insert({id, i < pos});
    }
    traverseFrom(tv_->fusion(), domain_vals, false);
  }

  using BackwardVisitor::handle;

  void handle(Split* split) override {
    IterDomain* outer = split->outer();
    IterDomain* inner = split->inner();
    bool in_is_ca = isCA(outer) || isCA(inner);
    ca_ids_.insert({split->in(), in_is_ca});
  }

  void handle(Merge* merge) override {
    bool is_ca = isCA(merge->out());
    ca_ids_.insert({merge->inner(), is_ca});
    ca_ids_.insert({merge->outer(), is_ca});
  }

  bool isCA(IterDomain* id) const {
    auto it = ca_ids_.find(id);
    TORCH_INTERNAL_ASSERT(it != ca_ids_.end());
    return it->second;
  }

 private:
  TensorView* tv_;
  std::unordered_map<IterDomain*, bool> ca_ids_;
};

kir::TensorIndex* getProducerIndex_impl2_rfactor(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    bool global) {
  using Idx = std::shared_ptr<IdxGraphNode>;

  const ComputeDomain* consumer_cd = consumer_tv->getComputeDomain();
  const std::vector<IterDomain*>& consumer_root = consumer_tv->getRootDomain();
  const size_t alloc_pos = global ? 0 : producer_tv->getThisComputeAtAxis();

  std::cerr << "getProducerIndex_impl2:\n"
            << "producer: " << producer_tv
            << ", consumer: " << consumer_tv
            << ", global?: " << global
            << std::endl;

  std::cerr << "producer root: " << producer_tv->getRootDomain() << std::endl;
  std::cerr << "producer maybe rfactor: " << producer_tv->getMaybeRFactorDomain() << std::endl;
  std::cerr << "consumer root: " << consumer_root << std::endl;
  std::cerr << "consumer cd: " << *consumer_cd << std::endl;

  // If the producer is computed at -1, no indexing is involved. This
  // is optional; it just skips the rest of the analysis.
  if (alloc_pos == producer_tv->nDims() && !global) {
    return new kir::TensorIndex(producer_tv, {});
  }

  //TORCH_INTERNAL_ASSERT(consumer_tv->hasRFactor());

  std::unordered_map<const IterDomain*, IterDomainInfo> consumer_map;
  for (size_t i = 0; i < consumer_tv->nDims(); ++i) {
    std::cerr << "idx: " << i << std::endl;
    IterDomain* dom = consumer_tv->axis(i);
    std::cerr << "dom: " << dom << std::endl;
    auto cd_axis_idx = consumer_cd->getComputeDomainAxisIndex(i);
    //IterDomain* cd_dom = consumer_cd->getAxisForReplay(cd_axis_idx);
    bool is_ca = i < producer_tv->getRelativeComputeAtAxis() && !global;
    std::shared_ptr<IdxGraphNode> ign;
    if (is_ca) {
      ign = std::make_shared<IdxGraphNode>(new kir::Int(0));
    } else {
      TORCH_INTERNAL_ASSERT(cd_axis_idx < loops.size());
      Val* loop_idx = loops.at(cd_axis_idx)->index();
      ign = std::make_shared<IdxGraphNode>(loop_idx);
    }
    std::cerr << "Initial entry: "
              << dom
              << *ign
              << std::endl;
    consumer_map.insert({dom, IterDomainInfo(ign, is_ca)});
  }

  std::cerr << "Initial consumer_idx_map\n";
  for (auto k: consumer_map) {
    std::cerr << k.first << " -> {" << k.second << "}\n";
  }

  std::vector<Val*> consumer_domain;
  std::transform(consumer_tv->domain()->domain().begin(),
                 consumer_tv->domain()->domain().end(),
                 std::back_inserter(consumer_domain),
                 [](IterDomain* id) {
                   return static_cast<Val*>(id);
                 });

  auto cd_exprs = ExprSort::getExprs(consumer_tv->fusion(), consumer_domain);

  //const auto& cd_exprs = consumer_cd->getExprsToRoot();

  DEBUG("Traversing consumer exprs upward");
  for (auto it = cd_exprs.rbegin(); it != cd_exprs.rend(); ++it) {
    Expr* expr = *it;
    std::cerr << "Traversing " << expr << std::endl;
    if (std::any_of(expr->outputs().begin(), expr->outputs().end(),
                    [&consumer_map](const Val* out) {
                      const auto* output_id = out->as<IterDomain>();
                      return (consumer_map.find(output_id) == consumer_map.end());
                    })) {
      //DEBUG("Ignoring unrelated expression");
      TORCH_INTERNAL_ASSERT(false, "Invalid expr: ", expr);
      continue;
    }
    if (expr->getExprType() == ExprType::Split) {
      Split* split = expr->as<Split>();
      IterDomain* outer = split->outer();
      IterDomain* inner = split->inner();
      auto outer_map = consumer_map.find(outer);
      TORCH_INTERNAL_ASSERT(outer_map != consumer_map.end(),
                            "Outer ID not found: ", outer);
      auto inner_map = consumer_map.find(inner);
      TORCH_INTERNAL_ASSERT(inner_map != consumer_map.end(),
                            "Inner ID not found: ", inner);
      const bool outer_ca = outer_map->second.isCA();
      const bool inner_ca = inner_map->second.isCA();
      const bool is_ca = outer_ca || inner_ca;
      //IterDomain* split_in =
      //consumer_cd->getAxisForReplay(split->in());
      IterDomain* split_in = split->in();

      auto outer_idx = outer_map->second.idx();
      auto inner_idx = inner_map->second.idx();


      //Val* in_idx = addx(mulx(outer_idx, inner_extent), inner_idx);
      Idx in_idx;

      if (outer_ca && !inner_ca) {
        DEBUG("Split: only outer is in CA");
        in_idx = std::make_shared<IdxGraphNode>(outer_idx, inner_idx);
      } else if (!outer_ca && inner_ca) {
        // This should not happen.
        TORCH_INTERNAL_ASSERT(false, "Split inner is CA but not outer");
      } else if (outer_ca && inner_ca) {
        DEBUG("Split: both in CA");
        in_idx = std::make_shared<IdxGraphNode>(outer_idx, inner_idx);
      } else {
        DEBUG("Split: none in CA");
        Val* inner_extent = kir::lowerValue(inner->extent());
        in_idx = std::make_shared<IdxGraphNode>(
            addx(mulx(outer_idx->idx(), inner_extent), inner_idx->idx()));
      }
      DEBUG("Inserting new map: ", split_in, " to ", in_idx,
            " (original split in: ", split->in(), ")");
      consumer_map.insert({split_in, IterDomainInfo(in_idx, is_ca)});
      consumer_map.erase(outer);
      consumer_map.erase(inner);
    } else if (expr->getExprType() == ExprType::Merge) {
      Merge* merge = expr->as<Merge>();
      auto out_map = consumer_map.find(merge->out());
      TORCH_INTERNAL_ASSERT(out_map != consumer_map.end());
      Idx out_idx = out_map->second.idx();

      //IterDomain* outer_id =
      //consumer_cd->getAxisForReplay(merge->outer());
      IterDomain* outer_id = merge->outer();
      //IterDomain* inner_id =
      //consumer_cd->getAxisForReplay(merge->inner());
      IterDomain* inner_id = merge->inner();
      Idx inner_idx, outer_idx;
      bool is_ca = out_map->second.isCA();
      if (is_ca) {
        DEBUG("Merge: out is in CA");
        inner_idx = out_idx;
        outer_idx = out_idx;
      } else {
        DEBUG("Merge: not in CA");
        Val* inner_extent = kir::lowerValue(merge->inner()->extent());
        inner_idx = std::make_shared<IdxGraphNode>(modx(out_idx->idx(), inner_extent));
        outer_idx = std::make_shared<IdxGraphNode>(divx(out_idx->idx(), inner_extent));
      }
      consumer_map.insert({outer_id, IterDomainInfo(outer_idx, is_ca)});
      DEBUG("Inserting new map: ", outer_id, " to ", outer_idx,
            " (original id: ", merge->outer(), ")");
      consumer_map.insert({inner_id, IterDomainInfo(inner_idx, is_ca)});
      DEBUG("Inserting new map: ", inner_id, " to ", inner_idx,
            " (original id: ", merge->inner(), ")");
      consumer_map.erase(merge->out());
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unexpected expr: ", expr);
    }
  }

  DEBUG("Root consumer_map");
  //std::unordered_map<const IterDomain*, IterDomainInfo> consumer_td_map;
  for (auto it = consumer_map.begin(); it != consumer_map.end(); ++it) {
    DEBUG(const_cast<IterDomain*>(it->first), " -> ", it->second);
#if 0
    auto td_axis = consumer_cd->getTensorDomainAxisForDependentAxis(it->first);
    if (td_axis == nullptr) {
      DEBUG("Axis not exist in TD; must be a broadcast domain and safe to ignore: ",
            const_cast<IterDomain*>(it->first));
      continue;
    }
    consumer_td_map.insert({td_axis, it->second});
    std::stringstream ss;
    ss << td_axis;
    DEBUG("Adding a root consumer map: ", ss.str());
#endif
  }

  std::unordered_map<IterDomain*, IterDomainInfo> producer_map;
  const auto root_mapping = TensorDomain::mapRootPandC(producer_tv->domain(),
                                                       consumer_tv->domain());
  auto root_ca_ids = getMaybeRFactorCAIDs(producer_tv);

  // Propagate consumer root to producer toot
  for (auto producer_root_id: producer_tv->getMaybeRFactorDomain()) {
    DEBUG("Producer root: ", producer_root_id);
    auto consumer_root_id = std::find_if(root_mapping.begin(), root_mapping.end(),
                                         [producer_root_id](const auto& pair) {
                                           return pair.first == producer_root_id;
                                         });
    if (consumer_root_id == root_mapping.end()) {
      TORCH_INTERNAL_ASSERT(producer_root_id->isReduction());
      continue;
    }
    DEBUG("Corresponding consumer TD root ID: ",
          const_cast<IterDomain*>(consumer_root_id->second));
    auto idx_map = consumer_map.find(consumer_root_id->second);
    TORCH_INTERNAL_ASSERT(idx_map != consumer_map.end(),
                          "Consumer root not found: ",
                          const_cast<IterDomain*>(consumer_root_id->second));
    const IterDomainInfo& info = idx_map->second;
    Idx producer_idx = info.idx();
    if (root_ca_ids.find(producer_root_id) == root_ca_ids.end() &&
        producer_root_id->isBroadcast()) {
      producer_idx = std::make_shared<IdxGraphNode>(new kir::Int(0));
    }
    IterDomainInfo producer_info{producer_idx};
    std::cerr << "Producer root dom: " << producer_root_id
              << ", " << producer_info << std::endl;
    producer_map.insert({producer_root_id, producer_info});
  }

  if (global) {
    // ".stride" is used here. Is it defined for an Rfactor tensor?
    // Assume not, so fail if this is an rfactor tensor.
    TORCH_INTERNAL_ASSERT(!producer_tv->hasRFactor(), "Invalid rfactor tensor.");
    const auto& producer_root = producer_tv->getRootDomain();
    std::vector<Val*> strided_inds;
    for (size_t i = 0; i < producer_root.size(); ++i) {
      IterDomain* id = producer_root[i];
      const IterDomainInfo& info = producer_map.at(id);
      std::cerr << "Global Producer root: " << id << std::endl;
      if (id->isReduction()) {
        std::cerr << "global root is reduction; ignored\n";
        continue;
      }
      if (id->isBroadcast()) {
        std::cerr << "global root is broadcast; ignored\n";
        continue;
      }
      auto idx = info.idx();
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << i << "]";
      strided_inds.push_back(
          mulx(idx->idx(), new kir::NamedScalar(ss.str(), DataType::Int)));
    }
    std::cerr << "Global getProducerIndex_impl2: Strided indices: " << strided_inds << std::endl;
    auto ti = new kir::TensorIndex(producer_tv, strided_inds);
    DEBUG("Generated TensorIndex: ", ti);
    return ti;
  }

  // Transform to the original producer domain
  std::vector<Val*> producer_domain;
  std::transform(producer_tv->domain()->domain().begin(),
                 producer_tv->domain()->domain().end(),
                 std::back_inserter(producer_domain),
                 [](IterDomain* dom) {
                   return dom->as<Val>();
                 });

  //auto producer_exprs = ExprSort::getExprs(producer_tv->fusion(),
  //producer_domain);
  auto producer_exprs = getExprsFromRFactorRoot(producer_tv, producer_domain);
  MarkCAIDs producer_ca_ids(producer_tv, alloc_pos);

  for (auto expr: producer_exprs) {
    std::cerr << "Producer expr: " << expr;
    if (expr->getExprType() == ExprType::Split) {
      Split* split = expr->as<Split>();
      IterDomain* in = split->in();
      if (producer_map.find(in) == producer_map.end()) {
        TORCH_INTERNAL_ASSERT(false);
        continue;
      }
      const auto& in_info = producer_map.find(in)->second;
      TORCH_INTERNAL_ASSERT(in_info.idx() != nullptr);
      Idx inner_idx;
      Idx outer_idx;
      if (producer_ca_ids.isCA(in)) {
        outer_idx = in_info.idx()->getOuter();
        inner_idx = in_info.idx()->getInner();
      } else {
        inner_idx = std::make_shared<IdxGraphNode>(modx(in_info.idx()->idx(), kir::lowerValue(split->factor())));
        outer_idx = std::make_shared<IdxGraphNode>(divx(in_info.idx()->idx(), kir::lowerValue(split->factor())));
      }
      producer_map.insert({split->outer(),
                           IterDomainInfo(outer_idx)});
      producer_map.insert({split->inner(),
                           IterDomainInfo(inner_idx)});
    } else if (expr->getExprType() == ExprType::Merge) {
      Merge* merge = expr->as<Merge>();
      TORCH_INTERNAL_ASSERT(producer_map.find(merge->inner()) != producer_map.end(),
                            "Merge inner not found: ", merge->inner());
      TORCH_INTERNAL_ASSERT(producer_map.find(merge->outer()) != producer_map.end(),
                            "Merge outer not found: ", merge->outer());
      const auto& inner = producer_map.find(merge->inner())->second;
      const auto& outer = producer_map.find(merge->outer())->second;
      //Val* out_idx = addx(mulx(outer.idx(), inner.extent()),
      //inner.idx());
      Idx out_idx;
      if (producer_ca_ids.isCA(merge->out())) {
        out_idx = inner.idx();
        std::cerr << "inner.idx: " << *(inner.idx()) << std::endl;
        std::cerr << "outer.idx: " << *(outer.idx()) << std::endl;
        TORCH_INTERNAL_ASSERT(inner.idx() == outer.idx());
      } else {
        out_idx = std::make_shared<IdxGraphNode>(
            addx(mulx(outer.idx()->idx(),
                      kir::lowerValue(merge->inner()->extent())), inner.idx()->idx()));
      }
      producer_map.insert({merge->out(), IterDomainInfo(out_idx)});
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unexpected expr: ", expr);
    }
  }

  std::vector<Val*> strided_inds;
  for (size_t i = alloc_pos; i < producer_tv->nDims(); ++i) {
    IterDomain* prod_id = producer_tv->axis(i);
    if (prod_id->isReduction()) {
      // this should not appear at the consumer
      continue;
    }
    if (producer_map.find(prod_id) == producer_map.end()) {
      // If the index is determined to be 0, no mapping entry is
      // created.
      TORCH_INTERNAL_ASSERT(false, "Mapping not detected for ", prod_id);
      continue;
    }
    if (prod_id->isThread()) {
      continue;
    }
    Idx idx = producer_map.find(prod_id)->second.idx();
    TORCH_INTERNAL_ASSERT(idx->idx() != nullptr);
    std::cerr << "Prod idx: " << idx->idx();
    if (idx->idx()->getOrigin()) {
      std::cerr << " (" << idx->idx()->getOrigin() << ")";
    }
    std::cerr << std::endl;
    Val* extent = nullptr;
    for (size_t j = i + 1; j < producer_tv->nDims(); ++j) {
      IterDomain* id = producer_tv->axis(j);
      if (id->isThread() || id->isReduction()) {
        continue;
      }
      extent = mulx(extent, kir::lowerValue(id->extent()));
    }
    strided_inds.push_back(mulx(idx->idx(), extent));
  }

  std::cerr << "getProducerIndex_impl2: Strided indices: " << strided_inds << std::endl;

  auto ti = new kir::TensorIndex(producer_tv, strided_inds);
  DEBUG("Generated TensorIndex: ", ti);
  return ti;
}

} // namespace

kir::TensorIndex* Index::getProducerIndex_impl2(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  using Idx = std::shared_ptr<IdxGraphNode>;
  const bool global = producer_tv->getMemoryType() == MemoryType::Global;

  if (consumer_tv->hasRFactor()) {
    return getProducerIndex_impl2_rfactor(producer_tv, consumer_tv, loops, global);
  }

  const ComputeDomain* consumer_cd = consumer_tv->getComputeDomain();
  const std::vector<IterDomain*>& consumer_root = consumer_tv->getRootDomain();
  const size_t alloc_pos = global ? 0 : producer_tv->getThisComputeAtAxis();

  std::cerr << "getProducerIndex_impl2:\n"
            << "\tproducer: " << producer_tv << "\n"
            << "\tconsumer: " << consumer_tv << "\n"
            << ", global?: " << global << "\n"
            << ", alloc_pos: " << alloc_pos << "\n";

  std::cerr << "producer root: " << producer_tv->getRootDomain() << std::endl;
  std::cerr << "producer maybe rfactor: " << producer_tv->getMaybeRFactorDomain() << std::endl;
  std::cerr << "consumer root: " << consumer_root << std::endl;
  std::cerr << "consumer cd: " << *consumer_cd << std::endl;
  for (auto m: consumer_cd->crossoverMap()) {
    std::cerr << "crossover map: " << m.first << " -> " << m.second << std::endl;
  }

  // If the producer is computed at -1, no indexing is involved. This
  // is optional; it just skips the rest of the analysis.
  if (alloc_pos == producer_tv->nDims() && !global) {
    return new kir::TensorIndex(producer_tv, {});
  }

  std::unordered_map<const IterDomain*, IterDomainInfo> consumer_map;
  for (size_t i = 0; i < consumer_tv->nDims(); ++i) {
    std::cerr << "idx: " << i << std::endl;
    IterDomain* dom = consumer_tv->axis(i);
    std::cerr << "dom: " << dom << std::endl;
    auto cd_axis_idx = consumer_cd->getComputeDomainAxisIndex(i);
    IterDomain* cd_dom = consumer_cd->getAxisForReplay(cd_axis_idx);
    bool is_ca = i < producer_tv->getRelativeComputeAtAxis() && !global;
    std::shared_ptr<IdxGraphNode> ign;
    if (is_ca) {
      ign = std::make_shared<IdxGraphNode>(new kir::Int(0));
    } else {
      TORCH_INTERNAL_ASSERT(cd_axis_idx < loops.size());
      Val* loop_idx = loops.at(cd_axis_idx)->index();
      ign = std::make_shared<IdxGraphNode>(loop_idx);
    }
    std::cerr << "Initial entry: "
              << dom
              << ", cd axis: " << cd_dom
              << *ign
              << std::endl;
    consumer_map.insert({cd_dom, IterDomainInfo(ign, is_ca)});
  }

  std::cerr << "Initial consumer_idx_map\n";
  for (auto k: consumer_map) {
    std::cerr << k.first << " -> {" << k.second << "}\n";
  }

  std::vector<Val*> consumer_domain;
  std::transform(consumer_tv->domain()->domain().begin(),
                 consumer_tv->domain()->domain().end(),
                 std::back_inserter(consumer_domain),
                 [](IterDomain* id) {
                   return static_cast<Val*>(id);
                 });

  const auto& cd_exprs = consumer_cd->getExprsToRoot();

  DEBUG("Traversing consumer exprs upward");
  for (auto it = cd_exprs.begin(); it != cd_exprs.end(); ++it) {
    Expr* expr = *it;
    std::cerr << "Traversing " << expr << std::endl;
    if (std::any_of(expr->outputs().begin(), expr->outputs().end(),
                    [&consumer_map](const Val* out) {
                      const auto* output_id = out->as<IterDomain>();
                      return (consumer_map.find(output_id) == consumer_map.end());
                    })) {
      DEBUG("Ignoring unrelated expression");
      continue;
    }
    if (expr->getExprType() == ExprType::Split) {
      Split* split = expr->as<Split>();
      IterDomain* outer = split->outer();
      IterDomain* inner = split->inner();
      auto outer_map = consumer_map.find(outer);
      TORCH_INTERNAL_ASSERT(outer_map != consumer_map.end(),
                            "Outer ID not found: ", outer);
      auto inner_map = consumer_map.find(inner);
      TORCH_INTERNAL_ASSERT(inner_map != consumer_map.end(),
                            "Inner ID not found: ", inner);
      const bool outer_ca = outer_map->second.isCA();
      const bool inner_ca = inner_map->second.isCA();
      const bool is_ca = outer_ca || inner_ca;
      IterDomain* split_in = consumer_cd->getAxisForReplay(split->in());

      auto outer_idx = outer_map->second.idx();
      auto inner_idx = inner_map->second.idx();


      //Val* in_idx = addx(mulx(outer_idx, inner_extent), inner_idx);
      Idx in_idx;

      if (outer_ca && !inner_ca) {
        DEBUG("Split: only outer is in CA");
        in_idx = std::make_shared<IdxGraphNode>(outer_idx, inner_idx);
      } else if (!outer_ca && inner_ca) {
        // This should not happen.
        TORCH_INTERNAL_ASSERT(false, "Split inner is CA but not outer");
      } else if (outer_ca && inner_ca) {
        DEBUG("Split: both in CA");
        in_idx = std::make_shared<IdxGraphNode>(outer_idx, inner_idx);
      } else {
        DEBUG("Split: none in CA");
        Val* inner_extent = kir::lowerValue(inner->extent());
        in_idx = std::make_shared<IdxGraphNode>(
            addx(mulx(outer_idx->idx(), inner_extent), inner_idx->idx()));
      }
      DEBUG("Inserting new map: ", split_in, " to ", *in_idx,
            " (original split in: ", split->in(), ")");
      consumer_map.insert({split_in, IterDomainInfo(in_idx, is_ca)});
      consumer_map.erase(outer);
      consumer_map.erase(inner);
    } else if (expr->getExprType() == ExprType::Merge) {
      Merge* merge = expr->as<Merge>();
      auto out_map = consumer_map.find(merge->out());
      TORCH_INTERNAL_ASSERT(out_map != consumer_map.end());
      Idx out_idx = out_map->second.idx();

      IterDomain* outer_id = consumer_cd->getAxisForReplay(merge->outer());
      IterDomain* inner_id = consumer_cd->getAxisForReplay(merge->inner());
      Idx inner_idx, outer_idx;
      bool is_ca = out_map->second.isCA();
      if (is_ca) {
        inner_idx = out_idx;
        outer_idx = out_idx;
      } else {
        Val* inner_extent = kir::lowerValue(merge->inner()->extent());
        inner_idx = std::make_shared<IdxGraphNode>(modx(out_idx->idx(), inner_extent));
        outer_idx = std::make_shared<IdxGraphNode>(divx(out_idx->idx(), inner_extent));
      }
      consumer_map.insert({outer_id, IterDomainInfo(outer_idx, is_ca)});
      DEBUG("Inserting new map for outer: ", outer_id, " to ", *outer_idx,
            " (original id: ", merge->outer(), ")");
      consumer_map.insert({inner_id, IterDomainInfo(inner_idx, is_ca)});
      DEBUG("Inserting new map for inner: ", inner_id, " to ", *inner_idx,
            " (original id: ", merge->inner(), ")");
      consumer_map.erase(merge->out());
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unexpected expr: ", expr);
    }
  }

  DEBUG("Root consumer_map");
  std::unordered_map<const IterDomain*, IterDomainInfo> consumer_td_map;
  for (auto it = consumer_map.begin(); it != consumer_map.end(); ++it) {
    DEBUG(const_cast<IterDomain*>(it->first), " -> ", it->second);
    auto td_axis = consumer_cd->getTensorDomainAxisForDependentAxis(it->first);
    if (td_axis == nullptr) {
      DEBUG("Axis not exist in TD; must be a broadcast domain and safe to ignore: ",
            const_cast<IterDomain*>(it->first));
      continue;
    }
    consumer_td_map.insert({td_axis, it->second});
    std::stringstream ss;
    ss << td_axis;
    DEBUG("Adding a root consumer map: ", ss.str());
  }

  std::unordered_map<IterDomain*, IterDomainInfo> producer_map;
  const auto root_mapping = TensorDomain::mapRootPandC(producer_tv->domain(),
                                                       consumer_tv->domain());
  auto root_ca_ids = getMaybeRFactorCAIDs(producer_tv);

  // Propagate consumer root to producer toot
  for (auto producer_root_id: producer_tv->getMaybeRFactorDomain()) {
    DEBUG("Producer root: ", producer_root_id);
    auto consumer_root_id = std::find_if(root_mapping.begin(), root_mapping.end(),
                                         [producer_root_id](const auto& pair) {
                                           return pair.first == producer_root_id;
                                         });
    if (consumer_root_id == root_mapping.end()) {
      TORCH_INTERNAL_ASSERT(producer_root_id->isReduction());
      continue;
    }
    DEBUG("Corresponding consumer TD root ID: ",
          const_cast<IterDomain*>(consumer_root_id->second));
    auto idx_map = consumer_td_map.find(consumer_root_id->second);
    TORCH_INTERNAL_ASSERT(idx_map != consumer_td_map.end(),
                          "Consumer root not found: ",
                          const_cast<IterDomain*>(consumer_root_id->second));
    const IterDomainInfo& info = idx_map->second;
    Idx producer_idx = info.idx();
    if (root_ca_ids.find(producer_root_id) == root_ca_ids.end() &&
        producer_root_id->isBroadcast()) {
      producer_idx = std::make_shared<IdxGraphNode>(new kir::Int(0));
    }
    IterDomainInfo producer_info{producer_idx};
    std::cerr << "Producer root dom: " << producer_root_id
              << ", " << producer_info
              << std::endl;
    producer_map.insert({producer_root_id, producer_info});
  }

  if (global) {
    // ".stride" is used here. Is it defined for an Rfactor tensor?
    // Assume not, so fail if this is an rfactor tensor.
    TORCH_INTERNAL_ASSERT(!producer_tv->hasRFactor(), "Invalid rfactor tensor.");
    const auto& producer_root = producer_tv->getRootDomain();
    std::vector<Val*> strided_inds;
    for (size_t i = 0; i < producer_root.size(); ++i) {
      IterDomain* id = producer_root[i];
      const IterDomainInfo& info = producer_map.at(id);
      std::cerr << "Global Producer root: " << id << std::endl;
      if (id->isReduction()) {
        std::cerr << "global root is reduction; ignored\n";
        continue;
      }
      if (id->isBroadcast()) {
        std::cerr << "global root is broadcast; ignored\n";
        continue;
      }
      auto idx = info.idx();
      std::stringstream ss;
      ss << "T" << producer_tv->name() << ".stride[" << i << "]";
      strided_inds.push_back(
          mulx(idx->idx(), new kir::NamedScalar(ss.str(), DataType::Int)));
    }
    std::cerr << "Global getProducerIndex_impl2: Strided indices: " << strided_inds << std::endl;
    auto ti = new kir::TensorIndex(producer_tv, strided_inds);
    DEBUG("Generated TensorIndex: ", ti);
    return ti;
  }

  // Transform to the original producer domain
  std::vector<Val*> producer_domain;
  std::transform(producer_tv->domain()->domain().begin(),
                 producer_tv->domain()->domain().end(),
                 std::back_inserter(producer_domain),
                 [](IterDomain* dom) {
                   return dom->as<Val>();
                 });

  //auto producer_exprs = ExprSort::getExprs(producer_tv->fusion(),
  //producer_domain);
  auto producer_exprs = getExprsFromRFactorRoot(producer_tv, producer_domain);
  MarkCAIDs producer_ca_ids(producer_tv, alloc_pos);

  for (auto expr: producer_exprs) {
    std::cerr << "Producer expr: " << expr;
    if (expr->getExprType() == ExprType::Split) {
      Split* split = expr->as<Split>();
      IterDomain* in = split->in();
      if (producer_map.find(in) == producer_map.end()) {
        TORCH_INTERNAL_ASSERT(false);
        continue;
      }
      const auto& in_info = producer_map.find(in)->second;
      TORCH_INTERNAL_ASSERT(in_info.idx() != nullptr);
      Idx inner_idx;
      Idx outer_idx;
      if (producer_ca_ids.isCA(in)) {
        outer_idx = in_info.idx()->getOuter();
        inner_idx = in_info.idx()->getInner();
      } else {
        inner_idx = std::make_shared<IdxGraphNode>(modx(in_info.idx()->idx(), kir::lowerValue(split->factor())));
        outer_idx = std::make_shared<IdxGraphNode>(divx(in_info.idx()->idx(), kir::lowerValue(split->factor())));
      }
      producer_map.insert({split->outer(),
                           IterDomainInfo(outer_idx)});
      producer_map.insert({split->inner(),
                           IterDomainInfo(inner_idx)});
    } else if (expr->getExprType() == ExprType::Merge) {
      Merge* merge = expr->as<Merge>();
      TORCH_INTERNAL_ASSERT(producer_map.find(merge->inner()) != producer_map.end(),
                            "Merge inner not found: ", merge->inner());
      TORCH_INTERNAL_ASSERT(producer_map.find(merge->outer()) != producer_map.end(),
                            "Merge outer not found: ", merge->outer());
      const auto& inner = producer_map.find(merge->inner())->second;
      const auto& outer = producer_map.find(merge->outer())->second;
      //Val* out_idx = addx(mulx(outer.idx(), inner.extent()),
      //inner.idx());
      Idx out_idx;
      if (producer_ca_ids.isCA(merge->out())) {
        out_idx = inner.idx();
        std::cerr << "inner.idx: " << *(inner.idx()) << std::endl;
        std::cerr << "outer.idx: " << *(outer.idx()) << std::endl;
        TORCH_INTERNAL_ASSERT(inner.idx() == outer.idx());
      } else {
        out_idx = std::make_shared<IdxGraphNode>(
            addx(mulx(outer.idx()->idx(),
                      kir::lowerValue(merge->inner()->extent())), inner.idx()->idx()));
      }
      producer_map.insert({merge->out(), IterDomainInfo(out_idx)});
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unexpected expr: ", expr);
    }
  }

  std::vector<Val*> strided_inds;
  for (size_t i = alloc_pos; i < producer_tv->nDims(); ++i) {
    IterDomain* prod_id = producer_tv->axis(i);
    if (prod_id->isReduction()) {
      // this should not appear at the consumer
      continue;
    }
    if (producer_map.find(prod_id) == producer_map.end()) {
      // If the index is determined to be 0, no mapping entry is
      // created.
      TORCH_INTERNAL_ASSERT(false, "Mapping not detected for ", prod_id);
      continue;
    }
    if (prod_id->isThread()) {
      continue;
    }
    Idx idx = producer_map.find(prod_id)->second.idx();
    TORCH_INTERNAL_ASSERT(idx->idx() != nullptr);
    std::cerr << "Prod idx: " << idx->idx();
    if (idx->idx()->getOrigin()) {
      std::cerr << " (" << idx->idx()->getOrigin() << ")";
    }
    std::cerr << std::endl;
    Val* extent = nullptr;
    for (size_t j = i + 1; j < producer_tv->nDims(); ++j) {
      IterDomain* id = producer_tv->axis(j);
      if (id->isThread() || id->isReduction()) {
        continue;
      }
      extent = mulx(extent, kir::lowerValue(id->extent()));
    }
    strided_inds.push_back(mulx(idx->idx(), extent));
  }

  std::cerr << "getProducerIndex_impl2: Strided indices: " << strided_inds << std::endl;

  auto ti = new kir::TensorIndex(producer_tv, strided_inds);
  DEBUG("Generated TensorIndex: ", ti);
  return ti;
}


kir::TensorIndex* Index::getGlobalConsumerIndex(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;
  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  auto index_map = generateIndexAndExtentMap(
                       tv_stack,
                       std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
                       loop_to_ind_map,
                       consumer_tv->domain()->contiguity())
                       .first;

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  bool inner_most_dim_contig =
      root_dom[root_dom.size() - 1]->getIterType() == IterType::Iteration &&
      consumer_tv->domain()->contiguity()[root_dom.size() - 1];

  int64_t stride_i = 0;
  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isReduction() ||
        root_dom[i]->getIterType() == IterType::BroadcastWithoutStride) {
      continue;
    } else if (root_dom[i]->getIterType() == IterType::BroadcastWithStride) {
      stride_i++;
      continue;
    }

    auto kir_root_dom_i = kir::lowerValue(root_dom[i])->as<kir::IterDomain>();

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        consumer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir_root_dom_i);
    auto ind = index_map.at(kir_root_dom_i);

    if (i == root_dom.size() - 1 && inner_most_dim_contig) {
      strided_inds.push_back(ind);
    } else if (ind->isZeroInt()) {
      stride_i++;
    } else {
      std::stringstream ss;
      ss << "T" << consumer_tv->name() << ".stride[" << stride_i++ << "]";
      strided_inds.push_back(
          kir::mulExpr(ind, new kir::NamedScalar(ss.str(), DataType::Int)));
    }
  }

  if (strided_inds.size() == 0)
    strided_inds.push_back(new kir::Int(0));

  auto ti = new kir::TensorIndex(consumer_tv, strided_inds);
  DEBUG("Generated TI: ", ti);
  return ti;
}

// Consumer index for either shared or local memory
kir::TensorIndex* Index::getConsumerIndex_impl(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {
  // grab all tensor views from consumer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map =
      indexMapFromTV(consumer_tv, loops);

  auto index_and_extent_map = generateIndexAndExtentMap(
      tv_stack,
      std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
      loop_to_ind_map,
      std::vector<bool>(consumer_tv->getRootDomain().size(), false));

  auto index_map = index_and_extent_map.first;
  auto extent_map = index_and_extent_map.second;

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.
  auto root_dom = consumer_tv->getMaybeRFactorDomain();

  std::vector<Val*> strided_inds;
  for (size_t i = 0; i < root_dom.size(); i++) {
    std::cerr << "root dom: " << root_dom[i] << std::endl;
    if (root_dom[i]->isReduction() || root_dom[i]->isBroadcast()) {
      continue;
    }

    auto kir_root_dom_i = kir::lowerValue(root_dom[i])->as<kir::IterDomain>();

    std::cerr << "root dom (lowered): " << kir_root_dom_i << std::endl;

    TORCH_INTERNAL_ASSERT(
        index_map.find(kir_root_dom_i) != index_map.end(),
        "Couldn't find root mapping for TV",
        consumer_tv->name(),
        " dim: ",
        i,
        " id: ",
        kir_root_dom_i);
    auto root_ind_i = index_map.at(kir_root_dom_i);
    TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ind_i));

    if (root_ind_i->isZeroInt()) {
      continue;
    }

    // Compute striding for this index.
    Val* stride = nullptr;
    for (size_t j = i + 1; j < root_dom.size(); j++) {
      if (root_dom[j]->isBroadcast() || root_dom[j]->isReduction()) {
        continue;
      }

      auto kir_root_dom_j = kir::lowerValue(root_dom[j])->as<kir::IterDomain>();

      TORCH_INTERNAL_ASSERT(
          index_map.find(kir_root_dom_j) != index_map.end() &&
              extent_map.find(kir_root_dom_j) != extent_map.end(),
          "Couldn't find root mapping for TV",
          consumer_tv->name(),
          " dim: ",
          i,
          " id: ",
          root_dom[i]);

      auto root_ind_j = index_map.at(kir_root_dom_j);
      auto root_ext_j = extent_map.at(kir_root_dom_j);
      TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(root_ext_j));
      if (!root_ind_j->isZeroInt()) {
        if (stride == nullptr) {
          stride = root_ext_j;
        } else {
          stride = kir::mulExpr(stride, root_ext_j);
        }
      }
    }

    if (stride != nullptr) {
      strided_inds.push_back(kir::mulExpr(root_ind_i, stride));
    } else {
      strided_inds.push_back(root_ind_i);
    }
  }

  std::cerr << "getConsumerIndex_impl::strided_inds 1: " << strided_inds << std::endl;
  if (strided_inds.size() == 0)
    strided_inds.push_back(new kir::Int(0));

  std::cerr << "getConsumerIndex_impl::strided_inds 2: " << strided_inds << std::endl;
  return new kir::TensorIndex(consumer_tv, strided_inds);
}

namespace {
// Special case for expressions initializing reduction
// buffers. This special handling is necessary because ComputeDomain
// do not refelect the initialization loop nest.
kir::TensorIndex* getConsumerIndexForReductionInit(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {

  TORCH_INTERNAL_ASSERT(loops.size() ==
                        consumer_tv->domain()->noReductions().size());
  std::vector<Val*> strided_inds;
  for (auto loop_i = loops.begin() + consumer_tv->getThisComputeAtAxis();
       loop_i != loops.end(); ++loop_i) {
    if ((*loop_i)->iter_domain()->isThread()) continue;
    if ((*loop_i)->iter_domain()->isBroadcast()) continue;
    auto idx = (*loop_i)->index();
    Val* extent = nullptr;
    for (auto loop_j = loop_i + 1; loop_j != loops.end(); ++loop_j) {
      if ((*loop_j)->iter_domain()->isThread()) continue;
      if ((*loop_j)->iter_domain()->isBroadcast()) continue;
      Val* extent_j = (*loop_j)->iter_domain()->extent();
      extent = (extent != nullptr) ? kir::mulExpr(extent, extent_j) : extent_j;
    }
    strided_inds.push_back(extent != nullptr ? kir::mulExpr(idx, extent) : idx);
  }
  //return new kir::TensorIndex(consumer_tv, strided_inds);
  auto ti = new kir::TensorIndex(consumer_tv, strided_inds);
  DEBUG("Generated TI: ", ti);
  return ti;
}

} // namespace

kir::TensorIndex* Index::getConsumerIndex_impl2(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {

  std::cerr << "getConsumerIndex_impl2: "
            << "Generating index val for " << consumer_tv
            << ", #loops: " << loops.size()
            << std::endl;

  const ComputeDomain* cd = consumer_tv->getComputeDomain();

  if (cd->nDims() != loops.size()) {
    // TODO (CD): Is this an error?
    // It happens when initializing reduction buffers. In that case,
    // the number of loops should be equal to the number of
    // non-reduction axes.
    if (loops.size() == consumer_tv->domain()->noReductions().size()) {
      return getConsumerIndexForReductionInit(consumer_tv, loops);
    }
    std::stringstream ss;
    int i = 0;
    for (auto loop: loops) {
      ss << "loop[" << i++ << "]: " << loop->index()
         << ", " << loop->iter_domain()
         << "\n";
    }
    std::cerr << ss.str();
    TORCH_INTERNAL_ASSERT(false, "Invalid number of loops: ",
                          loops.size(), ", whereas the ComputeDomain for the consumer TV is",
                          *cd);
  }

  // TODO (CD): TV compute-at position vs CD position
  auto tv_ca_pos = consumer_tv->getThisComputeAtAxis();
  if (tv_ca_pos == consumer_tv->nDims()) {
    // no indexing needed
    return new kir::TensorIndex(consumer_tv, {});
  }
  auto left_most_own_axis = cd->getComputeDomainAxisIndex(tv_ca_pos);
  std::vector<Val*> strided_inds;
  //size_t loop_idx = left_most_own_axis;
  for (size_t i = left_most_own_axis; i < cd->nDims(); ++i) {
    IterDomain* axis = cd->axis(i);
    std::cerr << "CD axis at " << i << ": " << axis << std::endl;
    // Parallelized domain doesn't need offsetting
    if (axis->isThread()) continue;
    // Broadcast domain doesn't contribute to offsetting
    if (axis->isBroadcast()) continue;
    // Reduction domain doesn't contribute to offsetting
    if (axis->isReduction()) continue;
    kir::IterDomain* kir_axis = kir::lowerValue(axis)->as<kir::IterDomain>();
    //loop_idx = i;
    //TORCH_INTERNAL_ASSERT(loop_idx < loops.size());
    TORCH_INTERNAL_ASSERT(i < loops.size());
    kir::ForLoop* loop = loops.at(i);
    Val* idx = loop->index();

    std::cerr << "axis: " << axis
              << ", kir axis: " << kir_axis
              << ", idx: " << idx
              << std::endl;

    //Val* stride = nullptr;
    Val* stride = new kir::Int(1);
    for (size_t j = i + 1; j < cd->nDims(); ++j) {
      std::cerr << "j: " << j << std::endl;
      IterDomain* axis_j = cd->axis(j);
      std::cerr << "axis_j: " << axis_j << std::endl;
      if (axis_j->isParallelized()) continue;
      if (axis_j->isBroadcast()) continue;
      if (axis_j->isReduction()) continue;
      Val* extent = kir::lowerValue(axis_j->extent());
      stride = mulx(stride, extent);
    }

    strided_inds.push_back(mulx(idx, stride));
  }

  std::cerr << "strided inds: " << strided_inds << std::endl;

  auto ti = new kir::TensorIndex(consumer_tv, strided_inds);
  DEBUG("Generated TI: ", ti);
  return ti;
}

// Producer is the inputs of an expression
kir::TensorIndex* Index::getProducerIndex(
    TensorView* producer,
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  if (producer->domain()->noReductions().size() == 0) {
    return new kir::TensorIndex(producer, {});
  }

  if (producer->getMemoryType() == MemoryType::Global) {
    if (std::getenv("GPINDEX1")) {
      return getGlobalProducerIndex(producer, consumer, loops);
    } else {
      return getProducerIndex_impl2(producer, consumer, loops);
    }
  }

  if (std::getenv("PINDEX1")) {
    return getProducerIndex_impl(producer, consumer, loops);
  } else {
    return getProducerIndex_impl2(producer, consumer, loops);
  }
}

// Consumer is the output of an expression
kir::TensorIndex* Index::getConsumerIndex(
    TensorView* consumer,
    const std::vector<kir::ForLoop*>& loops) {
  if (consumer->domain()->noReductions().size() == 0) {
    return new kir::TensorIndex(consumer, {});
  }

  if (consumer->getMemoryType() == MemoryType::Global) {
    return getGlobalConsumerIndex(consumer, loops);
  }

  if (std::getenv("CINDEX1")) {
    return getConsumerIndex_impl(consumer, loops);
  } else {
    return getConsumerIndex_impl2(consumer, loops);
  }
}

// Basically just copy getGlobalConsumerIndex, just don't do the striding and
// return std::vector of Vals
std::pair<std::vector<Val*>, bool> Index::getConsumerRootPredIndices(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops,
    const std::vector<bool>& root_contiguity,
    bool unroll) {
  // grab all tensor views from producer_tv <- computeAtRoot
  std::deque<TensorView*> tv_stack = getComputeAtTVStackFrom(consumer_tv);

  std::unordered_map<kir::ForLoop*, Val*> loop_to_ind_map;

  std::transform(
      loops.begin(),
      loops.end(),
      std::inserter(loop_to_ind_map, loop_to_ind_map.begin()),
      [](kir::ForLoop* fl) { return std::make_pair(fl, fl->index()); });

  if (unroll) {
    bool within_unroll = false;
    Val* one = new kir::Int(1);
    for (auto loop : loops) {
      if (loop->iter_domain()->getParallelType() == ParallelType::Unroll) {
        within_unroll = true;
      }

      if (within_unroll && !loop->iter_domain()->isThread()) {
        loop_to_ind_map[loop] =
            kir::subExpr(loop->iter_domain()->extent(), one);
      }
    }
  }

  auto index_map = generateIndexAndExtentMap(
                       tv_stack,
                       std::deque<kir::ForLoop*>(loops.begin(), loops.end()),
                       loop_to_ind_map,
                       root_contiguity)
                       .first;

  // Indices should now be mapped onto IterDomains in consumer, so just grab
  // and use them.

  // If we are generating a predicate for initialization check if we should use
  // rfactor instead of root_dom
  bool use_rfactor = true;
  if (consumer_tv->hasRFactor()) {
    auto rfactor_dom = consumer_tv->getMaybeRFactorDomain();
    for (auto rfactor_id : rfactor_dom) {
      if (rfactor_id->isReduction()) {
        auto kir_rfactor_id =
            kir::lowerValue(rfactor_id)->as<kir::IterDomain>();
        if (index_map.find(kir_rfactor_id) != index_map.end()) {
          if (!index_map.at(kir_rfactor_id)->isZeroInt()) {
            use_rfactor = false;
            break;
          }
        }
      }
    }
  }

  auto root_dom = use_rfactor ? consumer_tv->getMaybeRFactorDomain()
                              : consumer_tv->getRootDomain();

  std::vector<Val*> root_inds(root_dom.size(), new kir::Int(0));
  for (size_t i = 0; i < root_dom.size(); i++) {
    if (root_dom[i]->isBroadcast()) {
      continue;
    }

    auto kir_root_dom_i = kir::lowerValue(root_dom[i])->as<kir::IterDomain>();
    if (index_map.find(kir_root_dom_i) != index_map.end()) {
      auto ind = index_map.at(kir_root_dom_i);
      TORCH_INTERNAL_ASSERT(kir::isLoweredScalar(ind))
      root_inds[i] = ind;
    }
  }

  return std::make_pair(root_inds, use_rfactor);
}

} // namespace fuser
} // namespace jit
} // namespace torch
