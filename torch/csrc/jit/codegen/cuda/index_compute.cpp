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
        kir_root_dom_i);

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

kir::TensorIndex* Index::getProducerIndex_impl2(
    TensorView* producer_tv,
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {

  std::cerr << "getProducerIndex_impl2: "
            << "producer: " << producer_tv
            << ", consumer: " << consumer_tv
            << std::endl;

  const ComputeDomain* consumer_cd = consumer_tv->getComputeDomain();

  std::unordered_map<IterDomain*, Val*> consumer_idx_map;
  for (size_t i = 0; i < consumer_tv->nDims(); ++i) {
    std::cerr << "idx: " << i << std::endl;
    IterDomain* dom = consumer_tv->axis(i);
    std::cerr << "dom: " << dom << std::endl;
#if 0
    if (dom->isReduction()) {
      TORCH_INTERNAL_ASSERT(false, "reduction: ", producer_dom);
      std::cerr << "Ignoring reduciton: " << producer_dom << std::endl;
      continue;
    }
#endif
    auto cd_axis_idx = consumer_cd->getComputeDomainAxisIndex(i);
    //IterDomain* consumer_cd_axis = consumer_cd->axis(cd_axis_idx);
    TORCH_INTERNAL_ASSERT(cd_axis_idx < loops.size());
    Val* loop_idx = loops.at(cd_axis_idx)->index();
    //Val* loop_extent = kir::lowerValue(consumer_cd_axis->extent());
    std::cerr << "Initial entry: "
              << dom
              << ", " << loop_idx
              << std::endl;
    TORCH_INTERNAL_ASSERT(kir::isLoweredVal(loop_idx));
    //TORCH_INTERNAL_ASSERT(kir::isLoweredVal(loop_extent));
    consumer_idx_map.insert({dom, loop_idx});
  }

  std::cerr << "Initial consumer_idx_map\n";
  for (auto k: consumer_idx_map) {
    std::cerr << k.first << " -> {" << k.second << "}\n";
  }

  std::vector<Val*> consumer_domain;
  std::transform(consumer_tv->domain()->domain().begin(),
                 consumer_tv->domain()->domain().end(),
                 std::back_inserter(consumer_domain),
                 [](IterDomain* id) {
                   return static_cast<Val*>(id);
                 });
  auto domain_exprs = ExprSort::getExprs(consumer_tv->fusion(), consumer_domain);

  // traverse in a reverse order
  for (auto it = domain_exprs.rbegin(); it != domain_exprs.rend(); ++it) {
    Expr* expr = *it;
    std::cerr << "Traversing " << expr << std::endl;
    if (expr->getExprType() == ExprType::Split) {
      Split* split = expr->as<Split>();
      IterDomain* outer = split->outer();
      IterDomain* inner = split->inner();
      auto outer_map = consumer_idx_map.find(outer);
      TORCH_INTERNAL_ASSERT(outer_map != consumer_idx_map.end());
      auto inner_map = consumer_idx_map.find(inner);
      TORCH_INTERNAL_ASSERT(inner_map != consumer_idx_map.end());
      Val* outer_idx = outer_map->second;
      //Val* outer_extent = std::get<1>(outer_map->second);
      //IterDomain* inner_consumer_id = std::get<1>(inner_map->second);
      Val* inner_idx = inner_map->second;
      //Val* inner_extent = std::get<1>(inner_map->second);
      Val* inner_extent = kir::lowerValue(inner->extent());
      Val* in_idx = addx(mulx(outer_idx, inner_extent), inner_idx);
      //Expr* consumer_expr = inner_consumer_id->getOrigin();
      //TORCH_INTERNAL_ASSERT(consumer_expr->getExprType() == ExprType::Split);
      //IterDomain* consumer_in_id = consumer_expr->as<Split>()->in();
      consumer_idx_map.insert({split->in(), {in_idx}});
      consumer_idx_map.erase(outer);
      consumer_idx_map.erase(inner);
    } else if (expr->getExprType() == ExprType::Merge) {
      Merge* merge = expr->as<Merge>();
      auto out_map = consumer_idx_map.find(merge->out());
      TORCH_INTERNAL_ASSERT(out_map != consumer_idx_map.end());
      Val* out_idx = out_map->second;
      //auto out_extent = std::get<1>(out_map->second);
      //IterDomain* out_consumer_id = std::get<1>(out_map->second);
      //Expr* consumer_expr = out_consumer_id->getOrigin();
      //TORCH_INTERNAL_ASSERT(consumer_expr->getExprType() == ExprType::Merge);
      //IterDomain* consumer_inner_id = consumer_expr->as<Merge>()->inner();
      //Val* inner_extent =
      //kir::lowerValue(consumer_inner_id->extent());
      Val* inner_extent = kir::lowerValue(merge->inner()->extent());
      Val* inner_idx = kir::modExpr(out_idx, inner_extent);
      Val* outer_idx = kir::divExpr(out_idx, inner_extent);
      if (out_idx->isZeroInt()) {
        inner_idx = new kir::Int(0);
        outer_idx = new kir::Int(0);
      }
      consumer_idx_map.insert({merge->outer(), outer_idx});
      consumer_idx_map.insert({merge->inner(), inner_idx});
      consumer_idx_map.erase(merge->out());
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unexpected expr: ", expr);
    }
  }

  std::cerr << "Root consumer_idx_map\n";
  for (auto k: consumer_idx_map) {
    std::cerr << k.first << " -> {" << k.second << "}\n";
  }

  std::unordered_map<IterDomain*, Val*> producer_idx_map;
  const auto root_mapping = TensorDomain::mapRootPandC(producer_tv->domain(),
                                                      consumer_tv->domain());

  // TOOD (CD): Should this be getMaybeRFactorDomain()?
  for (auto producer_root_id: producer_tv->getRootDomain()) {
    std::cerr << "Producer root: " << producer_root_id << std::endl;
    auto consumer_root_id = std::find_if(root_mapping.begin(), root_mapping.end(),
                                         [producer_root_id](const auto& pair) {
                                           return pair.first == producer_root_id;
                                         });
    if (consumer_root_id == root_mapping.end()) {
      TORCH_INTERNAL_ASSERT(producer_root_id->isReduction());
      continue;
    }
    //TORCH_INTERNAL_ASSERT(std::find(root_domain.begin(),
    //root_domain.end(), root_dom) != root_domain.end());
    auto idx_map = consumer_idx_map.find(consumer_root_id->second);
    TORCH_INTERNAL_ASSERT(idx_map != consumer_idx_map.end(),
                          "Consumer root not found: ",
                          consumer_root_id->second);
    Val* idx = idx_map->second;
    //auto extent = std::get<1>(m.second);
    if (producer_root_id->isBroadcast()) {
      std::cerr << "Root is broadcast;\n";
      idx = new kir::Int(0);
      //extent = new kir::Int(1);
    }
    std::cerr << "Producer root dom: " << producer_root_id
              << ", idx: " << idx << std::endl;
    //auto dom_offset = std::distance(root_domain.begin(), it);
    //root_inds.push_back({dom_offset, idx, extent});
    producer_idx_map.insert({producer_root_id, idx});
  }

  // Transform to the original producer domain
  std::vector<Val*> strided_inds;
  {
    std::vector<Val*> producer_domain;
    std::transform(producer_tv->domain()->domain().begin(),
                   producer_tv->domain()->domain().end(),
                   std::back_inserter(producer_domain),
                   [](IterDomain* dom) {
                     return dom->as<Val>();
                   });
    // TODO (CD): This goes back all the way to the root
    // domain. Should stop at rfactor root?
    auto producer_exprs = ExprSort::getExprs(producer_tv->fusion(),
                                             producer_domain);
    for (auto expr: producer_exprs) {
      std::cerr << "Producer expr: " << expr;
      if (expr->getExprType() == ExprType::Split) {
        Split* split = expr->as<Split>();
        IterDomain* in = split->in();
        if (producer_idx_map.find(in) == producer_idx_map.end()) {
          TORCH_INTERNAL_ASSERT(false);
          continue;
        }
        Val* in_idx = producer_idx_map.find(in)->second;
        Val* inner_idx = kir::modExpr(in_idx, kir::lowerValue(split->factor()));
        Val* outer_idx = kir::divExpr(in_idx, kir::lowerValue(split->factor()));
        producer_idx_map.insert({split->outer(), outer_idx});
        producer_idx_map.insert({split->inner(), inner_idx});
      } else if (expr->getExprType() == ExprType::Merge) {
        Merge* merge = expr->as<Merge>();
        if (producer_idx_map.find(merge->inner()) == producer_idx_map.end() ||
            producer_idx_map.find(merge->outer()) == producer_idx_map.end()) {
          TORCH_INTERNAL_ASSERT(false);
          continue;
        }
        Val* inner_idx = producer_idx_map.find(merge->inner())->second;
        Val* outer_idx = producer_idx_map.find(merge->outer())->second;
        Val* inner_dim = kir::lowerValue(merge->inner()->extent());
        Val* out_idx = addx(mulx(outer_idx, inner_dim), inner_idx);
        producer_idx_map.insert({merge->out(), out_idx});
      } else {
        TORCH_INTERNAL_ASSERT(false, "Unexpected expr: ", expr);
      }
    }
    for (size_t i = producer_tv->getThisComputeAtAxis(); i < producer_tv->nDims(); ++i) {
      IterDomain* prod_id = producer_tv->axis(i);
      if (prod_id->isReduction()) {
        // this should not appear at the consumer
        continue;
      }
      if (producer_idx_map.find(prod_id) == producer_idx_map.end()) {
        // If the index is determined to be 0, no mapping entry is
        // created.
        TORCH_INTERNAL_ASSERT(false,
                              "Mapping not detected for ", prod_id);
        continue;
      }
      if (prod_id->isThread()) {
        continue;
      }
      Val* idx = producer_idx_map.find(prod_id)->second;
      std::cerr << "Prod idx: " << idx;
      if (idx->getOrigin()) {
        std::cerr << " (" << idx->getOrigin() << ")";
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
      strided_inds.push_back(mulx(idx, extent));
    }
  }

  std::cerr << "getConsumerIndex_impl2: Strided indices: " << strided_inds << std::endl;

  return new kir::TensorIndex(producer_tv, strided_inds);
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

  return new kir::TensorIndex(consumer_tv, strided_inds);
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

kir::TensorIndex* Index::getConsumerIndex_impl2(
    TensorView* consumer_tv,
    const std::vector<kir::ForLoop*>& loops) {

  std::cerr << "getConsumerIndex_impl2: "
            << "Generating index val for " << consumer_tv
            << ", #loops: " << loops.size()
            << std::endl;

  const ComputeDomain* cd = consumer_tv->getComputeDomain();
  // TODO: TV compute-at position vs CD position
  auto tv_ca_pos = consumer_tv->getThisComputeAtAxis();
  if (tv_ca_pos == consumer_tv->nDims()) {
    // no indexing needed
    return new kir::TensorIndex(consumer_tv, {});
  }
  auto left_most_own_axis = cd->getComputeDomainAxisIndex(tv_ca_pos);
  std::vector<Val*> strided_inds;
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

  return new kir::TensorIndex(consumer_tv, strided_inds);
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
    return getGlobalProducerIndex(producer, consumer, loops);
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
