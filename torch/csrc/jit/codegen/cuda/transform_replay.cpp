#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/compute_at.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

using id_map = std::unordered_map<IterDomain*, IterDomain*>;

namespace {

class ReplaySelf : public ReplayTransformations {
 private:
  // Took a good bit of this from ReplayTransformations::handle(Split...)
  void handle(Split* s) override {
    // Grab input to the split operation
    auto id_in = s->in();

    // Grab our mapping of that ID to the one we're replaying
    auto it = id_map_.find(id_in);

    // Make sure it exists in the map
    TORCH_INTERNAL_ASSERT(
        it != id_map_.end(),
        "Transform traversal failed, dependencies not met.");
    // Grab the ID we're going to replay on
    auto mapped = it->second;

    // This ID should be a leaf ID (meaning it has no uses we generated)
    TORCH_INTERNAL_ASSERT(
        leaf_ids_.find(mapped) != leaf_ids_.end(),
        "Transform traversal failed, modified a node but it was not a leaf node.");

    // outer loop size
    Val* oe = ceilDiv(mapped->extent(), s->factor());

    // Manually replay the split, following the output of the operations.
    // This is so rfactor ops are replayed correctly.
    IterDomain* ido = new IterDomain(
        new Int(0),
        oe->as<Int>(),
        s->outer()->getParallelType(),
        s->outer()->getIterType(),
        s->outer()->isRFactorProduct());

    // inner IterDomain
    IterDomain* idi = new IterDomain(
        new Int(0),
        s->factor(),
        s->inner()->getParallelType(),
        s->outer()->getIterType(),
        s->inner()->isRFactorProduct());

    // Generate the split node
    new Split(ido, idi, mapped, s->factor());

    // Remove mapped id from leaf IDs
    leaf_ids_.erase(mapped);

    // Add outputs to leaf IDs
    leaf_ids_[ido] = counter++;
    leaf_ids_[idi] = counter++;

    // Update our ID map to include these outputs
    id_map_[s->outer()] = ido;
    id_map_[s->inner()] = idi;
  }

  void handle(Merge* m) override {
    auto id_outer = m->outer();
    auto id_inner = m->inner();

    auto it_outer = id_map_.find(id_outer);
    auto it_inner = id_map_.find(id_inner);

    TORCH_INTERNAL_ASSERT(
        it_outer != id_map_.end() && it_inner != id_map_.end(),
        "Transform traversal failed, dependencies not met.");

    auto id_outer_mapped = it_outer->second;
    auto id_inner_mapped = it_inner->second;

    TORCH_INTERNAL_ASSERT(
        leaf_ids_.find(id_outer_mapped) != leaf_ids_.end() &&
            leaf_ids_.find(id_inner_mapped) != leaf_ids_.end(),
        "Transform traversal failed, modified ",
        id_outer_mapped,
        " and ",
        id_inner_mapped,
        " however one or both are not leaf nodes.");

    Val* merged_id_size =
        mul(id_outer_mapped->extent(), id_inner_mapped->extent());

    IterDomain* merged_id = new IterDomain(
        new Int(0),
        merged_id_size->as<Int>(),
        m->out()->getParallelType(),
        m->outer()->getIterType(),
        m->out()->isRFactorProduct());

    new Merge(merged_id, id_outer_mapped, id_inner_mapped);

    // Remove inputs from the leaf IDs
    leaf_ids_.erase(id_outer_mapped);
    leaf_ids_.erase(id_inner_mapped);

    // Add the output to the leaf IDs
    leaf_ids_[merged_id] = counter++;

    id_map_[m->out()] = merged_id;
  }

 public:
  ReplaySelf(const std::vector<IterDomain*>& _target_domain, id_map _id_map)
      : ReplayTransformations(_target_domain, std::move(_id_map), false) {}
};

} // namespace

// Self replay.
TensorDomain* TransformReplay::fullSelfReplay(
    const TensorDomain* new_self_root,
    const TensorDomain* self) {
  TORCH_INTERNAL_ASSERT(
      new_self_root->nDims() == self->getRootDomain().size(),
      "Invalid number of IterDomains provided.");

  // Map for replay, should be pretty simple.
  id_map axis_map;
  {
    size_t i = 0;
    for (auto id : self->getRootDomain()) {
      TORCH_INTERNAL_ASSERT(
          new_self_root->axis(i)->start() == id->start(),
          "Replay does not support IterDomains that do not start at 0.");

      TORCH_INTERNAL_ASSERT(
          new_self_root->axis(i)->getParallelType() == id->getParallelType() &&
              new_self_root->axis(i)->isReduction() == id->isReduction() &&
              new_self_root->axis(i)->isRFactorProduct() ==
                  id->isRFactorProduct() &&
              new_self_root->axis(i)->isBroadcast() == id->isBroadcast(),
          "Axes do not match for self replay.");
      axis_map[id] = new_self_root->axis(i);
      i++;
    }
  }

  // Replay producer dimensions.
  ReplaySelf replay(self->domain(), axis_map);
  std::vector<IterDomain*> new_domain(self->nDims(), nullptr);

  {
    size_t i = 0;
    for (auto id : self->domain()) {
      auto it = replay.getReplay().find(id);
      TORCH_INTERNAL_ASSERT(
          it != replay.getReplay().end(),
          "Error during replay, didn't replay an axis.");
      new_domain[i++] = it->second;
    }
  }

  return new TensorDomain(
      new_self_root->domain(), new_domain, self->contiguity());
}

namespace {
// TODO (CD)
template <typename IterDomainContainer>
std::vector<std::pair<IterDomain*, size_t>> replayMapFind(
    std::unordered_map<IterDomain*, IterDomain*> map,
    const IterDomainContainer& ids) {
  //for (auto id: ids) {
    //std::cerr << "replayMapFind:: id: " << id << std::endl;
  //}
  std::vector<std::pair<IterDomain*, size_t>> cd_map;
  std::unordered_set<size_t> found;
  // First, look for identical domains
  for (size_t i = 0; i < ids.size(); ++i) {
    const IterDomain* id = ids.at(i);
    auto it = std::find_if(map.begin(), map.end(),
                           [id](const auto& map_kv) {
                             IterDomain* id_in_map = map_kv.first;
                             return id == id_in_map;
                           });
    if (it != map.end()) {
      auto mapped_id = it->second;
      map.erase(it);
      cd_map.push_back({mapped_id, i});
      found.insert(i);
    }
  }
  // Next, try to find matching "same" IDs
  for (size_t i = 0; i < ids.size(); ++i) {
    if (found.find(i) == found.end()) continue;
    const IterDomain* id = ids.at(i);
    auto it = std::find_if(map.begin(), map.end(),
                           [id](const auto& map_kv) {
                             IterDomain* id_in_map = map_kv.first;
                             return ComputeDomain::sameAxes(id, id_in_map);
                           });
    if (it != map.end()) {
      auto mapped_id = it->second;
      map.erase(it);
      cd_map.push_back({mapped_id, i});
      found.insert(i);
    }
  }
  // Reorder the found IDs with the order of ids
  std::sort(cd_map.begin(), cd_map.end(),
            [](const auto& x, const auto&y) {
              return x.second < y.second;
            });
  return cd_map;
}

auto getRootCAIDs(const std::vector<IterDomain*>& domain,
                  size_t ca_pos) {
  std::vector<IterDomain*> ca_ids(
      domain.begin(),
      domain.begin() + ca_pos);
  auto root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(ca_ids.begin(), ca_ids.end()));
  std::unordered_set<IterDomain*> root_ca_ids{
    ir_utils::filterByType<IterDomain>(root_vals).begin(),
    ir_utils::filterByType<IterDomain>(root_vals).end()};
  return root_ca_ids;
}

// ca_pos: compute-at position in the reference TensorDomain
bool insertMissingDomains(std::vector<IterDomain*>& target_root,
                          std::vector<bool>& target_contig,
                          std::vector<bool>& ca_placeholder,
                          const std::vector<IterDomain*>& reference_root,
                          std::unordered_set<IterDomain*> reference_root_ca_ids,
                          bool reference_is_consumer,
                          bool target_has_rfactor) {
  std::stringstream ss;
  ss << "{";
  for (const auto id: reference_root_ca_ids) {
    ss << id << " ";
  }
  ss << "}";
  std::cerr << "insertMissingDomains: "
            << "target: " << target_root
            << ", reference: " << reference_root
            << ", reference_root_ca_ids: " << ss.str()
            << ", reference_is_consumer: " << reference_is_consumer
            << ", current placeholder: " << ca_placeholder
            << ", target_has_rfactor: " << target_has_rfactor
            << std::endl;

  if (!target_has_rfactor) {
    TORCH_INTERNAL_ASSERT(ca_placeholder.size() == target_root.size());
  }

  bool insertion_done = false;
  size_t target_offset = 0;
  size_t ref_offset = 0;
  auto ca_placeholder_it = ca_placeholder.begin();
  // This is very similar to TensorDomain::mapRootDomains. Refactoring possible?
  while (target_offset < target_root.size() ||
         ref_offset < reference_root.size()) {
    IterDomain* target_id = target_offset < target_root.size() ?
                                            target_root.at(target_offset) : nullptr;
    IterDomain* ref_id = ref_offset < reference_root.size() ?
                                      reference_root.at(ref_offset) : nullptr;
    TORCH_INTERNAL_ASSERT(!(target_id == nullptr && ref_id == nullptr));

    const bool target_is_reduction = target_id != nullptr && target_id->isReduction();
    const bool reference_is_reduction =
        ref_id != nullptr && ref_id->isReduction();
    const bool reference_is_broadcast =
        ref_id != nullptr && ref_id->isBroadcast();

    auto ref_offset_old = ref_offset;

    if (target_id == ref_id ||
        (target_id && ref_id && ComputeDomain::sameAxes(target_id, ref_id))) {
      ++target_offset;
      ++ref_offset;
      ++ca_placeholder_it;
    } else if (reference_is_consumer && target_is_reduction) {
      ++target_offset;
      ++ca_placeholder_it;
    } else if (!reference_is_consumer && reference_is_reduction) {
      ++ref_offset;
    } else if (reference_is_broadcast) {
      // target_id does not match ref_id, and reference_id is a
      // broadcast. If it's in the CA domain set, insert a dummy
      // IterDomain to the target domain. For now, just insert the
      // ref_id. This avoids inserting again even if this function is
      // called again.
      // Note that target_id may be a broadcast as well. We assume
      // that target_id and ref_id don't match when their pointer
      // values are different. Since the pointer values are already
      // checked, at this point target_id and ref_id are considered
      // different even if they are broadcast.
      if (reference_root_ca_ids.find(ref_id) != reference_root_ca_ids.end()) {
        std::cerr << "Inserting " << ref_id << " at position " << target_offset << std::endl;
        // Not supported for rfactor tensors
        TORCH_INTERNAL_ASSERT(!target_has_rfactor,
                              "Inserting placeholder not supported for rfactor tensros.");
        target_root.insert(target_root.begin() + target_offset, ref_id);
        target_contig.insert(target_contig.begin() + target_offset, true);
        ca_placeholder.insert(ca_placeholder_it, true);
        insertion_done = true;
        ++target_offset;
        ++ca_placeholder_it;
      }
      ++ref_offset;
    } else {
      TORCH_INTERNAL_ASSERT(target_id != nullptr);
      TORCH_INTERNAL_ASSERT(ref_id != nullptr);
      ++target_offset;
      ++ref_offset;
      ++ca_placeholder_it;
    }

    if (ref_offset != ref_offset_old) {
      DEBUG("Erasing ", ref_id);
      reference_root_ca_ids.erase(ref_id);
      if (reference_root_ca_ids.empty()) {
        DEBUG("Root ca empty");
        break;
      }
    }
  }

  //ca_placeholder.resize(target_root.size(), false);
  if (insertion_done) {
    TORCH_INTERNAL_ASSERT(ca_placeholder.size() == target_root.size());
  }

  std::cerr << "insertMissingDomains done: " << target_root
            << ", placeholder: " << ca_placeholder
            << std::endl;
  return insertion_done;
}
} // namespace

// Producer could have rfactor axes which consumer may want replayed. We can
// "replay" them as long as it doesn't modify the root rfactor axes. What we
// really want to do is validate if we replayed these axes to the ones they
// mapped to in the consumer the operations would all be the same. then we want
// to start the replay of the producer from the rfactor root axes, not the root.
std::tuple<TensorDomain*, unsigned int, std::vector<size_t>,
           std::unordered_map<IterDomain*, IterDomain*>> TransformReplay::replayPasC(
    const TensorDomain* producer,
    const TensorDomain* consumer,
    const ComputeDomain* consumer_cd,
    int consumer_compute_at_axis) {
  // producer_compute_at_axis is a position in the producer compute domain.
  normalizeComputeAtPos(consumer_compute_at_axis, consumer_cd->nDims());

  const auto td_pos = consumer_cd->getTensorDomainPos(consumer_compute_at_axis);

  std::cerr << "replayPasC: producer: " << producer
            << ", consumer: " << consumer << ", consumer_cd: " << *consumer_cd
            << ", CD pos: " << consumer_compute_at_axis
            << ", TD pos: " << td_pos << std::endl;

  TORCH_INTERNAL_ASSERT(
      consumer_compute_at_axis >= 0 &&
      (unsigned int)consumer_compute_at_axis <= consumer_cd->nDims(),
      "Invalid axis in transform replayPasC.");

  const auto& consumer_domain = consumer_cd->axes();

  // consumer ids we need to match in producer
  std::vector<IterDomain*> consumer_CA_ids(
      consumer_domain.begin(),
      consumer_domain.begin() + consumer_compute_at_axis);

  std::cerr << "consumer CA IDs: " << consumer_CA_ids << std::endl;

  // Figure out all inputs required to generate the compute_at dimensions
  std::unordered_set<Val*> consumer_CA_root_vals = IterVisitor::getInputsTo(
      std::vector<Val*>(consumer_CA_ids.begin(), consumer_CA_ids.end()));

  std::unordered_set<IterDomain*> consumer_CA_root_ids;
  for (auto val : consumer_CA_root_vals) {
    if (val->getValType().value() == ValType::IterDomain) {
      consumer_CA_root_ids.emplace(val->as<IterDomain>());
    }
  }

  for (const auto& id: consumer_CA_root_ids) {
    std::cerr << "consumer CA root ID: " << id << std::endl;
  }

  std::vector<IterDomain*> producer_root =
      producer->getMaybeRFactorDomain();
  std::cerr << "Producer root: " << producer_root << std::endl;
  auto producer_contig = producer->contiguity();
  std::vector<bool> ca_placeholder = producer->placeholder();
  bool placeholder_inserted = insertMissingDomains(
      producer_root,
      producer_contig,
      ca_placeholder,
      consumer->getRootDomain(),
      getRootCAIDs(consumer->domain(), td_pos),
      true, producer->hasRFactor());

  // Map of consumer_CA_root_ids to related producer_CA_ids
  auto replay_root_map =
      consumer_cd->mapRootDomain(producer_root, consumer_CA_root_ids);
  // Reduction IDs can't be shared.
  for (auto it = replay_root_map.begin(); it != replay_root_map.end();) {
    IterDomain* producer_root_id = it->second;
    if (producer_root_id->isReduction()) {
      DEBUG("Remove reduction ID from CA root mapping: ", producer_root_id);
      it = replay_root_map.erase(it);
    } else {
      ++it;
    }
  }

  // Track which root axes in producer we will send to replay
  std::unordered_set<IterDomain*> producer_roots4replay;
  for (auto entry : replay_root_map) {
    std::cerr << "replay root map: " << entry.first << " -> " << entry.second << std::endl;
    producer_roots4replay.emplace(entry.second);
  }

  // Instead of replaying from the root, lets try to play forward the history of
  // producer if they match ops on consumer. Enforce if we modify an rfactor
  // axis that those ops must match.
  BestEffortReplay forward_replay(
      producer->domain(), consumer_CA_ids, replay_root_map);

  // Make a new map based on all the leaves resulting from best effort replay
  id_map forwarded_replay_map;
  for (auto entry : forward_replay.getReplay()) {
    std::cerr << "BestEffortReplay: " << entry.first << " -> " << entry.second << std::endl;
    if (forward_replay.getUnorderedLeafIDs().find(entry.second) !=
        forward_replay.getUnorderedLeafIDs().end())
      forwarded_replay_map[entry.first] = entry.second;
  }

  for (const auto& leaf_id: forward_replay.getUnorderedLeafIDs()) {
    std::cerr << "leaf id: " << leaf_id << std::endl;
  }

  // Replay producer dimensions.
  ReplayTransformations replay_PasC(
      consumer_CA_ids, forwarded_replay_map, false);

  for (const auto& replay: replay_PasC.getReplay()) {
    std::cerr << "ReplayTransformation: " << replay.first << " -> " << replay.second << std::endl;
  }

  auto leaf_ids(replay_PasC.getUnorderedLeafIDs());
  auto crossover_map(replay_PasC.getLeafMap());

  // Remove all ids that map to the compute at axis, we're going to replay the
  // rest
#if 0
  for (auto c_id : consumer_CA_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it == replay_PasC.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          c_id->isBroadcast(),
          "Could not find axis, ",
          c_id,
          ", requested in replay.");
      continue;
    }
    if (leaf_ids.find(it->second) != leaf_ids.end()) {
      leaf_ids.erase(it->second);
    }
  }
#else
  for (auto replayed_id: replayMapFind(replay_PasC.getReplay(), consumer_CA_ids)) {
    if (leaf_ids.find(replayed_id.first) != leaf_ids.end()) {
      leaf_ids.erase(replayed_id.first);
      //crossover_map.erase(replayed_id.first);
    }
  }
#endif

  // leaf_ids now contains all producer ID products that are not used to satisfy
  // the computeAt Turn into a  map so we can play forward these IDs in producer
  // (if possible):
  id_map producer_self_replay_map;
  for (auto entry : leaf_ids)
    producer_self_replay_map[entry.first] = entry.first;

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto producer_root_id : producer_root)
    if (producer_roots4replay.find(producer_root_id) ==
        producer_roots4replay.end()) {
      producer_self_replay_map[producer_root_id] = producer_root_id;
    }

  // Play forward transformations all producer IDs we can
  auto producer_replayed_leaves = BestEffortReplay(
      producer->domain(), producer->domain(), producer_self_replay_map);

  /*
   * Accumulate axes in to the new domain in the following order, making sure to
   * avoid any duplicates:
   *
   * (1) replay_PasC.getReplay holds mappings from axes in consumer compute at
   * axes -> corresponding generated axes in producer
   *
   * (2) Any axes that were not added, that can be mapped directly from an ID in
   * consumer->domain(). These are axes that were "fully replayed" relative to
   * the consumer, even though it wasn't in the computeAt range.
   *
   * producer_replayed_leaves now contain ids that we tried to forward
   * back to what they were in producer. If they couldn't be forwarded they're
   * left in their "most forwarded" form which may be just a remainder of the
   * transformation required to generate the computeAt axes.
   *
   * (3) Axes in producer->domain() that are in producer_replayed_leaves
   *
   * (4) Axes not in producer->domain() that are in producer_replayed_leaves
   *
   */

  std::vector<IterDomain*> new_IDs;
  std::unordered_set<IterDomain*> used_IDs;
  std::vector<size_t> td2cd_map;
  // Add axes in (1)
#if 0
  for (auto c_id : consumer_CA_ids) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it == replay_PasC.getReplay().end()) {
      TORCH_INTERNAL_ASSERT(
          c_id->isBroadcast(),
          "Could not find axis, ",
           c_id,
          ", requested in replay.");
      continue;
    }
    auto replayed_id = it->second;
    std::cerr << "(1): " << replayed_id << std::endl;
    new_IDs.push_back(replayed_id);
    used_IDs.emplace(replayed_id);
  }
#else
  for (auto replayed_id: replayMapFind(replay_PasC.getReplay(), consumer_CA_ids)) {
    std::cerr << "(1): " << replayed_id << std::endl;
    new_IDs.push_back(replayed_id.first);
    used_IDs.emplace(replayed_id.first);
    td2cd_map.push_back(replayed_id.second);
  }
#endif

  unsigned int producer_compute_at_axis = new_IDs.size();
  // Add axes in (2)
#if 0
  for (auto c_id : consumer_domain) {
    auto it = replay_PasC.getReplay().find(c_id);
    if (it != replay_PasC.getReplay().end()) {
      auto id = it->second;
      if (used_IDs.find(id) == used_IDs.end()) {
        std::cerr << "(2): " << id << std::endl;
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }
#else
  for (auto replayed_id: replayMapFind(replay_PasC.getReplay(), consumer_domain)) {
    // If the leaf id from ReplayTransformation is used to move
    // forward with BestEffortReplay, it is not a final ID. It is
    // initialy set as a leaf for BestEffortReplay, and so if it does
    // not remain there, it means the ID is used to replay.
    if (producer_replayed_leaves.getUnorderedLeafIDs().find(replayed_id.first) ==
        producer_replayed_leaves.getUnorderedLeafIDs().end()) {
      continue;
    }
    if (used_IDs.find(replayed_id.first) == used_IDs.end()) {
      std::cerr << "(2): " << replayed_id << std::endl;
      new_IDs.push_back(replayed_id.first);
      used_IDs.emplace(replayed_id.first);
    }
  }
#endif

  // Add axes in (3)
  for (auto id : producer->domain()) {
    if (producer_replayed_leaves.getUnorderedLeafIDs().find(id) !=
        producer_replayed_leaves.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        std::cerr << "(3): " << id << std::endl;
        new_IDs.push_back(id);
        used_IDs.emplace(id);
      }
    }
  }

  // Add axes in (4)
  for (auto id : producer_replayed_leaves.getLeafIDs())
    if (used_IDs.find(id) == used_IDs.end()) {
      std::cerr << "(4): " << id << std::endl;
      new_IDs.push_back(id);
    }

  std::cerr << "New IDs: " << new_IDs << std::endl;

  TensorDomain* replayed = nullptr;
  if (placeholder_inserted) {
    TORCH_INTERNAL_ASSERT(producer->getRFactorDomain().size() == 0);
    replayed = new TensorDomain(
        producer_root,
        producer->getRFactorDomain(),
        new_IDs,
        producer_contig,
        ca_placeholder,
        crossover_map);
  } else {
    replayed = new TensorDomain(
        producer->getRootDomain(),
        producer->getRFactorDomain(),
        new_IDs,
        producer->contiguity(),
        ca_placeholder,
        crossover_map);
  }
  std::cerr << "replayPasC done: " << replayed
            << ", num shared axes: " << producer_compute_at_axis
            << std::endl;
  return {replayed, producer_compute_at_axis, td2cd_map,
          crossover_map};
}

std::tuple<TensorDomain*, unsigned int, std::vector<size_t>,
           std::unordered_map<IterDomain*, IterDomain*>> TransformReplay::replayCasP(
    const TensorDomain* consumer,
    const TensorDomain* producer,
    const ComputeDomain* producer_cd,
    int producer_compute_at_axis) {
  // producer_compute_at_axis is a position in the producer compute domain.
  normalizeComputeAtPos(producer_compute_at_axis, producer_cd->nDims());

  // Can't share reduction axes
  for (int i = 0; i < producer_compute_at_axis; ++i) {
    if (producer_cd->isComputeDomainAxisUsed(i)) {
      if (producer_cd->getTensorDomainAxis(i)->isReduction()) {
        producer_compute_at_axis = i;
        DEBUG("Moved compute-at position to ", i, " because the producer axis at ",
              producer_cd->getTensorDomainAxisIndex(i), " is a reduction domain.");
        break;
      }
    }
  }

  const auto td_pos = producer_cd->getTensorDomainPos(producer_compute_at_axis);

  std::cerr << "replayCasP: consumer: " << consumer
            << ", producer: " << producer << ", producer_cd: " << *producer_cd
            << ", CD pos: " << producer_compute_at_axis
            << ", TD pos: " << td_pos << std::endl;

  TORCH_INTERNAL_ASSERT(
      producer_compute_at_axis >= 0 &&
      (unsigned int)producer_compute_at_axis <= producer_cd->nDims(),
      "Invalid axis in transform replayCasP.");

  const auto& producer_domain = producer_cd->axes();

  // producer ids we need to match in consumer
  std::vector<IterDomain*> producer_CA_ids(
      producer_domain.begin(),
      producer_domain.begin() + producer_compute_at_axis);

  //producer_CA_ids = TensorDomain::noReductions(producer_CA_ids);

  std::cerr << "producer CA ids: " << producer_CA_ids << std::endl;

  // Grab root domains of producer and consumer
  std::vector<IterDomain*> consumer_root = consumer->getRootDomain();
  std::cerr << "Consumer root: " << consumer_root << std::endl;
  auto consumer_contig = consumer->contiguity();
  std::vector<bool> ca_placeholder = consumer->placeholder();
  bool placeholder_inserted = insertMissingDomains(consumer_root, consumer_contig, ca_placeholder,
                                                   producer->getRootDomain(),
                                                   getRootCAIDs(producer->domain(), td_pos),
                                                   false, consumer->hasRFactor());

#if 0
  // If producer has an rfactor root, that's what will match the consumer
  std::vector<IterDomain*> producer_root = producer->getMaybeRFactorDomain();
  // Figure out all inputs required to generate the compute_at dimensions. We
  // need all deps because inputs on producer may be in getRootDomain, but we
  // may need in rFactorDomain
  std::unordered_set<Val*> all_CA_id_deps = DependencyCheck::getAllValsBetween(
      {producer_root.begin(), producer_root.end()},
      {producer_CA_ids.begin(), producer_CA_ids.end()});

  // Figure out which root IDs we need:
  std::unordered_set<IterDomain*> producer_CA_root_ids;
  for (IterDomain* id : producer_root) {
    if (all_CA_id_deps.find(id) != all_CA_id_deps.end()) {
      producer_CA_root_ids.emplace(id);
      std::cerr << "producer CA root id: " << id << std::endl;
    }
  }
#else
  auto input_ids = IterVisitor::getInputsTo(
      std::vector<Val*>(producer_CA_ids.begin(), producer_CA_ids.end()));
  std::unordered_set<IterDomain*> producer_CA_root_ids{
    ir_utils::filterByType<IterDomain>(input_ids).begin(),
    ir_utils::filterByType<IterDomain>(input_ids).end()};
  for (const auto& id: producer_CA_root_ids) {
    std::cerr << "producer CA root id: " << id << std::endl;
  }
#endif

#if 0
  auto replay_root_map =
      TensorDomain::mapRootPtoC(producer, consumer, producer_CA_root_ids);
#else
  auto replay_root_map =
      producer_cd->mapRootDomain(consumer_root, producer_CA_root_ids);
#endif

  // Track which root axes in producer we will send to replay
  std::unordered_set<IterDomain*> consumer_roots4replay;
  for (auto entry : replay_root_map) {
    consumer_roots4replay.emplace(entry.second);
    std::cerr << "consumer root for replay: " << entry.second
              << " (<- " << entry.first << ")" << std::endl;
  }

  // Instead of replaying from the root, lets try to forward the history of
  // consumer if they match ops on producer. Enforce if we modify an rfactor
  // axis that those ops match.
  BestEffortReplay forward_replay(
      consumer->domain(), producer_CA_ids, replay_root_map);

  id_map forwarded_replay_map;
  for (auto entry : forward_replay.getReplay()) {
    if (forward_replay.getUnorderedLeafIDs().find(entry.second) !=
        forward_replay.getUnorderedLeafIDs().end())
      forwarded_replay_map[entry.first] = entry.second;
  }

  // Replay producer dimensions.
  ReplayTransformations replay_CasP(
      producer_CA_ids, forwarded_replay_map, false);

  for (const auto& replay: replay_CasP.getReplay()) {
    std::cerr << "ReplayTransformation: " << replay.first << " -> " << replay.second << std::endl;
  }

  auto leaf_ids(replay_CasP.getUnorderedLeafIDs());
  auto crossover_map(replay_CasP.getLeafMap());

  // Remove all ids that map to the compute at axis, we're going to replay the
  // rest
#if 0
  for (auto p_id : producer_CA_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    TORCH_INTERNAL_ASSERT(
        it != replay_CasP.getReplay().end(),
        "Could not find axis, ",
        p_id,
        ", requested in replay.");
    auto replayed_id = it->second;
    if (leaf_ids.find(replayed_id) != leaf_ids.end()) {
      leaf_ids.erase(replayed_id);
    }
  }
#else
  for (auto replayed_id: replayMapFind(replay_CasP.getReplay(), producer_CA_ids)) {
    if (leaf_ids.find(replayed_id.first) != leaf_ids.end()) {
      leaf_ids.erase(replayed_id.first);
      //crossover_map.erase(replayed_id.first);
    }
  }
#endif

  // leaf_ids now contains all consumer ID products that are not used to satisfy
  // the computeAt Turn into a  map so we can play forward these IDs in consumer
  // (if possible):
  id_map consumer_self_replay_map;
  for (auto entry : leaf_ids)
    consumer_self_replay_map[entry.first] = entry.first;

  // Any root domain that was not used to generate computeIDs we can also put in
  // the map to forward their transformations.
  for (auto consumer_root_id : consumer_root)
    if (consumer_roots4replay.find(consumer_root_id) ==
        consumer_roots4replay.end())
      consumer_self_replay_map[consumer_root_id] = consumer_root_id;

  // Play forward transformations all consumer IDs we can
  auto consumer_replayed_leaves = BestEffortReplay(
      consumer->domain(), consumer->domain(), consumer_self_replay_map);

  /*
   * Accumulate axes in to the new domain in the following order, making sure to
   * avoid any duplicates:
   *
   * (1) replay_PasC.getReplay holds mappings from axes in consumer compute at
   * axes -> corresponding generated axes in producer
   *
   * (2) Any axes that were not added, that can be mapped directly from an ID in
   * producer->domain(). These are axes that were "fully replayed" relative to
   * the producer, even though it wasn't in the computeAt range.
   *
   * producer_replayed_leaves now contain ids that we tried to forward
   * back to what they were in producer. If they couldn't be forwarded they're
   * left in their "most forwarded" form which may be just a remainder of the
   * transformation required to generate the computeAt axes.
   *
   * (3) Axes in producer->domain() that are in producer_replayed_leaves
   *
   * (4) Axes not in producer->domain() that are in producer_replayed_leaves
   *
   * TODO: Should (2) and (3) be swapped?
   */

  std::vector<IterDomain*> new_IDs;
  std::unordered_set<IterDomain*> used_IDs;
  std::vector<size_t> td2cd_map;
  // Add axes in (1)
#if 0
  for (auto p_id : producer_CA_ids) {
    auto it = replay_CasP.getReplay().find(p_id);
    TORCH_INTERNAL_ASSERT(
        it != replay_CasP.getReplay().end(),
        "Could not find axis, ",
        p_id,
        ", requested in replay.");
    auto replayed_id = it->second;
    new_IDs.push_back(replayed_id);
    used_IDs.emplace(replayed_id);
  }
#else
  for (auto replayed_id: replayMapFind(replay_CasP.getReplay(), producer_CA_ids)) {
    new_IDs.push_back(replayed_id.first);
    used_IDs.emplace(replayed_id.first);
    td2cd_map.push_back(replayed_id.second);
    std::cerr << "(1): " << replayed_id.first << std::endl;
  }
#endif

  auto num_shared_axes = new_IDs.size();

  // Add axes in (2)
#if 0
  for (auto p_id : producer->domain()) {
    auto it = replay_CasP.getReplay().find(p_id);
    if (it != replay_CasP.getReplay().end()) {
      IterDomain* replayed_id = it->second;
      if (used_IDs.find(replayed_id) == used_IDs.end()) {
        new_IDs.push_back(replayed_id);
        used_IDs.emplace(replayed_id);
      }
    }
  }
#else
  for (auto replayed_id: replayMapFind(replay_CasP.getReplay(), producer_cd->axes())) {
    // If the leaf id from ReplayTransformation is used to move
    // forward with BestEffortReplay, it is not a final ID. It is
    // initialy set as a leaf for BestEffortReplay, and so if it does
    // not remain there, it means the ID is used to replay.
    if (consumer_replayed_leaves.getUnorderedLeafIDs().find(replayed_id.first) ==
        consumer_replayed_leaves.getUnorderedLeafIDs().end()) {
      continue;
    }
    if (used_IDs.find(replayed_id.first) == used_IDs.end()) {
      new_IDs.push_back(replayed_id.first);
      used_IDs.emplace(replayed_id.first);
      std::cerr << "(2): " << replayed_id.first << std::endl;
    }
  }
#endif

  // Add axes in (3)
  for (auto id : consumer->domain()) {
    if (consumer_replayed_leaves.getUnorderedLeafIDs().find(id) !=
        consumer_replayed_leaves.getUnorderedLeafIDs().end()) {
      if (used_IDs.find(id) == used_IDs.end()) {
        new_IDs.push_back(id);
        used_IDs.emplace(id);
        std::cerr << "(3): " << id << std::endl;
      }
    }
  }

  // Add axes in (4)
  for (auto id : consumer_replayed_leaves.getLeafIDs()) {
    if (used_IDs.find(id) == used_IDs.end()) {
      new_IDs.push_back(id);
      std::cerr << "(4): " << id << std::endl;
    }
  }

  std::cerr << "New IDs: " << new_IDs << std::endl;

  TensorDomain* replayed = nullptr;

  if (placeholder_inserted) {
    TORCH_INTERNAL_ASSERT(consumer->getRFactorDomain().size() == 0);
    replayed = new TensorDomain(
        consumer_root,
        consumer->getRFactorDomain(),
        new_IDs,
        consumer_contig,
        ca_placeholder,
        crossover_map);
  } else {
    replayed = new TensorDomain(
        consumer->getRootDomain(),
        consumer->getRFactorDomain(),
        new_IDs,
        consumer->contiguity(),
        ca_placeholder,
        crossover_map);
  }

  std::cerr << "replayCasP done: " << replayed
            << ", num shared axes: " << num_shared_axes
            << std::endl;

  // TODO: computeAt position should be on a position in the compute domain.
  //return {replayed,
  //TensorDomain::noReductions(producer_CA_ids).size()};
  return {replayed, num_shared_axes, td2cd_map,
          crossover_map};
}

// replay Producer as Consumer
std::tuple<TensorView*, unsigned int, unsigned int> TransformReplay::replayPasC(
    TensorView* producer,
    TensorView* consumer,
    int compute_at_axis) {
  std::cerr << "replayPasC: " << producer << " -> " << consumer << std::endl;

  // If this is a reduction operation, we may call transform_replay on the
  // tensor view. When this happens, just return thet target view.
  if (producer == consumer) {
    return {producer, 0, 0};
  }

  auto replay =
      replayPasC(producer->domain(), consumer->domain(), consumer->getComputeDomain(),
                 compute_at_axis);
  producer->setDomain(std::get<0>(replay));
  producer->getComputeDomain()->computeAt(producer->domain(),
                                          std::get<1>(replay),
                                          consumer->getComputeDomain(),
                                          compute_at_axis,
                                          std::get<2>(replay),
                                          std::get<3>(replay));
  producer->getComputeDomain()->registerAsDependent(consumer->getComputeDomain());
  std::cerr << "producer new CD: " << *producer->getComputeDomain() << std::endl;
  return {producer, std::get<1>(replay), producer->getComputeDomain()->getComputeAtPos()};
}

std::tuple<TensorView*, unsigned int, unsigned int> TransformReplay::replayCasP(
    TensorView* consumer,
    TensorView* producer,
    int compute_at_axis) {
  std::cerr << "replayCasP: " << consumer << " -> " << producer
            << " at " << compute_at_axis
            << std::endl;

  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (consumer == producer)
    return {consumer, 0, 0};

  auto replay =
      replayCasP(consumer->domain(), producer->domain(), producer->getComputeDomain(),
                 compute_at_axis);
  consumer->setDomain(std::get<0>(replay));
  consumer->getComputeDomain()->computeAt(consumer->domain(),
                                          std::get<1>(replay),
                                          producer->getComputeDomain(),
                                          compute_at_axis,
                                          std::get<2>(replay),
                                          std::get<3>(replay));
  consumer->getComputeDomain()->registerAsDependent(producer->getComputeDomain());
  std::cerr << "consumer new CD: " << *consumer->getComputeDomain() << std::endl;
  return {consumer, std::get<1>(replay), consumer->getComputeDomain()->getComputeAtPos()};
}

} // namespace fuser
} // namespace jit
} // namespace torch
