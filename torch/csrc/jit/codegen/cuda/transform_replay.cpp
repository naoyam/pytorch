#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

TensorDomain* TransformReplay::runReplay(
    TensorDomain* replay_ref,
    TensorDomain* replay_target,
    int compute_at_axis) {
  if (compute_at_axis < 0)
    compute_at_axis += int(replay_ref->nDims()) + 1;

  TORCH_CHECK(
      compute_at_axis >= 0 && compute_at_axis < int(replay_ref->nDims()) + 1,
      "Transform replay ",
      replay_ref,
      " on ",
      replay_target,
      " with axis ",
      compute_at_axis,
      " is an illegal transform.");

  // Grab roots of both domains
  TensorDomain* target_root = replay_target->rootDomain();
  TensorDomain* ref_root = replay_ref->rootDomain();

  // Mark axes under compute_at as needing to be modified
  std::vector<bool> ref_influence(replay_ref->nDims(), false);
  for (int i = 0; i < compute_at_axis; i++)
    ref_influence[i] = true;

  // Check if we should include reduction axes from either domain
  // either include or no reductions found in either
  bool include_reductions = target_root->nDims() == ref_root->nDims();

  // Check which axes in ref_root need to be modified to honor transformations
  // to compute at axis
  ref_influence = TransformIter::getRootInfluence(replay_ref, ref_influence);

  // We want to see what axes in target_replay would need to be modified, we
  // want to keep those that don't need to be modified by replay
  std::vector<bool> target_influence(target_root->rootDomain()->nDims(), false);

  // Set up replay_axis_map from ref_root to original target_root this will be
  // modified later, if we don't do this now, we'd need the following logic
  // replicated later
  std::vector<int> replay_axis_map(ref_root->nDims());

  // Setup target_influence vector on root for replay
  decltype(target_influence.size()) it = 0, ir = 0;
  while (it < target_influence.size() && ir < ref_root->nDims()) {
    bool isreduction = target_root->axis(it)->isReduction();
    bool isrfactor = target_root->axis(it)->isRFactorProduct();
    if (!isreduction || (include_reductions && isreduction)) {
      if (ref_influence[ir]) {
        TORCH_CHECK(
            !isrfactor,
            "Invalid transformation found, an rfactor axis cannot be modified during replay.",
            " There likely is an invalid compute at. Found during replay of ",
            replay_ref,
            " onto ",
            target_root,
            " at comptue at axis ",
            compute_at_axis);
        replay_axis_map[ir] = it;
      } else {
        replay_axis_map[ir] = -1;
      }
      target_influence[it++] = ref_influence[ir++];
    } else {
      target_influence[it++] = false;
    }
  }

  // Run replay on target for axes that won't be modified by this replay
  // target_influence = TransformIter::replayInfluence(
  //      TransformIter::getHistory(replay_target), target_influence);

  // target_influence = TransformIter::getRootInfluence(
  //      replay_target, target_influence);

  // Set up target_axis_map to replay target transformations for axes that are
  // not modified by replay
  std::vector<int> target_axis_map(target_influence.size());
  for (decltype(target_axis_map.size()) i{0}; i < target_axis_map.size(); i++)
    target_axis_map[i] = target_influence[i] ? -1 : i;

  // Replay axes that won't be modified by replay, this is what we will replay
  // transformations on
  replay_target = TransformIter::replay(
      target_root, TransformIter::getHistory(replay_target), target_axis_map);

  // Check how our axes have been modified
  std::unordered_map<IterDomain*, int> new_position;
  for (decltype(replay_target->nDims()) i{0}; i < replay_target->nDims(); i++) {
    new_position[replay_target->axis(i)] = i;
  }

  // Adjust axis map from being on target_root to being on replay_target
  for (decltype(replay_axis_map.size()) i{0}; i < replay_axis_map.size(); i++) {
    if (replay_axis_map[i] == -1)
      continue;
    auto ax = target_root->axis(replay_axis_map[i]);
    TORCH_INTERNAL_ASSERT(
        new_position.find(ax) != new_position.end(),
        "Error hit during transform replay, could not find ",
        ax,
        " expected in root domain.");
    replay_axis_map[i] = new_position[ax];
  }

  // Run replay covering compute at axes.
  TensorDomain* replayed = TransformIter::replay(
      replay_target, TransformIter::getHistory(replay_ref), replay_axis_map);

  for (decltype(replayed->nDims()) i{0}; i < compute_at_axis; i++)
    if (!include_reductions && replayed->axis(i)->isReduction())
      TORCH_CHECK(
          false,
          "Generated a compute_at dependency where a reduction would be used before computed.");

  return replayed;
}

// Replay producer as consumer.
TensorDomain* TransformReplay::replayPasC(
    TensorDomain* producer,
    TensorDomain* consumer,
    int compute_at_axis) {
  // Want producer root with no reductions, rfactor included
  TensorDomain* producer_root = producer->rootDomain();
  // Producer root still has reductions

  // Want full consumer root, even before rfactor
  TensorDomain* consumer_root = TransformIter::getRoot(consumer);

  // We want to see which axes in the consumer root were modified to create axes
  // < compute_at_axis
  std::vector<bool> consumer_influence(consumer->nDims(), false);
  for (int i = 0; i < compute_at_axis; i++)
    consumer_influence[i] = true;

  // Check which axes in ref_root need to be modified to honor transformations
  // to compute at axis
  std::vector<bool> consumer_root_influence =
      TransformIter::getRootInfluence(consumer, consumer_influence);

  // We have the influence on the consumer root, we need it on producer, we
  // want to keep those axes that don't need to be modified by the replay
  std::vector<bool> producer_root_influence(
      producer->rootDomain()->nDims(), false);

  // Map is based on producer
  std::vector<int> replay_axis_map(consumer_root->nDims());

  // Setup producer_root_influence vector on root for replay
  decltype(producer_root_influence.size()) ip = 0, ic = 0;
  while (ip < producer_root_influence.size() && ic < consumer_root->nDims()) {
    bool is_reduction = producer_root->axis(ip)->isReduction();
    if (is_reduction) {
      producer_root_influence[ip++] = false;
    } else {
      if (consumer_root_influence[ic]) {
        replay_axis_map[ic] = ip;
      } else {
        replay_axis_map[ic] = -1;
      }
      producer_root_influence[ip++] = consumer_root_influence[ic++];
    }
  }

  for (decltype(producer_root->nDims()) i{0}; i < producer_root->nDims(); i++)
    TORCH_INTERNAL_ASSERT(
        !(producer_root_influence[i] &&
          producer_root->axis(i)->isRFactorProduct()),
        "An illegal attempt to modify an rfactor axis detected.");

  // We should have hit the end of the consumer root domain
  TORCH_INTERNAL_ASSERT(
      ic == consumer_root->nDims(),
      "Error when trying to run replay, didn't reach end of consumer/target root.");

  TORCH_INTERNAL_ASSERT(
      producer_root_influence.size() == producer_root->nDims(),
      "Error detected during replay, expected matching sizes of influence map to root dimensions.");

  std::vector<int> producer_replay_map(producer_root->nDims());
  for (decltype(producer_replay_map.size()) i{0};
       i < producer_replay_map.size();
       i++)
    producer_replay_map[i] = producer_root_influence[i] ? -1 : i;

  // Replay axes that won't be modified by transform replay
  TensorDomain* producer_replay_root = TransformIter::replay(
      producer_root, TransformIter::getHistory(producer), producer_replay_map);

  // Record axes positions.
  std::unordered_map<IterDomain*, int> new_position;
  for (decltype(producer_replay_root->nDims()) i{0};
       i < producer_replay_root->nDims();
       i++)
    new_position[producer_replay_root->axis(i)] = i;

  std::unordered_map<int, int> root_axis_map;
  // reorder producer_replay_root to respect replay_axis_map
  for (decltype(replay_axis_map.size()) i{0}; i < replay_axis_map.size(); i++) {
    if (replay_axis_map[i] == -1)
      continue;
    auto ax = producer_root->axis(replay_axis_map[i]);
    TORCH_INTERNAL_ASSERT(
        new_position.find(ax) != new_position.end(),
        "Error hit during transform replay, could not find ",
        ax,
        " expected in root domain.");
    root_axis_map[new_position[ax]] = i;
  }
  producer_replay_root = producer_replay_root->reorder(root_axis_map);

  // Finally replay producer as consumer on marked axes
  TensorDomain* replayed = TransformIter::replay(
      producer_replay_root,
      TransformIter::getHistory(consumer),
      replay_axis_map);

  TORCH_INTERNAL_ASSERT(
      std::none_of(
          replayed->domain().begin(),
          replayed->domain().begin() + compute_at_axis,
          [](IterDomain* id) { return id->isReduction(); }),
      "Reduction axes found within compute_at_axis in replay of producer.");

  return replayed;
}

// Replay producer as consumer.
TensorDomain* TransformReplay::replayCasP(
    TensorDomain* consumer,
    TensorDomain* producer,
    int compute_at_axis) {
  // Want producer root with no reductions, rfactor included
  TensorDomain* producer_root = TransformIter::replayRFactor2Root(producer);
  // Producer root still has reductions

  // Want full consumer root, even before rfactor
  TensorDomain* consumer_root = TransformIter::getRoot(consumer);

  // We want to see which axes in the producer root were modified to create axes
  // < compute_at_axis
  std::vector<bool> producer_influence(producer->nDims(), false);
  for (int i = 0; i < compute_at_axis; i++)
    producer_influence[i] = true;

  // Check which axes in ref_root need to be modified to honor transformations
  // to compute at axis
  std::vector<bool> producer_root_influence =
      TransformIter::getRootInfluence(producer, producer_influence);

  for (decltype(producer_root->nDims()) i{0}; i < producer_root->nDims(); i++) {
    TORCH_INTERNAL_ASSERT(
        !(producer_root_influence[i] && producer_root->axis(i)->isReduction()),
        "Error during replay, likely due to an illegal bad computeAt.");
  }

  // We have the influence on the producer root, we need it on consumer, we
  // want to keep those axes that don't need to be modified by the replay
  std::vector<bool> consumer_root_influence(
      consumer->rootDomain()->nDims(), false);

  // Producer -> consumer axis map
  std::vector<int> replay_axis_map(producer_root->nDims());

  // Setup consumer_root_influence vector on root for replay
  decltype(consumer_root_influence.size()) ip = 0, ic = 0;
  while (ic < consumer_root_influence.size() && ip < producer_root->nDims()) {
    bool is_reduction = producer_root->axis(ip)->isReduction();
    if (is_reduction) {
      replay_axis_map[ip++] = -1;
      continue;
    }
    if (producer_root_influence[ip]) {
      replay_axis_map[ip] = ic;
    } else {
      replay_axis_map[ip] = -1;
    }
    consumer_root_influence[ic++] = producer_root_influence[ip++];
  }

  // We should have hit the end of the consumer root domain
  TORCH_INTERNAL_ASSERT(
      ic == consumer_root->nDims(),
      "Error when trying to run replay, didn't reach end of consumer/reference root.",
      ic,
      " ",
      consumer_root->nDims());

  TORCH_INTERNAL_ASSERT(
      consumer_root_influence.size() == consumer_root->nDims(),
      "Error detected during replay, expected matching sizes of influence map to root dimensions.");

  std::vector<int> consumer_replay_map(consumer_root->nDims());
  for (decltype(consumer_replay_map.size()) i{0};
       i < consumer_replay_map.size();
       i++)
    consumer_replay_map[i] = consumer_root_influence[i] ? -1 : i;

  // Replay axes that won't be modified by transform replay
  TensorDomain* consumer_replay_root = TransformIter::replay(
      consumer_root, TransformIter::getHistory(consumer), consumer_replay_map);

  // Record axes positions.
  std::unordered_map<IterDomain*, int> new_position;
  for (decltype(consumer_replay_root->nDims()) i{0};
       i < consumer_replay_root->nDims();
       i++)
    new_position[consumer_replay_root->axis(i)] = i;

  std::unordered_map<int, int> root_axis_map;
  // reorder consumer_replay_root to respect replay_axis_map
  for (decltype(replay_axis_map.size()) i{0}; i < replay_axis_map.size(); i++) {
    if (replay_axis_map[i] == -1)
      continue;
    auto ax = consumer_root->axis(replay_axis_map[i]);
    TORCH_INTERNAL_ASSERT(
        new_position.find(ax) != new_position.end(),
        "Error hit during transform replay, could not find ",
        ax,
        " expected in root domain.");
    root_axis_map[new_position[ax]] = i;
  }
  consumer_replay_root = consumer_replay_root->reorder(root_axis_map);

  // Finally replay consumer as producer on marked axes
  return TransformIter::replay(
      consumer_replay_root,
      TransformIter::getHistory(producer),
      replay_axis_map);
}

TensorView* TransformReplay::runReplay(
    TensorView* replay_ref,
    TensorView* replay_target,
    int compute_at_axis) {
  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (replay_ref == replay_target)
    return replay_target;

  TensorDomain* td =
      runReplay(replay_ref->domain(), replay_target->domain(), compute_at_axis);
  replay_target->setDomain(td);
  return replay_target;
}

TensorView* TransformReplay::replay(
    TensorView* replay_ref,
    TensorView* replay_target,
    int compute_at_axis) {
  TransformReplay tr;
  tr.runReplay(replay_ref, replay_target, compute_at_axis);
  return replay_target;
}

TensorDomain* TransformReplay::replay(
    TensorDomain* replay_ref,
    TensorDomain* replay_target,
    int compute_at_axis) {
  TransformReplay tr;
  tr.runReplay(replay_ref, replay_target, compute_at_axis);
  return replay_target;
}

TensorView* TransformReplay::replayPasC(
    TensorView* producer,
    TensorView* consumer,
    int compute_at_axis) {
  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (producer == consumer)
    return producer;

  TensorDomain* td =
      replayPasC(producer->domain(), consumer->domain(), compute_at_axis);
  producer->setDomain(td);
  return producer;
}

TensorView* TransformReplay::replayCasP(
    TensorView* consumer,
    TensorView* producer,
    int compute_at_axis) {
  // If this is a reduction operation, we may call transform_replay on the same
  // tensor view. When this happens, just return thet target view.
  if (consumer == producer)
    return consumer;
  TensorDomain* td =
      replayCasP(consumer->domain(), producer->domain(), compute_at_axis);
  consumer->setDomain(td);
  return consumer;
}

} // namespace fuser
} // namespace jit
} // namespace torch
