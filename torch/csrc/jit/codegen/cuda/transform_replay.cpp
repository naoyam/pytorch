#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

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

  TensorDomain* target_root = replay_target->rootDomain();
  TensorDomain* ref_root = replay_ref->rootDomain();

  std::vector<bool> ref_influence(replay_ref->nDims(), false);
  for(int i=0; i<compute_at_axis; i++)
    ref_influence[i] = true;
  
  bool include_reductions = target_root->nDims() == ref_root->nDims();

  // Check which axes in ref_root need to be modified to honor transformations to compute at axis
  ref_influence = TransformIter::getRootInfluence(replay_ref, ref_influence);

  // We want to see what axes in target_replay would need to be modified, we want to keep those that don't need to be modified by replay
  std::vector<bool> target_influence(target_root->rootDomain()->nDims(), false);

  // Set up replay_axis_map from ref_root to original target_root this will be modified later, if we don't do this now, we'd need the following logic replicated later
  std::vector<int> replay_axis_map(ref_root->nDims());

  // Setup target_influence vector on root for replay
  decltype(target_influence.size()) it = 0, ir = 0;
  while(it < target_influence.size() && ir < ref_root->nDims()){
    bool isreduction = target_root->axis(it)->isReduction();
    bool isrfactor = target_root->axis(it)->isRFactorProduct();
    if( !isreduction || ( include_reductions && isreduction ) ){
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
    }else{
      target_influence[it++] = false;
    }
  }

  // Run replay on target for axes that won't be modified by this replay
  // target_influence = TransformIter::replayInfluence(
  //      TransformIter::getHistory(replay_target), target_influence);
  
  // target_influence = TransformIter::getRootInfluence(
  //      replay_target, target_influence);

  // Set up target_axis_map to replay target transformations for axes that are not modified by replay
  std::vector<int> target_axis_map(target_influence.size());
  for(decltype(target_axis_map.size()) i{0}; i < target_axis_map.size(); i++)
    target_axis_map[i] = target_influence[i] ? -1 : i;

  // Replay axes that won't be modified by replay, this is what we will replay transformations on
  replay_target = TransformIter::replay(
      target_root, TransformIter::getHistory(replay_target), target_axis_map);

  // Check how our axes have been modified
  std::unordered_map<IterDomain*, int> new_position;
  for (decltype(replay_target->nDims()) i{0}; i < replay_target->nDims(); i++) {
    new_position[replay_target->axis(i)] = i;
  }
  
  // Adjust axis map from being on target_root to being on replay_target
  for(decltype(replay_axis_map.size()) i{0}; i<replay_axis_map.size(); i++){
    if(replay_axis_map[i] == -1)
      continue;
    auto ax = target_root->axis(replay_axis_map[i]);
    TORCH_INTERNAL_ASSERT(new_position.find(ax) != new_position.end(),
      "Error hit during transform replay, could not find ",ax, " expected in root domain.");
    replay_axis_map[i] = new_position[ax];
  }

  // Run replay covering compute at axes.
  TensorDomain* replayed = TransformIter::replay(replay_target, TransformIter::getHistory(replay_ref), replay_axis_map);
  
  for (decltype(replayed->nDims()) i{0}; i < compute_at_axis; i++)
    if (!include_reductions && replayed->axis(i)->isReduction())
      TORCH_CHECK(
          false,
          "Generated a compute_at dependency where a reduction would be used before computed.");

  return replayed;
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

TensorView* TransformReplay::fullReplay(
    TensorView* replay_ref,
    TensorView* replay_target) {
  TransformReplay tr;
  return tr.runReplay(replay_ref, replay_target, -1);
}

TensorDomain* TransformReplay::fullReplay(
    TensorDomain* replay_ref,
    TensorDomain* replay_target) {
  TransformReplay tr;
  return tr.runReplay(replay_ref, replay_target, -1);
}

} // namespace fuser
} // namespace jit
} // namespace torch
