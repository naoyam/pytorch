#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

/*
 * 1) Take the reference, trace back its domain history to get all the
 * split/merge/reorder calls, as well as its original domain. Get the
 * original domain of the target as well.
 *
 * 2) We only need compute_at_axis and earlier dimensions to match for
 * compute_at. Therefore, we want to find all original axes that must have
 * been modified in order to produce the axes below compute_at_axis. We take a
 * bool vector called influence, and mark axes below compute_at_axis as true,
 * and all others as false. This vector is propagated up through
 * split/merge/reorder if split/merge/reorder output a marked axis, their
 * input will be marked as well. This marks all original axes required to be
 * modified to produce the axes below compute_at_axis.
 *
 * 3) We take the ordered list of split/merge/reorder and the influence vector
 * on the inputs and we apply split/merge/reorder operations on the
 * replay_target. We also forward propagate the influence vector again (as this
 * time it could be different than originally marked), a map from "fake axes"
 * (refrence axes corresponding to the full replay) to real axes (axes produced
 * by running the selected split/merge/reorder operations). Reorder replay's can
 * actually be partial and non-equivelent to the original, as some axes may
 * never have been produced based on split, and we don't want to reorder axes
 * outside of compute_at_axis.
 *
 */
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

  /* STEP 1 */
  // Reset the tensor domain of the target, this is the only way we can be
  // certain That we can actually replay the ops of ref.
  // Trace back to the root TensorDomain's of ref and target
  replay_target = replay_target->rootDomain();

  /* STEP 2 */
  // Mark compute_at_axis and below as "influenced", trace back through
  // operations, and map these axes to the ref root axis that were modified to
  // produce these axis
  // As we trace the ref, record the operations to go from replay_ref ->
  // ref_root, save in "record"
  TensorDomain* ref_root = replay_ref->rootDomain();

  std::vector<bool> root_influence_vector(replay_ref->nDims(), false);
  for(int i=0; i<compute_at_axis; i++)
    root_influence_vector[i] = true;

  // We're going to save a copy of this vector, class member influnce will be
  // used during replay to forward propagate influence.
  root_influence_vector = TransformIter::getRootInfluence(replay_ref, root_influence_vector);

  // In lowering code we replay domains on to copies of themselves. This is done
  // during the replacement of symbolic sizes for inputs and outputs (i.e. TV0[
  // iS{i3}, iS{i4} ] -> TV0[ TV0.size[0], TV0.size[1] ]). If number of axes and
  // reduction axes match exactly include the reduction axes during replay.
  // Otherwise assume consumer/producer relationship and ignore reduction axes
  // on target.
  bool include_reductions = replay_target->nDims() == ref_root->nDims();
  std::vector<int> axis_map(ref_root->nDims(), -1);
  decltype(replay_target->nDims()) it = 0, ir = 0;
  while(it < replay_target->nDims() && ir < ref_root->nDims()){
    if(include_reductions || !replay_target->axis(it)->isReduction()){
      if(root_influence_vector[ir]){
        axis_map[ir++] = it++;
      }else{
        axis_map[ir++] = -1;
        it++;
      }
    }else{
      it++;
    }
  }

  /* STEP 3 */
  // Replay operations while forward propagating influence. The resulting
  // influence can be different in forward propagation, than in backward
  // propagation depending on the combination of merge/split/reorder nodes
  // There are multiple things we have to track here. We need to track
  // the propagation of axes for all operations, though we only want to
  // actually execute those based on influence. If we didn't track all
  // axes, we wouldn't know what axis split/merge/reorder are referencing
  // as they're relative to the "full" replay that produced the reference.
  TensorDomain* replayed = TransformIter::replay(replay_target, TransformIter::getHistory(replay_ref), axis_map);
  
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
