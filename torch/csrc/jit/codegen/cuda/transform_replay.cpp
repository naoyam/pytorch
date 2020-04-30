#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

namespace{

struct Influence  : public TransformIter {
private:
  /*
   * Functions to backward propagate influence from split/merge/reorder
   */

TensorDomain* replayBackward(Split* split, TensorDomain* td) override {
  int axis = split->axis();
  TORCH_INTERNAL_ASSERT(
      axis + 1 < influence.size(),
      "Error during replay backwards, influence is not sized correctly.");
  influence[axis] = influence[axis] | influence[axis + 1];
  influence.erase(influence.begin() + axis + 1);
  return split->in();
}

TensorDomain* replayBackward(Merge* merge, TensorDomain* td) override {
  int axis = merge->axis();
  TORCH_INTERNAL_ASSERT(
      axis < influence.size(),
      "Error during replay backwards, influence is not sized correctly.");
  influence.insert(influence.begin() + axis + 1, influence[axis]);
  return merge->in();
}

TensorDomain* replayBackward(Reorder* reorder, TensorDomain* td) override {
  // pos2axis[new_pos] = old_pos Generate new axis2pos map
  const std::vector<int>& pos2axis = reorder->pos2axis();

  std::vector<bool> reorder_influence(influence.size(), false);
  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    int new_pos = i;
    int old_pos = pos2axis[i];
    TORCH_INTERNAL_ASSERT(
        new_pos < influence.size() && old_pos < reorder_influence.size(),
        "Error during replay backwards, influence is not sized correctly.");
    reorder_influence[old_pos] = influence[new_pos];
  }

  influence = reorder_influence;
  return reorder->in();
}

std::vector<bool> influence;

Influence(TensorDomain* td, std::vector<bool> td_influence):influence(td_influence){
  TransformIter::runBackward(td);
}

public:

static std::vector<bool> compute(TensorDomain* td, std::vector<bool> td_influence){
  TORCH_INTERNAL_ASSERT(
      td_influence.size() == td->nDims(),
      "Tried to compute backward influence computation, but recieved an influence vector that does not match the TensorDomain size.");
  Influence inf(td, td_influence);
  return inf.influence;
}

}; //struct Influence



struct Replay : public TransformIter {

/*
 * Replay functions, takes a TensorDomain and steps through the operations in
 * "record" based on influence axes. Will also update influence and propagate
 * it forward.
 */
TensorDomain* replay(Split* split, TensorDomain* td) {
  int saxis = split->axis();

  TORCH_INTERNAL_ASSERT(
      saxis >= 0 && saxis < axis_map.size(),
      "TransformReplay tried to modify an axis out of range, recieved ",
      saxis,
      " but this value should be >=0 and <",
      axis_map.size());

  if (axis_map[saxis] == -1) {
    // don't modify path, we need an extra axis as there would have been one
    // there, but we shouldn't modify it.
    axis_map.insert(axis_map.begin() + saxis + 1, -1);
    return td;
  }

  // Recreate the merge, axis is relative to the td
  int axis = axis_map[saxis];
  // Move indices up as we now have an extra axis
  std::transform(
      axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
        return i > axis ? i + 1 : i;
      });

  // Insert new axis in map
  axis_map.insert(axis_map.begin() + saxis + 1, axis_map[saxis] + 1);

  TORCH_INTERNAL_ASSERT(
      split->factor()->isConst(),
      "Cannot replay split as it's not based on a const value.");
  td = td->split(axis, split->factor()->value().value());
  
  return td;
}

TensorDomain* replay(Merge* merge, TensorDomain* td) {

  int maxis = merge->axis();

  TORCH_INTERNAL_ASSERT(
      maxis >= 0 && maxis < axis_map.size(),
      "TransformReplay tried to modify an axis out of range, recieved ",
      maxis,
      " but this value should be >= 0 and < axis_map.size()");

  // Get axis relative to what we actually have in td.
  int axis = axis_map[maxis];
  int axis_p_1 = axis_map[maxis + 1];
  // If either dim is not to be touch, set both not to be touched
  axis = axis_p_1 == -1 ? -1 : axis;
  axis_map[maxis] = axis;

  // Remove axis from axis_map as in original transformations it didn't exist
  axis_map.erase(axis_map.begin() + maxis + 1);

  // Don't modify:
  if (axis == -1)
    return td;

  // Move indices down as we're removing an axis
  std::transform(
      axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
        return i > axis ? i - 1 : i;
      });

  return td->merge(axis);

}


TensorDomain* replay(Reorder* reorder, TensorDomain* td) {
  const std::vector<int>& pos2axis_orig = reorder->pos2axis();

  // We want to convert pos2axis to something with td->nDims which it isn't
  // guarenteed to be
  // pos2axis[new_position] = old_position
  std::vector<int> pos2axis(td->nDims(), -1);

  std::set<int> old_pos_left;
  for (decltype(axis_map.size()) i{0}; i < axis_map.size(); i++)
    old_pos_left.emplace(i);

  for (decltype(pos2axis_orig.size()) i{0}; i < pos2axis_orig.size(); i++) {
    int old_pos = axis_map[i];
    int new_pos = pos2axis_orig[i];

    if (old_pos != -1) {
      pos2axis[new_pos] = old_pos;
      TORCH_INTERNAL_ASSERT(
          old_pos_left.find(old_pos) != old_pos_left.end(),
          "Internal error, duplicate in reorder map found.");
      old_pos_left.erase(old_pos);
    }
  }

  for (decltype(pos2axis.size()) i{0}; i < pos2axis.size(); i++) {
    if (pos2axis[i] == -1 || pos2axis[i] >= td->nDims()) {
      pos2axis[i] = *(old_pos_left.begin());
      old_pos_left.erase(old_pos_left.begin());
    }
  }

  pos2axis.erase(pos2axis.begin() + td->nDims(), pos2axis.end());

  bool nullopt = true;
  std::unordered_map<int, int> axis2pos;
  for(decltype(pos2axis.size()) i{0}; i<pos2axis.size(); i++){
    int old_pos = i;
    int new_pos = pos2axis[i];
    if(old_pos != new_pos)
      nullopt = false;
    axis2pos[old_pos] = new_pos;
  }
  if(nullopt)
    return td;

  return td->reorder(axis2pos);

}

  std::vector<int> axis_map;
  Replay(std::vector<int> _axis_map):axis_map(_axis_map){}

public:
 // Replays history provided on td, axis_map is the mapping from td axes to
 // those expected in history, if an axis shouldn't be transformed, it needs to
 // be marked as -1 in the axis_map
 static TensorDomain* replay(TensorDomain* td, std::vector<Expr*> history, std::vector<int> axis_map) {
   Replay r(axis_map);
   return r.runReplay(td, history);
 }

}; //struct Replay

} // namespace

std::vector<bool> TransformReplay::getRootInfluence(TensorDomain* td, std::vector<bool> td_influence){
  return Influence::compute(td, td_influence);
}

TensorDomain* TransformReplay::replay(TensorDomain* td, std::vector<Expr*> history, std::vector<int> axis_map) {
  return Replay::replay(td, history, axis_map);
}

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
  root_influence_vector = getRootInfluence(replay_ref, root_influence_vector);

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
  TensorDomain* replayed = replay(replay_target, TransformIter::getHistory(replay_ref), axis_map);
  
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
