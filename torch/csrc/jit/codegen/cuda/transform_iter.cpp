#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

namespace torch {
namespace jit {
namespace fuser {

TensorDomain* TransformIter::replayBackward(Split* split, TensorDomain* td) {
  return split->in();
}

TensorDomain* TransformIter::replayBackward(Merge* merge, TensorDomain* td) {
  return merge->in();
}

TensorDomain* TransformIter::replayBackward(Reorder* reorder, TensorDomain* td) {
  return reorder->in();
}

TensorDomain* TransformIter::replayBackward(Expr* expr, TensorDomain* td) {
  TORCH_INTERNAL_ASSERT(
      expr->isExpr(),
      "Dispatch in transform iteration is expecting Exprs only.");
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      return replayBackward(static_cast<Split*>(expr), td);
    case (ExprType::Merge):
      return replayBackward(static_cast<Merge*>(expr), td);
    case (ExprType::Reorder):
      return replayBackward(static_cast<Reorder*>(expr), td);
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Could not detect expr type in replayBackward.");
  }
}

std::vector<Expr*> TransformIter::getHistory(TensorDomain* td) {
  std::vector<Expr*> ops;
  TensorDomain* root = td; // backward running td
  Fusion* fusion = FusionGuard::getCurFusion();

  // Get my origin
  Expr* orig = fusion->origin(root);
  std::set<Expr*> visited_exprs;

  // If I'm not back to the original td
  while (orig != nullptr) {
    if (visited_exprs.find(orig) != visited_exprs.end())
      TORCH_INTERNAL_ASSERT(
          false,
          "TransformReplay::runBackward is not traversing a correct history.");
    ops.push_back(orig);
    visited_exprs.emplace(orig);
    TensorDomain* previous_td = nullptr;
    // Check inputs of this operation, make sure there isn't more than one TD
    // I can only record operations that only take this TD as an input.
    for (Val* inp : orig->inputs())
      if (inp->getValType() == ValType::TensorDomain) {
        if (previous_td != nullptr)
          TORCH_INTERNAL_ASSERT(
              false,
              "TransformReplay::runBackward could not decifer transform history of a TensorDomain.");

        // Traverse back
        root = static_cast<TensorDomain*>(inp);
        orig = fusion->origin(root);
      }
  }
  return std::vector<Expr*>(ops.rbegin(), ops.rend());
}

TensorDomain* TransformIter::runBackward(TensorDomain* td) {

  std::vector<Expr*> ops = getHistory(td);

  // We want to iterate backwards, reverse history.
  ops = std::vector<Expr*>(ops.rbegin(), ops.rend());

  Fusion* fusion = FusionGuard::getCurFusion();

  TensorDomain* running_td = td;
  for (Expr* op : ops)
    running_td = replayBackward(op, running_td);

  return running_td;

}

TensorDomain* TransformIter::replay(Split* expr, TensorDomain* td) {
  return td->split(
      expr->axis(), static_cast<Int*>(expr->factor())->value().value());
}

TensorDomain* TransformIter::replay(Merge* expr, TensorDomain* td) {
  return td->merge(expr->axis());
}

TensorDomain* TransformIter::replay(Reorder* expr, TensorDomain* td) {
  std::unordered_map<int, int> axis2pos;
  for (decltype(expr->pos2axis().size()) i{0}; i < expr->pos2axis().size(); i++)
    axis2pos[expr->pos2axis()[i]] = i;
  return td->reorder(axis2pos);
}

TensorDomain* TransformIter::replay(Expr* expr, TensorDomain* td) {
  TORCH_INTERNAL_ASSERT(expr->isExpr());
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      return replay(static_cast<Split*>(expr), td);
    case (ExprType::Merge):
      return replay(static_cast<Merge*>(expr), td);
    case (ExprType::Reorder):
      return replay(static_cast<Reorder*>(expr), td);
    default:
      TORCH_INTERNAL_ASSERT(false, "Could not detect expr type in replay.");
  }
}

TensorDomain* TransformIter::runReplay(TensorDomain* td, std::vector<Expr*> history) {
  for (Expr* op : history)
    td = TransformIter::replay(op, td);
  return td;
}


namespace{

struct Influence  : public TransformIter {
private:

// BACKWARD INFLUENCE

TensorDomain* replayBackward(Split* split, TensorDomain* td) override {
  int axis = split->axis();
  TORCH_INTERNAL_ASSERT(
      axis + 1 < influence.size(),
      "Error during replay backwards, td/influence size mismatch.");
  influence[axis] = influence[axis] | influence[axis + 1];
  influence.erase(influence.begin() + axis + 1);
  return split->in();
}

TensorDomain* replayBackward(Merge* merge, TensorDomain* td) override {
  int axis = merge->axis();
  TORCH_INTERNAL_ASSERT(
      axis < influence.size(),
      "Error during replay backwards, td/influence size mismatch.");
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
        "Error during replay backwards, td/influence size mismatch.");
    reorder_influence[old_pos] = influence[new_pos];
  }

  influence = reorder_influence;
  return reorder->in();
}

// FORWARD INFLUENCE

TensorDomain* replay(Split* split, TensorDomain* td) {
  int axis = split->axis();
  TORCH_INTERNAL_ASSERT(
      axis < influence.size(),
      "Error during replay backwards, td/influence size mismatch.");
  influence.insert(influence.begin() + axis + 1, influence[axis]);
  return nullptr;
}

TensorDomain* replay(Merge* merge, TensorDomain* td) {
  int axis = merge->axis();
  TORCH_INTERNAL_ASSERT(
      axis > 0 && axis + 1 < influence.size(),
      "Error during replay backwards, td/influence size mismatch.");
  influence[axis] = influence[axis] | influence[axis + 1];
  influence.erase(influence.begin() + axis + 1);
  return nullptr;
}

TensorDomain* replay(Reorder* reorder, TensorDomain* td) {
  // pos2axis[new_pos] = old_pos Generate new axis2pos map
  const std::vector<int>& pos2axis = reorder->pos2axis();

  std::vector<bool> reorder_influence(influence.size(), false);
  for (decltype(pos2axis.size()) i = 0; i < pos2axis.size(); i++) {
    int new_pos = i;
    int old_pos = pos2axis[i];
    TORCH_INTERNAL_ASSERT(
        new_pos < influence.size() && old_pos < reorder_influence.size(),
        "Error during replay backwards, td/influence size mismatch.");
    reorder_influence[new_pos] = influence[old_pos];
  }

  influence = reorder_influence;
  return nullptr;
}

// INTERFACE

std::vector<bool> influence;

// BACKWARD INTERFACE
Influence(TensorDomain* td, std::vector<bool> td_influence):influence(td_influence){
  TransformIter::runBackward(td);
}

// FORWARD INTERFACE
Influence(std::vector<Expr*> history, std::vector<bool> td_influence):influence(td_influence){
  TransformIter::runReplay(nullptr, history);
}

public:

static std::vector<bool> computeBackward(TensorDomain* td, std::vector<bool> td_influence){
  TORCH_INTERNAL_ASSERT(
      td_influence.size() == td->nDims(),
      "Tried to compute backward influence, but recieved an influence vector that does not match the TensorDomain size.");
  Influence inf(td, td_influence);
  return inf.influence;
}

static std::vector<bool> computeForward(std::vector<Expr*> history, std::vector<bool> td_influence){
  if(history.empty())
    return td_influence;

  TORCH_INTERNAL_ASSERT(
      history[0]->input(0)->getValType().value() == ValType::TensorDomain &&  static_cast<TensorDomain*>(history[0]->input(0))->nDims() == td_influence.size(),
      "Tried to compute influence, but recieved an influence vector that does not match the expected size.");
  Influence inf(history, td_influence);
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

  // Axis relative to td
  int axis = axis_map[saxis];

  if (axis == -1) {
    // don't modify path, we need an extra axis as there would have been one
    // there, but we shouldn't modify it.
    axis_map.insert(axis_map.begin() + saxis + 1, -1);
    return td;
  }

  // Move indices up as we now have an extra axis
  std::transform(
      axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
        return i > axis ? i + 1 : i;
      });

  // Insert new axis in map
  axis_map.insert(axis_map.begin() + saxis + 1, axis + 1);

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


// This is, maybe surprisingly, one of the more difficult transforms to replay
TensorDomain* replay(Reorder* reorder, TensorDomain* td) {
  // pos2axis is new2old lets convert to old2new as it makes this easier to do,
  // and we need that map anyways in the end to replay reorder
  const std::vector<int>& new2old_orig = reorder->pos2axis();
  std::vector<int> old2new_orig(new2old_orig.size());
  for(decltype(new2old_orig.size()) i{0}; i<new2old_orig.size(); i++)
    old2new_orig[new2old_orig[i]] = i;

  // Convert map with our axis maping, ignore if old_pos == -1
  std::vector<int> old2new(old2new_orig.size(), -1);
  for(decltype(old2new.size()) i{0}; i<old2new.size(); i++){
    int old_pos = axis_map[i];
    int new_pos = old2new_orig[i];
    if(old_pos != -1)
      old2new[old_pos] = new_pos;
  }
  
  // Going to move all new_pos to the left so there's no gaps, compute offset
  std::vector<int> offset(old2new.size(), 0);
  for(decltype(offset.size()) i{0}; i<offset.size(); i++){
    offset[i] = old2new[i] == -1 ? -1 : 0;
  }
  
  // Prefix sum offset
  for(decltype(offset.size()) i{1}; i<offset.size(); i++){
    offset[i] = offset[i-1];
  }

  // move positions over
  for(decltype(old2new.size()) i{0}; i<old2new.size(); i++){
    if (old2new[i] == -1)
      continue;

    int new_pos = old2new[i] - offset[i];
    TORCH_INTERNAL_ASSERT(
        new_pos >= 0 && new_pos < old2new.size(),
        "Error in offset calculation in replay reorder.");
    old2new[i] = new_pos;
  }

  // We have missing entries in our map as old_pos could have been -1 which
  // means we ignored the reordering for them, fill these values in order in the
  // empty spots in the right side
  int max_new_pos = -1;
  for(decltype(old2new.size()) i{0}; i<old2new.size(); i++)
    max_new_pos = max_new_pos > old2new[i] ? max_new_pos : old2new[i];

  for(decltype(old2new.size()) i{0}; i<old2new.size(); i++)
    if(old2new[i] == -1)
      old2new[i] = ++max_new_pos;

  // Reorder the axis_map based on this reorder
  std::vector<int> reordered_axis_map(axis_map.size());
  for(decltype(reordered_axis_map.size())i{0}; i<reordered_axis_map.size(); i++)
    reordered_axis_map[old2new[i]] = axis_map[i];
  axis_map = reordered_axis_map;

  // Check if this is a null opt (no actual reordering done), create map, make
  // sure we don't have entries > td->nDims which we could have
  bool nullopt = true;
  std::unordered_map<int, int> old2new_map;
  for(decltype(td->nDims()) i{0}; i<td->nDims(); i++){
    if(old2new[i] != i){
      nullopt = false;
      break;
    }
      old2new_map[i] = old2new[i];
  }
  
  // If null opt do nothing, return td
  if(nullopt)
    return td;

  // Rerun reorder
  auto reordered = td->reorder(old2new_map);
  return reordered;

}

  std::vector<int> axis_map;
  Replay(std::vector<int> _axis_map):axis_map(_axis_map){}

public:
 // Replays history provided on td, axis_map is the mapping from td axes to
 // those expected in history, if an axis shouldn't be transformed, it needs to
 // be marked as -1 in the axis_map
 static TensorDomain* replay(TensorDomain* td, std::vector<Expr*> history, std::vector<int> axis_map) {
   if (history.empty())
     return td;

   TORCH_INTERNAL_ASSERT(
       history[0]->input(0)->getValType().value() == ValType::TensorDomain &&
           static_cast<TensorDomain*>(history[0]->input(0))->nDims() ==
               axis_map.size(),
       "Tried to replay transforms, but received an invalid axis_map.");

   for (auto ent : axis_map)
     TORCH_INTERNAL_ASSERT(
         ent >= -1 && ent < (int) td->nDims(),
         "Tried to replay transforms, but received an invalid axis_map.");

   Replay r(axis_map);
   return r.runReplay(td, history);
 }

}; //struct Replay

struct TORCH_CUDA_API TransformBackward : public TransformIter {
private:
 // axis_map goes from the transform position to the position in our modified td.
TensorDomain* replayBackward(
    Split* split,
    TensorDomain* td) {
  int saxis = split->axis();

  TORCH_INTERNAL_ASSERT(
      saxis >= 0 && saxis < axis_map.size(),
      "TransformBackward tried to modify an axis out of range, recieved ",
      saxis,
      " but this value should be >= 0 and < axis_map.size()");

  // Get axis relative to what we actually have in td.
  int axis = axis_map[saxis];
  int axis_p_1 = axis_map[saxis + 1];
  // If either dim is not to be touch, set both not to be touched
  axis = axis_p_1 == -1 ? -1 : axis;
  axis_map[saxis] = axis;

  // Remove axis from axis_map as in original transformations it didn't exist
  axis_map.erase(axis_map.begin() + saxis + 1);

  // Don't modify:
  if (axis == -1)
    return td;

  // Move indices down as previously we didn't have the split axis
  std::transform(
      axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
        return i > axis ? i - 1 : i;
      });

  // Create new domain reflecting pre-split
  std::vector<IterDomain*> new_domain;
  for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
    if (i == axis) {
      IterDomain* orig_axis = split->in()->axis(saxis);
      // Insert pre-split axis, make sure isReduction matches what is expected
      new_domain.push_back(new IterDomain(
        orig_axis->start(),
        orig_axis->extent(),
        orig_axis->parallel_method(),
        td->axis(axis)->isReduction(),
        td->axis(axis)->isRFactorProduct()));
    } else if (i != axis_p_1) {
      // Add in all other axes, these may not match the input td to the split.
      new_domain.push_back(td->axis(i));
    }
  }

  TensorDomain* replayed_inp = new TensorDomain(new_domain);
  Split* replayed_split = new Split(td, replayed_inp, axis, split->factor());
  return replayed_inp;
}

TensorDomain* replayBackward(
    Merge* merge,
    TensorDomain* td) {
  /*
   * Remember axis_map goes from merge information -> how it's stored in td
   * When we're done we want axis_map to match the returned td before or not
   * before the merge depending on should_modify.
   */

  int maxis = merge->axis();

  TORCH_INTERNAL_ASSERT(
      maxis >= 0 && maxis < axis_map.size(),
      "TransformBackward tried to modify an axis out of range, recieved ",
      maxis,
      " but this value should be >=0 and <",
      axis_map.size());

  if (axis_map[maxis] == -1) {
    // don't modify path, we need an extra axis as there was previously one
    // there, but we shouldn't modify it.
    axis_map.insert(axis_map.begin() + maxis + 1, -1);
    return td;
  }

  // Recreate the merge, axis is relative to the td
  int axis = axis_map[maxis];
  // Move indices up as previously we had an extra axis
  std::transform(
      axis_map.begin(), axis_map.end(), axis_map.begin(), [axis](int i) {
        return i > axis ? i + 1 : i;
      });

  // Insert pre-merged axis back into map
  axis_map.insert(axis_map.begin() + maxis + 1, axis_map[maxis] + 1);

  // Create new domain reflecting pre-split
  std::vector<IterDomain*> new_domain;
  for (decltype(td->nDims()) i{0}; i < td->nDims(); i++) {
    if (i == axis) {
      IterDomain* td_axis = td->axis(axis);
      IterDomain* maxis_1 = merge->in()->axis(maxis);
      IterDomain* maxis_2 = merge->in()->axis(maxis + 1);
      new_domain.push_back(new IterDomain(
          maxis_1->start(),
          maxis_1->extent(),
          ParallelType::Serial,
          td_axis->isReduction(),
          td_axis->isRFactorProduct()));
      new_domain.push_back(new IterDomain(
          maxis_2->start(),
          maxis_2->extent(),
          ParallelType::Serial,
          td_axis->isReduction(),
          td_axis->isRFactorProduct()));
    } else {
      // Add in all other axes, these may not match the input td to the split.
      new_domain.push_back(td->axis(i));
    }
  }

  TensorDomain* replayed_inp = new TensorDomain(new_domain);
  Merge* replayed_merge = new Merge(td, replayed_inp, axis);
  return replayed_inp;
}

TensorDomain* replayBackward(
    Reorder* reorder,
    TensorDomain* td) {
  const std::vector<int>& pos2axis_orig = reorder->pos2axis();

  // We want to convert pos2axis to something with td->nDims which it isn't
  // guarenteed to be
  std::vector<int> pos2axis(td->nDims(), -1);

  std::set<int> old_pos_left;
  for (decltype(axis_map.size()) i{0}; i < axis_map.size(); i++)
    old_pos_left.emplace(i);

  for (decltype(pos2axis_orig.size()) i{0}; i < pos2axis_orig.size(); i++) {
    int new_pos = axis_map[i]; // position in td
    int old_pos = pos2axis_orig[i]; // position it should be at before td

    if (new_pos != -1) {
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

  // pos2axis_orig[reorder->out()->pos] = reorder->in()->pos
  // axis_map[reorder->out()->pos] = td->pos
  // pos2axis[td->pos] = old_td->pos
  // NEED: new_axis_map[reorder->in()->pos] = old_td->pos

  std::vector<int> new_axis_map(axis_map.size(), -1);
  for (decltype(new_axis_map.size()) i{0}; i < new_axis_map.size(); i++) {
    int reorder_out_pos = i;
    int reorder_in_pos = pos2axis_orig[reorder_out_pos];
    int td_pos = axis_map[reorder_out_pos];
    int old_td_pos = td_pos == -1 ? -1 : pos2axis[td_pos];

    new_axis_map[reorder_in_pos] = old_td_pos;
  }

  axis_map = new_axis_map;

  std::vector<IterDomain*> old_td(td->nDims(), nullptr);
  for (decltype(pos2axis.size()) i{0}; i < pos2axis.size(); i++) {
    // pos2axis[new] = old relative to td
    int new_pos = i; // position in td
    int old_pos = pos2axis[i]; // position it should be at before td
    old_td[old_pos] = td->axis(new_pos);
  }

  TensorDomain* replayed_inp = new TensorDomain(old_td);
  Reorder* replayed_split = new Reorder(td, replayed_inp, pos2axis);
  return replayed_inp;
}

TensorDomain* replayBackward(Expr* expr, TensorDomain* td) {
  TORCH_INTERNAL_ASSERT(
      expr->isExpr(),
      "Dispatch in transform iteration is expecting Exprs only.");
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      return replayBackward(static_cast<Split*>(expr), td);
    case (ExprType::Merge):
      return replayBackward(static_cast<Merge*>(expr), td);
    case (ExprType::Reorder):
      return replayBackward(static_cast<Reorder*>(expr), td);
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Could not detect expr type in replayBackward.");
  }
}

// Entry for backward influence propagation on td following record, history
// should be present -> past as you go through the vector
TensorDomain* replayBackward(
    TensorDomain* td,
    std::vector<Expr*> history) {
  TensorDomain* running_td = td;
  history = std::vector<Expr*>(history.rbegin(), history.rend());
  for (Expr* op : history)
    running_td = replayBackward(op, running_td);
  return running_td;
}

std::vector<int> axis_map;

TransformBackward(std::vector<int> _axis_map) : axis_map(_axis_map){};

public:
  static TensorDomain* replay(TensorDomain* td, std::vector<Expr*> history, std::vector<int> axis_map) {
    TransformBackward tb(axis_map);
    return tb.replayBackward(td, history);
  }

};

} // namespace

std::vector<bool> TransformIter::getRootInfluence(TensorDomain* td, std::vector<bool> td_influence){
  return Influence::computeBackward(td, td_influence);
}

std::vector<bool> TransformIter::replayInfluence(std::vector<Expr*> history, std::vector<bool> td_influence){
  return Influence::computeForward(history, td_influence);
}

TensorDomain* TransformIter::replay(TensorDomain* td, std::vector<Expr*> history, std::vector<int> axis_map) {
  return Replay::replay(td, history, axis_map);
}

TensorDomain* TransformIter::replayBackward(TensorDomain* td, std::vector<Expr*> history, std::vector<int> axis_map) {
  return TransformBackward::replay(td, history, axis_map);
}

} // namespace fuser
} // namespace jit
} // namespace torch
