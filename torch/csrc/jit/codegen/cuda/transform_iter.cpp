#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

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
  for (Expr* op : history) {
    td = TransformIter::replay(op, td);
  }
  return td;
}


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

std::vector<bool> TransformIter::getRootInfluence(TensorDomain* td, std::vector<bool> td_influence){
  return Influence::compute(td, td_influence);
}

TensorDomain* TransformIter::replay(TensorDomain* td, std::vector<Expr*> history, std::vector<int> axis_map) {
  return Replay::replay(td, history, axis_map);
}


} // namespace fuser
} // namespace jit
} // namespace torch
