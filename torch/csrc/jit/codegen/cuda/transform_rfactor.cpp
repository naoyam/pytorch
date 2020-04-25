#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

namespace torch {
namespace jit {
namespace fuser {

/*
 * Functions to backward propagate influence from split/merge/reorder
 */
// axis_map needs to be updated
// should_modify needs to be updated
// axis_map goes from the transform position to the position in our modified td.
TensorDomain* TransformRFactor::tdReplayBackward(
    Split* split,
    TensorDomain* td) {
  int saxis = split->axis();

  TORCH_INTERNAL_ASSERT(
      saxis >= 0 && saxis < axis_map.size(),
      "TransformRFactor tried to modify an axis out of range, recieved ",
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
      if (split->in()->axis(saxis)->isReduction() !=
          td->axis(axis)->isReduction()) {
        new_domain.push_back(new IterDomain(
            orig_axis->start(),
            orig_axis->extent(),
            orig_axis->parallel_method(),
            td->axis(axis)->isReduction()));
      } else {
        // preserve orig axis if it matches
        new_domain.push_back(orig_axis);
      }
    } else if (i != axis_p_1) {
      // Add in all other axes, these may not match the input td to the split.
      new_domain.push_back(td->axis(i));
    }
  }

  TensorDomain* replayed_inp = new TensorDomain(new_domain);
  Split* replayed_split = new Split(td, replayed_inp, axis, split->factor());
  return replayed_inp;
}

TensorDomain* TransformRFactor::tdReplayBackward(
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
      "TransformRFactor tried to modify an axis out of range, recieved ",
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
      // Insert pre-split axis
      new_domain.push_back(merge->in()->axis(maxis));
      new_domain.push_back(merge->in()->axis(maxis + 1));
    } else {
      // Add in all other axes, these may not match the input td to the split.
      new_domain.push_back(td->axis(i));
    }
  }

  TensorDomain* replayed_inp = new TensorDomain(new_domain);
  Merge* replayed_merge = new Merge(td, replayed_inp, axis);
  return replayed_inp;
}

TensorDomain* TransformRFactor::tdReplayBackward(
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

TensorDomain* TransformRFactor::tdReplayBackward(Expr* expr, TensorDomain* td) {
  TORCH_INTERNAL_ASSERT(
      expr->isExpr(),
      "Dispatch in transform iteration is expecting Exprs only.");
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      return tdReplayBackward(static_cast<Split*>(expr), td);
    case (ExprType::Merge):
      return tdReplayBackward(static_cast<Merge*>(expr), td);
    case (ExprType::Reorder):
      return tdReplayBackward(static_cast<Reorder*>(expr), td);
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Could not detect expr type in tdReplayBackward.");
  }
}

// Entry for backward influence propagation on td following record, history
// should be present -> past as you go through the vector
void TransformRFactor::replayBackward(
    TensorDomain* td,
    std::vector<Expr*> history) {
  TensorDomain* running_td = td;

  for (Expr* op : history)
    running_td = tdReplayBackward(op, running_td);
}

TensorDomain* TransformRFactor::runReplay(
    TensorDomain* in_td,
    std::vector<int> axes) {
  int ndims = (int)in_td->nDims();

  // Adjust and check provided axes
  std::transform(axes.begin(), axes.end(), axes.begin(), [ndims](int i) {
    TORCH_CHECK(
        i >= -ndims && i < ndims,
        "Rfactor replay recieved an axis outside the number of dims in the tensor, acceptable inclusive range is ",
        -ndims,
        " to ",
        ndims - 1);
    return i < 0 ? i + ndims : i;
  });

  // remove duplicates, and put into a set for searching
  std::set<int> axes_set(axes.begin(), axes.end());

  // Make a copy of in_td as we're going to change its history:
  std::vector<IterDomain*> domain_copy;
  for (int i{0}; i < ndims; i++) {
    if (axes_set.find(i) != axes_set.end()) {
      IterDomain* orig_axis = in_td->axis(i);
      TORCH_CHECK(
          orig_axis->isReduction(),
          "Tried to rFactor an axis that is not a reduction.");
      domain_copy.push_back(new IterDomain(
          orig_axis->start(),
          orig_axis->extent(),
          orig_axis->parallel_method(),
          false));
    } else {
      domain_copy.push_back(in_td->axis(i));
    }
  }

  // TD that we will actually modify
  TensorDomain* td = new TensorDomain(domain_copy);

  TransformRFactor trf;
  trf.axis_map.resize(ndims);

  for (decltype(ndims) i{0}; i < ndims; i++) {
    trf.axis_map[i] = axes_set.find(i) == axes_set.end() ? i : -1;
  }

  trf.replayBackward(td, TransformIter::getHistory(in_td));

  return td;
}

TensorDomain* TransformRFactor::runReplay2(
    TensorDomain* in_td,
    std::vector<int> axes) {
  int ndims = (int)in_td->nDims();

  // Adjust and check provided axes
  std::transform(axes.begin(), axes.end(), axes.begin(), [ndims](int i) {
    TORCH_CHECK(
        i >= -ndims && i < ndims,
        "Rfactor replay recieved an axis outside the number of dims in the tensor, acceptable inclusive range is ",
        -ndims,
        " to ",
        ndims - 1);
    return i < 0 ? i + ndims : i;
  });

  // remove duplicates, and put into a set for searching
  std::set<int> axes_set(axes.begin(), axes.end());

  TransformRFactor trf;
  trf.axis_map.resize(ndims);

  for (decltype(ndims) i{0}; i < ndims; i++) {
    trf.axis_map[i] = axes_set.find(i) == axes_set.end() ? i : -1;
  }

  // Make a copy of in_td as we're going to change its history:
  std::vector<IterDomain*> domain_copy;
  int it = 0;
  for (int i{0}; i < ndims; i++) {
    IterDomain* orig_axis = in_td->axis(i);
    if (axes_set.find(i) != axes_set.end()) {
      domain_copy.push_back(orig_axis);
      trf.axis_map[i] = -1;
      it++;
    } else if (!orig_axis->isReduction()) {
      domain_copy.push_back(orig_axis);
      trf.axis_map[i] = it++;
    } else {
      trf.axis_map[i] = -1;
    }
  }
  // TD that we will actually modify
  TensorDomain* td = new TensorDomain(domain_copy);

  trf.replayBackward(td, TransformIter::getHistory(in_td));

  return td;
}

} // namespace fuser
} // namespace jit
} // namespace torch
