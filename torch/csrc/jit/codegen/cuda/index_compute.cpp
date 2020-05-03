#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>

namespace torch {
namespace jit {
namespace fuser {

TensorDomain* IndexCompute::replayBackward(Split* split, TensorDomain* td) {
  int ax = split->axis();
  TORCH_INTERNAL_ASSERT(
      ax >= 0 && ax + 1 < indices.size(),
      "Hit an invalid Split transformation during IndexCompute, axis is not within bounds.");
  indices[ax] = add(mul(indices[ax], split->factor()), indices[ax + 1]);
  indices.erase(indices.begin() + ax + 1);
  return split->in();
}

TensorDomain* IndexCompute::replayBackward(Merge* merge, TensorDomain* td) {
  int ax = merge->axis();
  TORCH_INTERNAL_ASSERT(
      ax >= 0 && ax < indices.size(),
      "Hit an invalid MERGE transformation during IndexCompute, axis is not within bounds.");

  Val* I = merge->in()->axis(ax + 1)->extent();
  Val* ind = indices[ax];
  indices[ax] = div(ind, I);
  indices.insert(indices.begin() + ax + 1, mod(ind, I));
  return merge->in();
}

TensorDomain* IndexCompute::replayBackward(Reorder* reorder, TensorDomain* td) {
  // new2old[new_pos] = old_pos Generate new old2new map
  const std::vector<int>& new2old = reorder->new2old();

  std::vector<Val*> reordered_indices;

  // Reverse the map so we can simply push back into reordered_indices
  // old2new[old_pos] = new_pos
  std::vector<int> old2new(new2old.size(), -1);

  for (decltype(new2old.size()) i = 0; i < new2old.size(); i++) {
    int new_pos = i;
    int old_pos = new2old[i];
    TORCH_INTERNAL_ASSERT(
        new_pos >= 0 && new_pos < indices.size() && old_pos >= 0 &&
            old_pos < indices.size(),
        "Hit an invalid reorder transformation during IndexCompute,"
        " at least one move position is not within bounds.");
    old2new[old_pos] = new_pos;
  }
  for (decltype(old2new.size()) i = 0; i < old2new.size(); i++) {
    int new_pos = old2new[i];
    int old_pos = i;
    // reordered_indices[old_pos] = indices[new_pos];
    reordered_indices.push_back(indices[new_pos]);
  }

  indices = reordered_indices;
  return reorder->in();
}

IndexCompute::IndexCompute(const TensorView* tv, std::vector<Val*> _indices) {
  indices = std::move(_indices);

  TensorDomain* td = tv->domain();

  bool exclude_reduction = td->nDims() > indices.size();

  TORCH_CHECK(
      exclude_reduction || td->nDims() == indices.size(),
      "For IndexCompute the number of axis should match the number of dimensions"
      " in the TensorView.");

  // If we need to ignore the reduction dimensions because a tensor is
  // being consumed, not produced, then insert dummy dimensions in the
  // indices for bookkeeping while replaying split/merge/reorder operations.
  if (exclude_reduction)
    for (decltype(td->nDims()) i{0}; i < td->nDims(); i++)
      if (td->axis(i)->isReduction())
        indices.insert(indices.begin() + i, new Int(-1));

  // Run the split/merge/reorder operations backwards. This will
  // Modify std::vector<Int*> indices so it can be used to index
  // the root TensorDomain which should now match the physical axes.
  TensorDomain* root = TransformIter::runBackward(td);

  TORCH_INTERNAL_ASSERT(
      root->nDims() == indices.size(),
      "Error during IndexCompute. The number of indices generated"
      " after running the transformations backwards should match"
      " the number of dimensions of the root TensorView.");

  // Remove indices associated with reduction axes, we had them just for
  // bookkeeping.
  if (exclude_reduction) {
    for (auto i = root->nDims() - 1; i >= 0; i--)
      if (root->axis(i)->isReduction())
        indices.erase(indices.begin() + i);
  }
}

std::vector<Val*> IndexCompute::computeIndices(
    const TensorView* tv,
    std::vector<Val*> _indices) {
  IndexCompute ic(tv, std::move(_indices));
  return ic.indices;
}

} // namespace fuser
} // namespace jit
} // namespace torch
