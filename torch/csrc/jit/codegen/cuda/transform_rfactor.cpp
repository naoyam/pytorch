#include <torch/csrc/jit/codegen/cuda/transform_rfactor.h>

#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>

namespace torch {
namespace jit {
namespace fuser {

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
  bool found_rfactor = false;
  std::vector<IterDomain*> domain_copy;
  for (int i{0}; i < ndims; i++) {
    IterDomain* orig_axis = in_td->axis(i);
    if (axes_set.find(i) != axes_set.end())
      TORCH_CHECK(
          orig_axis->isReduction(),
          "Tried to rFactor an axis that is not a reduction.");

    if (orig_axis->isReduction()) {
      if (axes_set.find(i) == axes_set.end()) {
        domain_copy.push_back(new IterDomain(
            orig_axis->start(),
            orig_axis->extent(),
            orig_axis->parallel_method(),
            false,
            true));
        found_rfactor = true;
      } else {
        domain_copy.push_back(new IterDomain(
            orig_axis->start(),
            orig_axis->extent(),
            orig_axis->parallel_method(),
            true,
            true));
      }
    } else {
      domain_copy.push_back(in_td->axis(i));
    }
  }
  TORCH_CHECK(found_rfactor, "Could not find axis to rfactor out.");

  // TD that we will actually modify
  TensorDomain* td = new TensorDomain(domain_copy);

  std::vector<int> axis_map(ndims);
  std::iota(axis_map.begin(), axis_map.end(), 0);

  TransformIter::replayBackward(td, TransformIter::getHistory(in_td), axis_map);

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
  std::vector<int> axis_map(ndims);

  // Make a copy of in_td as we're going to change its history:
  std::vector<IterDomain*> domain_copy;
  bool found_rfactor = false;
  int it = 0;
  for (int i{0}; i < ndims; i++) {
    IterDomain* orig_axis = in_td->axis(i);
    if(axes_set.find(i) != axes_set.end())
      TORCH_CHECK(
          orig_axis->isReduction(),
          "Tried to rFactor an axis that is not a reduction.");
    
    if (orig_axis->isReduction() && axes_set.find(i) == axes_set.end()) {
      domain_copy.push_back(orig_axis);
      axis_map[i] = -1;
      it++;
      found_rfactor = true;
    } else if (!orig_axis->isReduction()) {
      domain_copy.push_back(orig_axis);
      axis_map[i] = it++;
    } else {
      axis_map[i] = -1;
    }
  }

  TORCH_CHECK(found_rfactor, "Could not find axis to rfactor out.");

  // TD that we will actually modify
  TensorDomain* td = new TensorDomain(domain_copy);

  TransformIter::replayBackward(td, TransformIter::getHistory(in_td), axis_map);

  return td;
}

} // namespace fuser
} // namespace jit
} // namespace torch
