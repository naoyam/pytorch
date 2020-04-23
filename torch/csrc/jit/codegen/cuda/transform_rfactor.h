#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/transform_iter.h>

#include <algorithm>
#include <vector>

namespace torch {
namespace jit {
namespace fuser {

// TODO: Only replay dispatch is really borrowed from TransformIter, we should
// reevaluate the reuse of dispatch for classes that inherit TransformIter.
struct TORCH_CUDA_API TransformRFactor : public TransformIter {
 private:
  /*
   * Functions to backward propagate influence from split/merge/reorder
   *
   * We can't override TransformIter in this case as we're trying to produce a
   * new root tensor domain, so we want a return value.
   */
  TensorDomain* tdReplayBackward(Split*, TensorDomain*);
  TensorDomain* tdReplayBackward(Merge*, TensorDomain*);
  TensorDomain* tdReplayBackward(Reorder*, TensorDomain*);

  // Have to have our own dispatch because of ther return type
  TensorDomain* tdReplayBackward(Expr*, TensorDomain*);

  // Entry for backward influence propagation on td following record, this transformation is in place on td
  void replayBackward(TensorDomain* td, std::vector<Expr*> history);

  std::vector<int> axis_map;

public:
 // Create a copy of td, change its history by presrving axes so they appear in the root domain
 static TensorDomain* runReplay(TensorDomain*, std::vector<int> axes);
 static TensorDomain* runReplay2(TensorDomain*, std::vector<int> axes);
};

} // namespace fuser
} // namespace jit
} // namespace torch