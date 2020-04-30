#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {

/*
 * TransformIter iterates on the split/merge/reorder graph of TensorDomain
 *
 * Running backward will execute these Exprs in reverse order. If you record
 * these events (generate_record=true) you can then replay them on another
 * tensor domain.
 */
struct TORCH_CUDA_API TransformIter : public IterVisitor {
 protected:
  virtual TensorDomain* replayBackward(Split*, TensorDomain*);
  virtual TensorDomain* replayBackward(Merge*, TensorDomain*);
  virtual TensorDomain* replayBackward(Reorder*, TensorDomain*);

  // dispatch
  TensorDomain* replayBackward(Expr*, TensorDomain*);

  // Iterates td's history starting with td, then origin(td), origin(origin(td))
  // etc. Returns root TensorDomain once it iterates through history. If
  // generate_record=true It will record the history of td in record. Record is
  // order operations root->td.
  virtual TensorDomain* runBackward(TensorDomain*);

  virtual TensorDomain* replay(Split*, TensorDomain*);
  virtual TensorDomain* replay(Merge*, TensorDomain*);
  virtual TensorDomain* replay(Reorder*, TensorDomain*);

  // dispatch
  virtual TensorDomain* replay(Expr*, TensorDomain*);

  // Runs through operations in history and applies them to TD, runs exprs from begining to end
  virtual TensorDomain* runReplay(TensorDomain*, std::vector<Expr*>);

 public:

  // Returns transformation exprs in reverse order (as seen processing
  // backwards)
  static std::vector<Expr*> getHistory(TensorDomain*);
  
  static TensorDomain* getRoot(TensorDomain* td) {
    TransformIter ti;
    return ti.runBackward(td);
  }
};

} // namespace fuser
} // namespace jit
} // namespace torch
