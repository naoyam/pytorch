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

} // namespace fuser
} // namespace jit
} // namespace torch
