#include <torch/csrc/jit/codegen/cuda/transform_iter.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>

namespace torch {
namespace jit {
namespace fuser {

void TransformIter::replayBackward(Split* expr) {}

void TransformIter::replayBackward(Merge* expr) {}

void TransformIter::replayBackward(Reorder* expr) {}

void TransformIter::replayBackward(Expr* expr) {
  TORCH_INTERNAL_ASSERT(
      expr->isExpr(),
      "Dispatch in transform iteration is expecting Exprs only.");
  switch (*(expr->getExprType())) {
    case (ExprType::Split):
      replayBackward(static_cast<Split*>(expr));
      break;
    case (ExprType::Merge):
      replayBackward(static_cast<Merge*>(expr));
      break;
    case (ExprType::Reorder):
      replayBackward(static_cast<Reorder*>(expr));
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Could not detect expr type in replayBackward.");
  }
}

std::vector<Expr*> TransformIter::getHistory(TensorDomain* td){

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
  return ops;
}

TensorDomain* TransformIter::runBackward(
    TensorDomain* td,
    bool generate_record) {

  std::vector<Expr*> ops = getHistory(td);

  if (generate_record)
    record = std::vector<Expr*>(ops.rbegin(), ops.rend());

  Fusion* fusion = FusionGuard::getCurFusion();

  for(Expr* op : ops)
    replayBackward(op);

  if (ops.size() > 0) {
    for (auto inp : ops[ops.size() - 1]->inputs())
      if (inp->getValType() &&
          inp->getValType().value() == ValType::TensorDomain)
        return static_cast<TensorDomain*>(inp);
  } else {
    return td;
  }

  TORCH_INTERNAL_ASSERT(false, "Issue found when running tensor transform backwards. Could not find a root domain.");
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

TensorDomain* TransformIter::runReplay(TensorDomain* td) {
  for (auto it = record.begin(); it < record.end(); ++it) {
    td = TransformIter::replay(*it, td);
  }
  return td;
}

} // namespace fuser
} // namespace jit
} // namespace torch
