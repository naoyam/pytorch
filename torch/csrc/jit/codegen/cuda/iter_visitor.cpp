#include <torch/csrc/jit/codegen/cuda/iter_visitor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/type.h>

namespace torch {
namespace jit {
namespace fuser {

std::vector<Statement*> IterVisitor::next(Statement* statement) {
  if (statement->isVal())
    return next(static_cast<Val*>(statement));
  else if (statement->isExpr())
    return next(static_cast<Expr*>(statement));
  else
    TORCH_INTERNAL_ASSERT(
        false, "IterVisitor could not detect type in next_dispatch.");
}

std::vector<Statement*> IterVisitor::next(Val* v) {
  if (FusionGuard::getCurFusion()->origin(v) != nullptr)
    return {FusionGuard::getCurFusion()->origin(v)};
  return {};
}

std::vector<Statement*> IterVisitor::next(Expr* expr) {
  FusionGuard::getCurFusion()->assertInFusion(expr, "Cannot traverse expr, ");
  return {expr->inputs().begin(), expr->inputs().end()};
}

// Remove any stmt in stmts that is in visited
namespace{
  void remove_visited(std::vector<Statement*>& stmts, const std::unordered_set<Statement*>& visited){
    std::stack<std::vector<Statement*>::iterator> to_erase;
    for(auto it = stmts.begin(); it != stmts.end(); it++){
      if(visited.find(*it) != visited.end() )
        to_erase.push(it);
    }

    while(!to_erase.empty()){
      stmts.erase(to_erase.top());
      to_erase.pop();
    }

  }
}

void IterVisitor::traverseFrom(
    Fusion* const fusion,
    const std::vector<Val*>& from,
    bool traverseAllPaths) {
  FusionGuard fg(fusion);
  std::unordered_set<Statement*> visited;
  stmt_stack.clear();
  stmt_stack.push_back(std::deque<Statement*>(from.rbegin(), from.rend()));

  while (!stmt_stack.empty()) {
    auto next_stmts = next(stmt_stack.back().back());
    
    // Remove statements we already visited if we're not traversing all paths
    if(!traverseAllPaths)
      remove_visited(next_stmts, visited);
    
    // Traverse down until we get to a leaf
    while (!next_stmts.empty()) {
      stmt_stack.push_back(std::deque<Statement*>(next_stmts.rbegin(), next_stmts.rend()));
      next_stmts = next(stmt_stack.back().back());
      // Remove statements we already visited if we're not traversing all paths
      if(!traverseAllPaths)
        remove_visited(next_stmts, visited);
    }

    // Traverse back up
    // Mark visited
    visited.emplace(stmt_stack.back().back());
    // Handle
    handle(stmt_stack.back().back());
    // Remove
    stmt_stack.back().pop_back();
    
    while (!stmt_stack.empty() && stmt_stack.back().empty()) {

      stmt_stack.pop_back();
      if (!stmt_stack.empty()) {
        // Mark visited
        visited.emplace(stmt_stack.back().back());
        // Handle
        handle(stmt_stack.back().back());
        // Remove
        stmt_stack.back().pop_back();
      }
    }

  }

}

void IterVisitor::traverse(
    Fusion* const fusion,
    bool from_outputs_only,
    bool breadth_first) {
  FusionGuard fg(fusion);
  if (breadth_first)
    TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");

  std::vector<Val*> outputs;
  if (from_outputs_only) {
    for (Val* out : fusion->outputs()) {
      outputs.push_back(out);
    }
    // Search for Vals with no uses (output edges)
  } else
    for (Val* val : fusion->vals()) {
      if (!fusion->used(val))
        outputs.push_back(val);
    }

  traverseFrom(fusion, outputs, false);
}

void IterVisitor::traverseAllPaths(
    Fusion* const fusion,
    bool from_outputs_only,
    bool breadth_first) {
  FusionGuard fg(fusion);
  if (breadth_first)
    TORCH_INTERNAL_ASSERT(false, "Not implemented yet.");

  std::vector<Val*> outputs;
  if (from_outputs_only) {
    for (Val* out : fusion->outputs()) {
      outputs.push_back(out);
    }
    // Search for Vals with no uses (output edges)
  } else
    for (Val* val : fusion->vals()) {
      if (!fusion->used(val))
        outputs.push_back(val);
    }

  traverseFrom(fusion, outputs, true);
}

void DependencyCheck::handle(Val* val) {
  if (val->sameAs(dependency_)){
    is_dependency = true;
    std::stack<Val*> deps;
    for(auto stack : stmt_stack){
      if(stack.back()->isVal())
        deps.push(static_cast<Val*>(stack.back()));
    }
    dep_chain = deps;
  }
}

bool DependencyCheck::check() {
  is_dependency = false;
  IterVisitor::traverseFrom(of_->fusion(), {of_}, false);
  return is_dependency;
}

std::stack<Val*> DependencyCheck::getDependencyChain(Val* dependency, Val* of) {
  DependencyCheck dp(dependency, of);
  dp.check();

  // Return the reversed stack, we start from output and go to the input,
  // including of, but not dependency
  std::stack<Val*> dep_copy = dp.dep_chain;
  std::stack<Val*> reversed_clean;

  while (!dep_copy.empty()) {
    Val* next = dep_copy.top();
    dep_copy.pop();
    reversed_clean.push(next);
  }
  return reversed_clean;
}

} // namespace fuser
} // namespace jit
} // namespace torch
