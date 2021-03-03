#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class ShiftPredicateInserter : public kir::MutableIrVisitor {
  void handle(kir::Expr* expr) {
    auto set_expr = dynamic_cast<kir::UnaryOp*>(expr);
    if (set_expr == nullptr || set_expr->operation() != UnaryOpType::Set ||
        !expr->outputs()[0]->isA<kir::TensorView>()) {
      expr->accept(this);
      return;
    }

    auto out_tv = expr->outputs()[0]->as<kir::TensorView>();
    auto fuser_expr = dynamic_cast<ShiftOp*>(out_tv->fuserTv()->definition());
    if (fuser_expr == nullptr) {
      expr->accept(this);
      return;
    }

    const auto& root_domain = out_tv->domain()->rootDomain();
    auto num_dims = root_domain.size();
    auto pred_contiguity = std::vector<bool>(num_dims, false);

    auto pred_inds =
        Index::getConsumerRootPredIndices(out_tv, for_loops_, pred_contiguity)
            .first;

    TORCH_INTERNAL_ASSERT(pred_inds.size() == num_dims);

    kir::Bool* shift_pred_all = nullptr;

    for (size_t i = 0; i < num_dims; ++i) {
      int shift_offset = fuser_expr->offset(i);
      if (shift_offset == 0) {
        continue;
      }
      kir::Bool* pred = nullptr;
      if (shift_offset > 0) {
        auto offset_expr = ir_builder_.subExpr(
            pred_inds.at(i), ir_builder_.create<kir::Int>(shift_offset));
        pred = ir_builder_.geExpr(offset_expr, ir_builder_.create<kir::Int>(0))
                   ->as<kir::Bool>();
      } else {
        auto offset_expr = ir_builder_.addExpr(
            pred_inds.at(i), ir_builder_.create<kir::Int>(-shift_offset));
        pred = ir_builder_.ltExpr(offset_expr, root_domain[i]->extent())
                   ->as<kir::Bool>();
      }

      if (shift_pred_all) {
        shift_pred_all =
            ir_builder_.andExpr(shift_pred_all, pred)->as<kir::Bool>();
      } else {
        shift_pred_all = pred;
      }
    }

    if (shift_pred_all == nullptr) {
      return;
    }

    std::cerr << "Shift pred:" << kir::toString(shift_pred_all) << std::endl;
    auto shift_ite = ir_builder_.create<kir::IfThenElse>(shift_pred_all);

    auto& scope = for_loops_.back()->body();

    // Insert the if statement
    scope.insert_before(expr, shift_ite);

    // Remove the expr from the list
    scope.erase(expr);

    // Place the expr inside the if statement
    shift_ite->thenBody().push_back(expr);

    // Pads by zero for the ouf-of-bound accesses
    auto pad_expr = ir_builder_.create<kir::UnaryOp>(
        UnaryOpType::Set, out_tv, ir_builder_.create<kir::Int>(0));
    shift_ite->elseBody().push_back(pad_expr);
  }

  void visit(kir::ForLoop* fl) final {
    for_loops_.push_back(fl);
    // Modifying in place, make a copy of the vector
    const std::vector<kir::Expr*>& exprs = fl->body().exprs();
    for (auto expr : exprs) {
      handle(expr);
    }
    for_loops_.pop_back();
  }

  ShiftPredicateInserter(std::vector<kir::Expr*> loop_nests)
      : loop_nests_(std::move(loop_nests)),
        gpu_lower_(GpuLower::current()),
        ir_builder_(gpu_lower_->kernel()) {
    const std::vector<kir::Expr*> exprs = loop_nests_;
    for (auto expr : exprs) {
      handle(expr);
    }
  }

 private:
  std::vector<kir::Expr*> loop_nests_;
  GpuLower* gpu_lower_;
  kir::IrBuilder ir_builder_;
  std::vector<kir::ForLoop*> for_loops_;

 public:
  static std::vector<kir::Expr*> insert(
      const std::vector<kir::Expr*>& loop_nests) {
    ShiftPredicateInserter inserter(loop_nests);
    return inserter.loop_nests_;
  }
};

} // namespace

HaloInfo& HaloMap::findOrCreate(IterDomain* id) {
  auto it = map_.find(id);
  if (it == map_.end()) {
    it = map_.insert({id, HaloInfo()}).first;
  }
  return it->second;
}

HaloInfo HaloMap::get(IterDomain* id) const {
  auto it = map_.find(id);
  if (it == map_.end()) {
    return HaloInfo();
  } else {
    return it->second;
  }
}

void HaloMap::build() {
  std::cerr << "Building HaloMap\n";

  Fusion* fusion = FusionGuard::getCurFusion();
  TORCH_INTERNAL_ASSERT(fusion != nullptr);

  auto exprs = fusion->exprs();
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    auto expr = *it;
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    std::cerr << "HaloInfoMap expr: " << expr << std::endl;
    propagateHaloInfo(expr);
  }

  std::cerr << "HaloMap built\n";
}

void HaloMap::propagateHaloInfo(Expr* expr) {
  for (auto output : expr->outputs()) {
    auto out_tv = dynamic_cast<TensorView*>(output);
    if (out_tv == nullptr) {
      continue;
    }
    for (auto input : expr->inputs()) {
      auto in_tv = dynamic_cast<TensorView*>(input);
      if (in_tv == nullptr) {
        continue;
      }
      propagateHaloInfo(in_tv, out_tv, expr);
    }
  }
}

void HaloMap::propagateHaloInfo(
    TensorView* producer,
    TensorView* consumer,
    Expr* expr) {
  auto c2p = PairwiseRootDomainMap(producer, consumer)
                 .mapConsumerToProducer(consumer->domain(), producer->domain());

  const auto& c_root = consumer->getRootDomain();

  for (size_t i = 0; i < c_root.size(); ++i) {
    auto c_id = c_root[i];
    auto it = c2p.find(c_id);
    if (it == c2p.end()) {
      // nothing to propagate
      continue;
    }

    auto p_id = it->second;

    if (std::find(
            producer->domain()->domain().begin() +
                producer->getComputeAtPosition(),
            producer->domain()->domain().end(),
            p_id) != producer->domain()->domain().end()) {
      continue;
    }

    auto& c_info = findOrCreate(c_id);
    auto& p_info = findOrCreate(p_id);

    if (auto shift_op = dynamic_cast<ShiftOp*>(expr)) {
      const int offset = shift_op->offset(i);
      if (offset == 0) {
        p_info.merge(c_info);
      } else {
        int pos = (offset > 0) ? 0 : 1;
        p_info.merge(pos, c_info.width(pos) + std::abs(offset));
      }
    } else {
      p_info.merge(c_info);
    }
  }
}

std::string HaloMap::toString() const {
  std::stringstream ss;

  ss << "HaloMap:\n";
  for (auto kv : map_) {
    ss << "  " << kv.first << " -> (" << kv.second.toString() << ")\n";
  }

  return ss.str();
}

std::vector<kir::Expr*> insertShiftPredicates(
    const std::vector<kir::Expr*>& exprs) {
  FUSER_PERF_SCOPE("insertShiftPredicates");

  HaloMap halo_map;
  halo_map.build();
  std::cerr << halo_map.toString() << std::endl;

  return ShiftPredicateInserter::insert(exprs);
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
