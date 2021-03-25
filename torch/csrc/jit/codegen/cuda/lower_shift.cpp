#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class ShiftPredicateInserter : public kir::MutableIrVisitor {
  void handle(kir::Expr* expr) {
    for (auto output : expr->outputs()) {
      auto out_tv = dynamic_cast<kir::TensorView*>(output);
      if (out_tv == nullptr) {
        continue;
      }
      for (auto input : expr->inputs()) {
        auto in_tv = dynamic_cast<kir::TensorView*>(input);
        if (in_tv == nullptr) {
          continue;
        }

        insertProducerPredicate(expr, in_tv, out_tv);
      }
    }

    expr->accept(this);
    return;
  }

  kir::Bool* makeAndExpr(kir::Bool* lhs, kir::Bool* rhs) {
    if (lhs == nullptr) {
      return rhs;
    } else if (rhs == nullptr) {
      return lhs;
    } else {
      kir::IrBuilder ir_builder(GpuLower::current()->kernel());
      return ir_builder.andExpr(lhs, rhs)->as<kir::Bool>();
    }
  }

  kir::Val* makeAddExpr(kir::Val* lhs, int rhs) {
    if (rhs == 0) {
      return lhs;
    } else if (rhs > 0) {
      kir::IrBuilder ir_builder(GpuLower::current()->kernel());
      return ir_builder.addExpr(lhs, ir_builder.create<kir::Int>(rhs));
      return lhs;
    } else {
      kir::IrBuilder ir_builder(GpuLower::current()->kernel());
      return ir_builder.subExpr(lhs, ir_builder.create<kir::Int>(-rhs));
    }
  }

  void insertProducerPredicate(
      kir::Expr* definition,
      kir::TensorView* producer,
      kir::TensorView* consumer) {
    std::cerr << "InsertProducerPred: "
              << "prod: TV" << producer->name() << ", cons: TV"
              << consumer->name() << std::endl;

    TensorView* producer_fuser_tv = producer->fuserTv();
    TensorView* consumer_fuser_tv = consumer->fuserTv();

    // If not fusion input, no predicate should be needed since the
    // buffer is expanded.
    if (!producer_fuser_tv->isFusionInput()) {
      return;
    }

    TensorDomain* producer_td = producer->fuserTv()->domain();
    TensorDomain* consumer_td = consumer->fuserTv()->domain();

    const auto num_dims = producer->domain()->rootDomain().size();

    auto pred_inds = Index::getProducerRootPredIndices(
        producer->fuserTv(), consumer->fuserTv(), for_loops_);

    TORCH_INTERNAL_ASSERT(pred_inds.size() == num_dims);

    auto fuser_expr = consumer->fuserTv()->definition();

    kir::Bool* shift_pred = nullptr;

    TORCH_INTERNAL_ASSERT(!producer_td->hasRFactor());
    const auto& producer_root = producer_td->getRootDomain();

    auto p2c = PairwiseRootDomainMap(producer_fuser_tv, consumer_fuser_tv)
                   .mapProducerToConsumer(producer_td, consumer_td);

    const HaloMap& halo_map = gpu_lower_->haloMap();

    for (size_t i = 0; i < num_dims; ++i) {
      std::cerr << "idx: " << i << std::endl;

      auto producer_id = producer_root[i];
      auto consumer_id_it = p2c.find(producer_id);
      // If no corresponding consmer id exits, there's nothing to
      // predicate.
      if (consumer_id_it == p2c.end()) {
        continue;
      }

      auto consumer_id = consumer_id_it->second;

      const int offset =
          Index::getProducerHaloOffset(i, producer_td, consumer_td, fuser_expr);

      const auto producer_halo_info = halo_map.get(producer_id);
      const auto consumer_halo_info = halo_map.get(consumer_id);

      int shift_offset = 0;
      if (auto shift_expr = dynamic_cast<ShiftOp*>(fuser_expr)) {
        shift_offset = shift_expr->offset(i);
      }

      if (offset < 0) {
        shift_pred = makeAndExpr(
            shift_pred,
            ir_builder_.geExpr(pred_inds[i],
                               ir_builder_.create<kir::Int>(-offset))->as<kir::Bool>());
      }

      if (producer_halo_info.width(1) < consumer_halo_info.width(1) ||
          shift_offset < 0) {
        shift_pred = makeAndExpr(
            shift_pred,
            ir_builder_
                .ltExpr(
                    makeAddExpr(pred_inds[i], offset),
                    producer->domain()->rootDomain()[i]->extent())
                ->as<kir::Bool>());
      }
    }

    if (shift_pred == nullptr) {
      return;
    }

    std::cerr << "Shift pred:" << kir::toString(shift_pred) << std::endl;
    auto shift_ite = ir_builder_.create<kir::IfThenElse>(shift_pred);

    auto& scope = for_loops_.back()->body();

    // Insert the if statement
    scope.insert_before(definition, shift_ite);

    // Remove the expr from the list
    scope.erase(definition);

    // Place the expr inside the if statement
    shift_ite->thenBody().push_back(definition);

    // Pads by zero for the ouf-of-bound accesses
    auto pad_expr = ir_builder_.create<kir::UnaryOp>(
        UnaryOpType::Set, consumer, ir_builder_.create<kir::Int>(0));
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
  // No need to set halo for input tensors
  if (producer->definition() == nullptr) {
    return;
  }

  auto c2p = PairwiseRootDomainMap(producer, consumer)
                 .mapConsumerToProducer(consumer->domain(), producer->domain());

  const auto& c_root = consumer->getRootDomain();

  //auto producer_alloc_point = loop_utils::getAllocPoint(producer);

  for (size_t i = 0; i < c_root.size(); ++i) {
    auto c_id = c_root[i];
    auto it = c2p.find(c_id);
    if (it == c2p.end()) {
      // nothing to propagate
      continue;
    }

    auto p_id = it->second;

#if 0
    if (std::find(
            producer->domain()->domain().begin() + producer_alloc_point,
            producer->domain()->domain().end(),
            p_id) != producer->domain()->domain().end()) {
      continue;
    }
#endif
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

  return ShiftPredicateInserter::insert(exprs);
}

std::vector<kir::ForLoop*> removeHaloLoops(
    const std::vector<kir::ForLoop*>& loops) {
  std::vector<kir::ForLoop*> out;
  const auto& halo_iter_map = GpuLower::current()->haloIterMap();
  for (auto fl : loops) {
    if (halo_iter_map.find(fl->iter_domain()) == halo_iter_map.end()) {
      out.push_back(fl);
    }
  }
  return out;
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
