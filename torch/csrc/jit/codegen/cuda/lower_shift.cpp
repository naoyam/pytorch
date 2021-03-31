#include <torch/csrc/jit/codegen/cuda/index_compute.h>
#include <torch/csrc/jit/codegen/cuda/instrumentation.h>
#include <torch/csrc/jit/codegen/cuda/ir_iostream.h>
#include <torch/csrc/jit/codegen/cuda/ir_utils.h>
#include <torch/csrc/jit/codegen/cuda/kernel_expr_evaluator.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_builder.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir_printer.h>
#include <torch/csrc/jit/codegen/cuda/lower2device.h>
#include <torch/csrc/jit/codegen/cuda/lower_shift.h>
#include <torch/csrc/jit/codegen/cuda/lower_utils.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>

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

  kir::Bool* makeAndExpr(kir::Val* lhs, kir::Val* rhs) {
    if (lhs == nullptr) {
      return rhs->as<kir::Bool>();
    } else if (rhs == nullptr) {
      return lhs->as<kir::Bool>();
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

    // This is actually not the case. test3 failed.
#if 0
    // If not fusion input, no predicate should be needed since the
    // buffer is expanded.
    if (!producer_fuser_tv->isFusionInput()) {
      return;
    }
#endif

    auto fuser_expr = consumer->fuserTv()->definition();

    // Unless the producer is a fusion input, if the expression is a
    // shift, it should be expanded already, so no prediate is needed.
    if (fuser_expr->isA<ShiftOp>() && !producer_fuser_tv->isFusionInput()) {
      return;
    }

    TensorDomain* producer_td = producer->fuserTv()->domain();
    TensorDomain* consumer_td = consumer->fuserTv()->domain();

    const auto num_dims = producer->domain()->rootDomain().size();

    auto prod_inds = Index::getProducerRootPredIndices(
        producer->fuserTv(), consumer->fuserTv(), for_loops_);
    auto pred_contiguity =
        std::vector<bool>(consumer_td->getRootDomain().size(), true);
    auto consumer_inds = Index::getConsumerRootPredIndices(
        consumer, for_loops_, pred_contiguity).first;

    TORCH_INTERNAL_ASSERT(consumer_inds.size() == consumer->domain()->rootDomain().size());
    TORCH_INTERNAL_ASSERT(prod_inds.size() == num_dims);

    kir::Bool* shift_pred = nullptr;

    TORCH_INTERNAL_ASSERT(!producer_td->hasRFactor());
    const auto& producer_root = producer_td->getRootDomain();

    auto p2c = PairwiseRootDomainMap(producer_fuser_tv, consumer_fuser_tv)
                   .mapProducerToConsumer(producer_td, consumer_td);

    const HaloMap& halo_map = gpu_lower_->haloMap();


    bool has_any_halo = false;
    for (size_t i = 0; i < num_dims; ++i) {
      auto consumer_id = consumer_td->getRootDomain()[i];
      const auto consumer_halo_info = halo_map.getHalo(consumer_id);
      if (consumer_halo_info.hasHalo()) {
        has_any_halo = true;
        break;
      }
    }

    for (size_t i = 0; i < num_dims; ++i) {
      // std::cerr << "idx: " << i << std::endl;

      auto producer_id = producer_root[i];
      auto consumer_id_it = p2c.find(producer_id);
      // If no corresponding consmer id exists, there's nothing to
      // predicate.
      if (consumer_id_it == p2c.end()) {
        continue;
      }

      auto consumer_id = consumer_id_it->second;

#if 0
      const int offset =
          Index::getProducerHaloOffset(i, producer_td, consumer_td, fuser_expr);

      const auto producer_halo_info = halo_map.getHalo(producer_id);
#endif

      const auto consumer_halo_info = halo_map.getHalo(consumer_id);

      int shift_offset = 0;
      if (auto shift_expr = dynamic_cast<ShiftOp*>(fuser_expr)) {
        shift_offset = shift_expr->offset(i);
      }

#if 0
      if (offset < 0) {
        shift_pred = makeAndExpr(
            shift_pred,
            ir_builder_.geExpr(pred_inds[i],
                               ir_builder_.create<kir::Int>(-offset))->as<kir::Bool>());
      }
#else
      unsigned left_limit = consumer_halo_info.width(0);
      if (shift_offset > 0) {
        left_limit += (unsigned)shift_offset;
      }
      if (left_limit > 0) {
        shift_pred = makeAndExpr(
            shift_pred,
            ir_builder_.geExpr(consumer_inds[i],
                               ir_builder_.create<kir::Int>(left_limit)));
      }
#endif
#if 0
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
#else
      auto right_offset = consumer_inds[i];
      if (shift_offset < 0) {
        right_offset = makeAddExpr(right_offset, -shift_offset);
      }
      if (shift_offset < 0 || has_any_halo) {
        shift_pred = makeAndExpr(
            shift_pred,
            ir_builder_
            .ltExpr(
                right_offset,
                makeAddExpr(consumer->domain()->rootDomain()[i]->extent(),
                            consumer_halo_info.width(0))));
      }
#endif


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
  auto it = halo_map_.find(id);
  if (it == halo_map_.end()) {
    it = halo_map_.insert({id, HaloInfo()}).first;
  }
  return it->second;
}

HaloInfo HaloMap::getHalo(IterDomain* id) const {
  auto it = halo_map_.find(id);
  if (it == halo_map_.end()) {
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

  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    updateExtents(tv);
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

void HaloMap::updateExtents(TensorView* tv) {
  std::unordered_map<IterDomain*, HaloInfo> inherited_halo;

  auto gpu_lower = GpuLower::current();

  for (auto root_axis: tv->getRootDomain()) {
    auto& halo_info = findOrCreate(root_axis);
    auto halo_width = halo_info.width();
    if (halo_width == 0) {
      continue;
    }
    auto expanded_extent = add(root_axis->rawExtent(), new Int(halo_width));
    extent_map_.insert({root_axis, expanded_extent});
    kir_extent_map_.insert({gpu_lower->lowerValue(root_axis)->as<kir::IterDomain>(),
        gpu_lower->lowerValue(expanded_extent)});
    inherited_halo.insert({root_axis, halo_info});
  }

  auto exprs = ExprSort::getExprs(
      FusionGuard::getCurFusion(),
      std::vector<Val*>(tv->domain()->domain().begin(), tv->domain()->domain().end()));

  // Splitting merged overlapped IterDomains is not allowed
  std::unordered_set<IterDomain*> merged_shifted_ids;

  for (auto expr: exprs) {
    if (auto split = dynamic_cast<Split*>(expr)) {
      TORCH_INTERNAL_ASSERT(merged_shifted_ids.find(split->in()) == merged_shifted_ids.end(),
                            "Splitting IterDomain that is a merged domain of shifted domains is not allowed");
      auto in_id = split->in();
      if (extent_map_.find(in_id) == extent_map_.end()) {
        continue;
      }
      TORCH_INTERNAL_ASSERT(inherited_halo.find(in_id) != inherited_halo.end());
      const auto& halo_info = inherited_halo.at(in_id);
      // propagate to inner domain
      auto out_id = split->inner();
      auto expanded_extent = add(out_id->rawExtent(), new Int(halo_info.width()));
      extent_map_.insert({out_id, expanded_extent});
      kir_extent_map_.insert({gpu_lower->lowerValue(out_id)->as<kir::IterDomain>(),
          gpu_lower->lowerValue(expanded_extent)});
      inherited_halo.insert({out_id, halo_info});
    } else if (auto merge = dynamic_cast<Merge*>(expr)) {
      if (extent_map_.find(merge->inner()) != extent_map_.end() ||
          extent_map_.find(merge->outer()) != extent_map_.end()) {
        auto inner_extent = getExtent(merge->inner());
        if (inner_extent == nullptr) {
          inner_extent = merge->inner()->rawExtent();
        }
        auto outer_extent = getExtent(merge->outer());
        if (outer_extent == nullptr) {
          outer_extent = merge->outer()->rawExtent();
        }
        auto expanded_extent = mul(outer_extent, inner_extent);
        extent_map_.insert({merge->out(), expanded_extent});
        kir_extent_map_.insert({gpu_lower->lowerValue(merge->out())->as<kir::IterDomain>(),
            gpu_lower->lowerValue(expanded_extent)});

        std::cerr << "Merged extent: "
                  << kir::toString(gpu_lower->lowerValue(expanded_extent))
                  << std::endl;

      }
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported expr: ", expr);
    }
  }
}

Val* HaloMap::getExtent(IterDomain* id) const {
  auto it = extent_map_.find(id);
  if (it != extent_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

kir::Val* HaloMap::getExtent(kir::IterDomain* id) const {
  auto it = kir_extent_map_.find(id);
  if (it != kir_extent_map_.end()) {
    return it->second;
  } else {
    return nullptr;
  }
}

#if 0
void HaloMap::buildStartMap(Fusion* fusion) {
  std::cerr << "Building start map\n";

  auto exprs = fusion->exprs();
  for (auto it = exprs.begin(); it != exprs.end(); ++it) {
    auto expr = *it;
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    std::cerr << "HaloInfoMap expr: " << expr << std::endl;
    propagateStartInfo(expr);
  }

  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    updateExtents(tv);
  }

  std::cerr << "HaloMap built\n";
}
#endif

std::string HaloMap::toString() const {
  std::stringstream ss;

  ss << "HaloMap:\n";
  for (auto kv : halo_map_) {
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
