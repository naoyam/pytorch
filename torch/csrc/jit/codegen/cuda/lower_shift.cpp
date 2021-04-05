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

#include <functional>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

class ShiftPredicateInserter : public kir::MutableIrVisitor {
  void handle(kir::Expr* expr) {
    if (expr->outputs().size() > 0) {
      auto out_tv = dynamic_cast<kir::TensorView*>(expr->outputs()[0]);
      if (out_tv != nullptr) {
        insertProducerPredicate(expr, out_tv);
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
      kir::TensorView* consumer) {
#if 0
    std::cerr << "InsertProducerPred: TV"
              << consumer->name() << std::endl;
#endif

    TensorView* consumer_fuser_tv = consumer->fuserTv();

    auto fuser_expr = consumer->fuserTv()->definition();

    TensorDomain* consumer_td = consumer->fuserTv()->domain();

    const auto num_dims = consumer->domain()->rootDomain().size();

    auto pred_contiguity =
        std::vector<bool>(consumer_td->getRootDomain().size(), true);
    auto consumer_inds = Index::getConsumerRootPredIndices(
        consumer, for_loops_, pred_contiguity).first;

    TORCH_INTERNAL_ASSERT(consumer_inds.size() == consumer->domain()->rootDomain().size());

    kir::Bool* shift_pred = nullptr;
    kir::Bool* bounds_pred = nullptr;

    const HaloMap& halo_map = gpu_lower_->haloMap();

    auto shift_expr = dynamic_cast<ShiftOp*>(fuser_expr);

    bool needs_shift_predicate = false;
    for (size_t i = 0; i < num_dims; ++i) {
      auto consumer_id = consumer_td->getRootDomain()[i];
      const auto consumer_halo_info = halo_map.getHalo(consumer_id);
      if (consumer_halo_info.hasHalo()) {
        needs_shift_predicate = true;
        break;
      }
      if (shift_expr != nullptr && shift_expr->offset(i) != 0) {
        needs_shift_predicate = true;
        break;
      }
    }

    for (size_t i = 0; i < num_dims; ++i) {
      auto consumer_id = consumer_fuser_tv->getRootDomain()[i];

      const auto consumer_halo_info = halo_map.getHalo(consumer_id);

      int shift_offset = 0;
      if (auto shift_expr = dynamic_cast<ShiftOp*>(fuser_expr)) {
        shift_offset = shift_expr->offset(i);
      }

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

      auto right_offset = consumer_inds[i];
      if (shift_offset < 0) {
        right_offset = makeAddExpr(right_offset, -shift_offset);
      }
      if (needs_shift_predicate) {
        shift_pred = makeAndExpr(
            shift_pred,
            ir_builder_
            .ltExpr(
                right_offset,
                makeAddExpr(consumer->domain()->rootDomain()[i]->extent(),
                            consumer_halo_info.width(0))));

        bounds_pred = makeAndExpr(
            bounds_pred,
            ir_builder_.ltExpr(
                consumer_inds[i],
                makeAddExpr(consumer->domain()->rootDomain()[i]->extent(),
                            consumer_halo_info.width())));
      }
    }

    if (shift_pred == nullptr) {
      return;
    }

    //std::cerr << "Shift pred:" << kir::toString(shift_pred) << std::endl;
    auto shift_ite = ir_builder_.create<kir::IfThenElse>(shift_pred);

    auto& scope = for_loops_.back()->body();

    // Insert the if statement
    scope.insert_before(definition, shift_ite);

    // Remove the expr from the list
    scope.erase(definition);

    // Place the expr inside the if statement
    shift_ite->thenBody().push_back(definition);

    // Pads by zero
    auto bounds_ite = ir_builder_.create<kir::IfThenElse>(bounds_pred);
    auto pad_expr = ir_builder_.create<kir::UnaryOp>(
        UnaryOpType::Set, consumer, ir_builder_.create<kir::Int>(0));
    bounds_ite->thenBody().push_back(pad_expr);

    shift_ite->elseBody().push_back(bounds_ite);
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

void HaloMap::build(Fusion* fusion) {
  auto exprs = fusion->exprs();
  for (auto it = exprs.rbegin(); it != exprs.rend(); ++it) {
    auto expr = *it;
    if (!expr->outputs()[0]->isA<TensorView>()) {
      continue;
    }

    propagateHaloInfo(expr);
  }

  auto used_vals = DependencyCheck::getAllValsBetween(
      {fusion->inputs().begin(), fusion->inputs().end()}, fusion->outputs());

  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    updateExtents(tv);
  }

  // Note that validation requires consumer halo info
  for (auto tv : ir_utils::filterByType<TensorView>(used_vals)) {
    validate(tv);
  }
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

  for (size_t i = 0; i < c_root.size(); ++i) {
    auto c_id = c_root[i];
    auto it = c2p.find(c_id);
    if (it == c2p.end()) {
      // nothing to propagate
      continue;
    }

    auto p_id = it->second;

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
  updateExtents(tv->domain());
}

void HaloMap::updateExtents(TensorDomain* td) {
  std::unordered_map<IterDomain*, HaloInfo> inherited_halo;

  auto gpu_lower = GpuLower::current();

  for (auto root_axis: td->getRootDomain()) {
    //std::cerr << "root axis: " << root_axis << std::endl;
    auto& halo_info = findOrCreate(root_axis);
    auto halo_width = halo_info.width();
    if (halo_width == 0) {
      //std::cerr << "root axis has zero halo\n";
      halo_extent_map_.insert({root_axis, 0});
      continue;
    }
    auto expanded_extent = add(root_axis->rawExtent(), new Int(halo_width));
    extent_map_.insert({root_axis, expanded_extent});
    kir_extent_map_.insert({gpu_lower->lowerValue(root_axis)->as<kir::IterDomain>(),
        gpu_lower->lowerValue(expanded_extent)});
    inherited_halo.insert({root_axis, halo_info});
    halo_extent_map_.insert({root_axis, halo_width});
  }

  auto exprs = ExprSort::getExprs(
      td->fusion(),
      std::vector<Val*>(td->domain().begin(), td->domain().end()));

  // Splitting merged overlapped IterDomains is not allowed
  std::unordered_set<IterDomain*> merged_shifted_ids;

  for (auto expr: exprs) {
    //std::cerr << "Visiting expr: " << expr << std::endl;
    if (auto split = dynamic_cast<Split*>(expr)) {
      TORCH_INTERNAL_ASSERT(merged_shifted_ids.find(split->in()) == merged_shifted_ids.end(),
                            "Splitting IterDomain that is a merged domain of shifted domains is not allowed");
      auto in_id = split->in();
      if (extent_map_.find(in_id) == extent_map_.end()) {
        //std::cerr << "Skipping\n";
        halo_extent_map_.insert({split->outer(), 0});
        halo_extent_map_.insert({split->inner(), 0});
        continue;
      }
      if (inherited_halo.find(in_id) == inherited_halo.end()) {
        std::cerr << "inherited halo:\n";
        for (auto kv: inherited_halo) {
          std::cerr << "Inherited: " << kv.first << " -> " << kv.second.toString() << std::endl;
        }
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
      halo_extent_map_.insert({split->outer(), 0});
      halo_extent_map_.insert({split->inner(), halo_info.width()});
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
        merged_shifted_ids.insert(merge->out());
        // Note that halo_extent_map_ is not updated as no halo is
        // meaningfully defined for a merged axis.
      }
    } else {
      TORCH_INTERNAL_ASSERT(false, "Unsupported expr: ", expr);
    }
  }
}

//! Restriction 1: When allocation is outside of a shifted
//! axis, the shifted axis must be guarantted to have a smaller extent
//! than the concrete axis. For now, shifted axes always mean expanded
//! allocations when the axis is located inside the allocation
//! point. This restriction is validated at the allocation lowering
//! pass.
//!
//! Restriction 2: If an expanded axis is parallelized, its memory
//! must be accessible by all other threads. More specifically:
//! - TIDx: It must be on shared memory. May want to consider
//! utilizing the shuffle instructions as well.
//! - BIDx: Not supported. If on global memory, Cooperative Launch
//! may be used to support it, however, it's unclear in what
//! situations block-level parallelization should be used.
//!
//! Other types of parallelization should be supported except for
//! vectorization. Vectorization should be eventually supported but
//! needs further work.
void HaloMap::validate(TensorView* tv) const {
  const auto& par_map = GpuLower::current()->caParallelMap();
  const auto& loop_map = GpuLower::current()->caLoopMap();
  const auto mem_type = tv->getMemoryType();

  for (auto axis : tv->domain()->domain()) {
    auto halo_extent = getExtent(axis);
    // If no halo extent is associated with this axis, it means the
    // axis is not extended.
    if (halo_extent == nullptr) {
      continue;
    }

    // Enforce restrictions on parallelization and memory type
    const auto ptype = par_map.getConcreteMappedID(axis)->getParallelType();

    if (ptype == ParallelType::Serial) {
      continue;
    }

    // Only threading parallelism is considered for now
    TORCH_CHECK(isParallelTypeThread(ptype),
                "Unsupported parallel type: ", ptype);

    bool shared_mem_needed = false;
    for (auto use: tv->uses()) {
      if (!ir_utils::isTVOp(use)) {
        continue;
      }
      if (use->isA<ShiftOp>()) {
        shared_mem_needed = true;
        break;
      }
      auto consumer = use->outputs()[0]->as<TensorView>();
      // Find the corresponding axis in the consumer
      auto it = std::find_if(
          consumer->domain()->domain().begin(),
          consumer->domain()->domain().end(),
          [&](IterDomain* consumer_axis) {
            return loop_map.areMapped(axis, consumer_axis);
          });
      if (it == consumer->domain()->domain().end()) {
        continue;
      }
      if (!extentEqual(axis, *it)) {
        shared_mem_needed = true;
        break;
      }
    }

    if (!shared_mem_needed) {
      continue;
    }

    if (isParallelTypeThreadDim(ptype)) {
      // If all the consumers have the same extent and none of the
      //expressions is shift, any memory should be fine. Otherwise, it
      //must be accessible by all threads involved in the
      //parallelization.
      TORCH_CHECK(mem_type == MemoryType::Shared,
                  "TV", tv->name(), " must be allocated on shared memory as its halo-extended axis is parallelized by ", ptype);

    } else if (isParallelTypeBlockDim(ptype)) {
      TORCH_CHECK(false,
                  "Block-based parallelization of a halo-extended axis is not supported: ",
                  axis);
    }
  }
  return;
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

unsigned HaloMap::getHaloExtent(IterDomain* id) const {
  auto it = halo_extent_map_.find(id);
  TORCH_INTERNAL_ASSERT(it != halo_extent_map_.end());
  return it->second;
}

bool HaloMap::hasHaloExtent(IterDomain* id) const {
  return halo_extent_map_.find(id) != halo_extent_map_.end();
}

namespace {

template <typename Cmp>
bool extentCompare(const HaloMap& halo_map, IterDomain* id1, IterDomain* id2,
                   Cmp cmp) {
  auto gpu_lower = GpuLower::current();
  TORCH_INTERNAL_ASSERT(gpu_lower->caLoopMap().areMapped(id1, id2),
                        "Invalid axes to compare");

  if (halo_map.hasHaloExtent(id1)) {
    TORCH_INTERNAL_ASSERT(halo_map.hasHaloExtent(id2),
                          "Comparing ", id1, " and ", id2, " is invalid.");
    return cmp(halo_map.getHaloExtent(id1), halo_map.getHaloExtent(id2));
  } else {
    TORCH_INTERNAL_ASSERT(!halo_map.hasHaloExtent(id2));
    // Both must be an output of a merge
    auto merge1 = dynamic_cast<Merge*>(id1->definition());
    TORCH_INTERNAL_ASSERT(merge1 != nullptr);
    auto merge2 = dynamic_cast<Merge*>(id2->definition());
    TORCH_INTERNAL_ASSERT(merge2 != nullptr);
    auto inner_le = extentCompare(halo_map, merge1->inner(), merge2->inner(), cmp);
    auto outer_le = extentCompare(halo_map, merge1->outer(), merge2->outer(), cmp);
    return inner_le && outer_le;
  }
}

} // namespace

bool HaloMap::extentLessEqual(IterDomain* id1, IterDomain* id2) const {
  return extentCompare(*this, id1, id2, std::less_equal<unsigned>());
}

bool HaloMap::extentEqual(IterDomain* id1, IterDomain* id2) const {
  return extentCompare(*this, id1, id2, std::equal_to<unsigned>());
}

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
