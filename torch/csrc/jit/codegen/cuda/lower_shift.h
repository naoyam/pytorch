#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <torch/csrc/jit/codegen/cuda/ir_all_nodes.h>
#include <torch/csrc/jit/codegen/cuda/kernel_ir.h>

#include <vector>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

//! Insert predicates for shifts
std::vector<kir::Expr*> insertShiftPredicates(
    const std::vector<kir::Expr*>& exprs);

class HaloInfo {
 public:
  unsigned int width(int pos) const {
    TORCH_INTERNAL_ASSERT(pos >= 0 && pos < 2);
    return widths_[pos];
  }

  unsigned int width() const {
    return std::accumulate(widths_.begin(), widths_.end(), 0);
  }

  const auto& widths() const {
    return widths_;
  }

  void setWidth(int pos, unsigned int width) {
    TORCH_INTERNAL_ASSERT(pos >= 0 && pos < 2);
    widths_[pos] = width;
  }

  void merge(int pos, unsigned int other) {
    setWidth(pos, std::max(width(pos), other));
  }

  void merge(const HaloInfo& other) {
    for (size_t i = 0; i < widths_.size(); ++i) {
      merge(i, other.width(i));
    }
  }

  std::string toString() const {
    std::stringstream ss;
    ss << width(0) << ", " << width(1);
    return ss.str();
  }

  bool hasHalo() const {
    return std::any_of(
        widths_.begin(), widths_.end(), [](auto w) { return w != 0; });
  }

 private:
  std::array<unsigned int, 2> widths_;
};


class HaloMap {
 public:
  void build();

  void updateExtents(TensorView* tv);
  void updateExtents(TensorDomain* td);

  HaloInfo getHalo(IterDomain* id) const;

  //! Returns an extent if id is extended for halo. Nullptr is
  //! returned otherwise.
  Val* getExtent(IterDomain* id) const;
  kir::Val* getExtent(kir::IterDomain* id) const;

  const auto& getExtentMap() const {
    return extent_map_;
  }

  std::string toString() const;

  HaloInfo& findOrCreate(IterDomain* id);
  
 private:  
  void propagateHaloInfo(Expr* expr);
  void propagateHaloInfo(
      TensorView* producer,
      TensorView* consumer,
      Expr* expr);

#if 0
  void buildStartMap(Fusion* fusion);
  void propagateStartInfo(Expr* expr);
  void propagateStartInfo(
      TensorView* producer,
      TensorView* consumer,
      Expr* expr);
#endif

 private:
  std::unordered_map<IterDomain*, HaloInfo> halo_map_;
  std::unordered_map<IterDomain*, Val*> extent_map_;
  std::unordered_map<kir::IterDomain*, kir::Val*> kir_extent_map_;
  //  std::unordered_map<IterDomain*, int> start_map_;
};

std::vector<kir::ForLoop*> removeHaloLoops(
    const std::vector<kir::ForLoop*>& loops);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
