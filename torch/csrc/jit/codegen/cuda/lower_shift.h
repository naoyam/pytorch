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

  HaloInfo get(IterDomain* id) const;

  std::string toString() const;

 private:
  HaloInfo& findOrCreate(IterDomain* id);
  void propagateHaloInfo(Expr* expr);
  void propagateHaloInfo(
      TensorView* producer,
      TensorView* consumer,
      Expr* expr);

  void propagateFromRootDomain(TensorView* tv);

 private:
  std::unordered_map<IterDomain*, HaloInfo> map_;
};

std::vector<kir::ForLoop*> removeHaloLoops(
    const std::vector<kir::ForLoop*>& loops);

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
