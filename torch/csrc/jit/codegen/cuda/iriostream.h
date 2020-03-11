#pragma once

#include <torch/csrc/jit/codegen/cuda/dispatch.h>
#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

struct Fusion;

struct Statement;

struct Val;
struct Expr;

struct UnaryOp;
struct BinaryOp;

struct ForLoop;
struct IfThenElse;

struct Tensor;
struct TensorDomain;
struct TensorView;
struct IterDomain;
struct TensorIndex;

struct TensorContiguity;


struct Split;
struct Merge;
struct Reorder;

struct Float;
struct Int;
struct Add;

struct TORCH_API IRPrinter : public OptInConstDispatch {

  std::ostream& os;
  bool print_inline_ = false;

public:

  IRPrinter(std::ostream& _os):os(_os){}

  virtual void handle(Fusion* const f);
  
  //handle calls some non const fusion ops,
  //eventhough fusion should remain unchanged.
  //Need to look into this.
  virtual void handle(const Fusion* const f){
    handle(const_cast<Fusion*>(f));
  }
  virtual void handle(Fusion& f){handle(&f);}

  virtual void handle(const Statement* const s) { OptInConstDispatch::handle(s); };

  virtual void handle(const Val* const v) { OptInConstDispatch::handle(v); };
  virtual void handle(const Expr* const e) { OptInConstDispatch::handle(e); };

  virtual void handle(const Tensor* const);
  virtual void handle(const TensorDomain* const);
  virtual void handle(const TensorView* const);
  virtual void handle(const IterDomain* const);
  virtual void handle(const TensorIndex* const);
  virtual void handle(const TensorContiguity* const);

  virtual void handle(const Float* const);
  virtual void handle(const Int* const);

  virtual void handle(const UnaryOp* const);
  virtual void handle(const BinaryOp* const);

  virtual void handle(const ForLoop* const);
  virtual void handle(const IfThenElse* const);

  virtual void handle(const Split* const);
  virtual void handle(const Merge* const);
  virtual void handle(const Reorder* const);

  void print_inline(const Statement* const stmt){
    bool prev = print_inline_;
    print_inline_ = true;
    handle(stmt);
    print_inline_ = prev;
  }

  };

  TORCH_API std::ostream& operator<< (std::ostream& os, const Statement* const stmt);
  TORCH_API std::ostream& operator<< (std::ostream& os, Fusion* f);
  TORCH_API std::ostream& operator<< (std::ostream& os, Fusion& f);

} // namespace fuser
} // namespace jit
} // namespace torch
