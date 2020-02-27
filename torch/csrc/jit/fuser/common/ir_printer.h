#pragma once

#include <torch/csrc/jit/fuser/common/iriostream.h>
#include <torch/csrc/jit/fuser/common/iter_visitor.h>

#include <iostream>

namespace torch {
namespace jit {
namespace fuser {

class IRPrinter : public IterVisitor{
public:
  IRPrinter(std::ostream& os) : 
    IterVisitor(),
    irstream_(os) { } 
  
  void print(const Fusion* const fusion){
    std::cout << "Printing IR..." << std::endl;
    traverse(fusion, false /*from_outputs_only*/, {ValType::TensorView}, false /*breadth_first*/);
  }
  void handle(Statement* s) override {
    if (s->isExpr()) {
      Statement::dispatch(this, s);
      //irstream_ << " >>> " << static_cast<const Statement*>(s) << std::endl;
    }
  }                                      
  void handle(Float* val) override { 
    if (val->isSymbolic()) {
      irstream_ << "%f" << val->name();
    } else {
      irstream_ << *(val->value()) << "f";
    }
  }

protected:                             
  std::ostream& irstream_;
};             

} // namespace fuser
} // namespace jit
} // namespace torch
