#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/codegen/cuda/type.h>

#include <cstdint>
#include <deque>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

//  TODO: Add more types (int32, int64)
//  TODO: add scopes (like the region of a loop) more similarly to Fusion

/*
 * This file defines the basic IR structure.
 * IR is any information that the code generation stack may need for analysis.
 * By analysis we're refering to anything done in response to a user facing call
 * of this stack. This could be careful tracking of user calls, and any
 * transformation including optimizing transformations, user declared
 * transformations, and lowering the IR.
 *
 * For now the IR has 4 major classes:
 *
 * Statement:
 * Statement should be inhereited at some point by every IR node. It may be
 * better to rename Statement to node. We use Statements to pass around nodes of
 * unknown compile type. Therefore it is also important for the design to have a
 * dispatch system for a Statment. Basically beinng able to succienctly traverse
 * down the inhereitance stack of a Statment at runtime. This is currently
 * implemented in dispatch.h
 *
 * Val:
 * Val can generally be thought of as representing any type of data. This could
 * be a float that is either a constant (constants in this context could be
 * compile time or run time known from the perspective of a pytorch end user).
 * Some examples:
 *     a constant size like convolution filter width
 *     a runtime constant like batch normalizations momentum
 *     a "symbolic" tensor like one passed down from the JIT
 *     a memory buffer used in device code
 *
 * Adding a Val:
 * Right now adding a Val is quite involved. Val's can be defined in ir.h or in
 * their own header file. The following is what is currently needed to add a new
 * Val: 1) Definition inheriting from Val
 *     - Members must be private or protected
 *     - Accessor functions for members
 *     - Must call Val constructor, Val constructor registers with fusion
 *     - Implementation of bool same_as(...)
 * 2) dispatch.h/.cpp must be updated to include dispatch of the new Val
 * 3) Default mutator function should be added to mutator.h/.cpp
 * 4) Printing functions should be added to iriostream.h/.cpp
 * 5) An enum value must be added to ValType in type.h
 * 6) A string entry must be added in val_type_string_map
 *
 * IRInputOutput:
 * A function on Vals. Has inputs and outputs that are all Vals. Anything that
 * connects values and therefore would be used during dependency analysis.
 * Examples:
 *   binary operations on tensors, scalar values, or a combination, a thread all
 * reduce, for loops
 *
 *
 * Expr
 * Expr is an IRInputOutput node. It takes multiple inputs and does *an*
 * operation. There are specializations of BinaryOp which takes 2 inputs and
 * produces 1 output, and UnaryOp which takes 1 input and produces 1 output.
 *
 * The IR is static single assignment (SSA). Values can only be defined once. If
 * they are re-defined the original definition is deleted from the program, as
 * opposed to an ordered redefinition of the value in the program.
 *
 * Adding an Expr:
 * Right now adding an Expr is quite involved. Expr's can be defined in ir.h or
 * in their own header file. The following is what is currently needed for Expr
 * definitions: 1) Definition inheriting from Expr.
 *     - Members must be private or protected
 *     - Accessor functions for members
 *     - Constructors need to register with the Fusion after inputs/outputs are
 * defined
 *     - Implementation of bool same_as(...)
 * 2) dispatch.h/.cpp must be updated to include dispatch of the new Val
 * 3) Default mutator function should be added to mutator.h/.cpp
 * 4) Printing functions should be added to iriostream.h/.cpp
 * 5) Lower case convenience functions should be added to arith.h/.cpp
 * 6) An enum value must be added to ExprType in type.h
 * 7) A string entry must be added in expr_type_string_map
 *
 */

namespace torch {
namespace jit {
namespace fuser {

using StmtNameType = unsigned int;
constexpr StmtNameType UNINITIALIZED_STMTNAMETYPE =
    std::numeric_limits<unsigned int>::max();

struct Fusion;
struct FusionGuard;
struct Expr;
struct UnaryOp;
struct BinaryOp;
struct IterDomain;

/*
 * Statement is the highest level node representation. Everything that is
 * considered "IR" will be derived from this class eventually. Both Values and
 * Expr's are a Statement. If there will ever be any more fundamental types,
 * they will also derive from Statement.
 */
struct TORCH_API Statement {
  virtual ~Statement() = 0;

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Statement*);

  template <typename T>
  static void const_dispatch(T handler, const Statement* const);

  template <typename T>
  static Statement* mutator_dispatch(T mutator, Statement*);

  // Accessor functions to types. Vals always have a DataType, Exprs never do
  virtual c10::optional<ValType> getValType() const noexcept {
    return c10::nullopt;
  }
  virtual c10::optional<DataType> getDataType() const {
    return c10::nullopt;
  }
  virtual c10::optional<ExprType> getExprType() const noexcept {
    return c10::nullopt;
  }

  // Short cut to figure out if it is a value/expression
  bool isVal() const noexcept {
    return getValType() != c10::nullopt;
  }
  bool isExpr() const noexcept {
    return getExprType() != c10::nullopt;
  }

  // Return the fusion this statement belongs to
  Fusion* fusion() const noexcept {
    return fusion_;
  }

  // Return the int that represents its name
  StmtNameType name() const noexcept {
    return name_;
  }

  // Return if this statement is the same as another statement
  virtual bool same_as(const Statement* const other) const {
    return this == other;
  }

 protected:
  StmtNameType name_ = UNINITIALIZED_STMTNAMETYPE;
  Fusion* fusion_ = nullptr;
};

/*
 * A Val represents a "value." These are objects, like tensors, scalars, and
 * memory locations, that are inputs and outputs of computations (represented
 * by Exprs, below). They also represent the flow of data through a program.
 *
 * Vals are constant and unique. Vals should always be passed around as a
 * pointer.
 */
struct TORCH_API Val : public Statement {
 public:
  virtual ~Val() = 0;

  Val() = delete;
  Val(ValType _vtype, DataType _dtype = DataType::Null);

  // TODO: Values are unique and not copyable
  Val(const Val& other) = delete;
  Val& operator=(const Val& other) = delete;

  Val(Val&& other) = delete;
  Val& operator=(Val&& other) = delete;

  c10::optional<ValType> getValType() const noexcept override;

  // Throws if no DataType is found. Vals must have a DataType
  c10::optional<DataType> getDataType() const override;

  bool isScalar();

  // Returns the Expr that this value is an output of, returns nullptr if none
  // was found
  Expr* getOrigin();

  // TODO: Make this more sophisticated. A value being the same as another value
  // should be evaluated based on the DAG that created it, and that DAGs leaf
  // nodes
  bool same_as(const Val* const other) const {
    return this == other;
  }

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Val*);

  template <typename T>
  static void const_dispatch(T handler, const Val* const);

  template <typename T>
  static Statement* mutator_dispatch(T mutator, Val*);

 protected:
  const ValType vtype_;
  const DataType dtype_;
};

/*
 * IRInputOutput should be used for any type of node that has values as inputes
 * and outputs. Typically classes that inherit from IRInputOutput will also
 * inherit from Expr. Expr's are expected for most dependency based operations
 * like IterVisitor, or DependencyCheck.
 */
struct TORCH_API IRInputOutput {
  virtual ~IRInputOutput() = 0;

  // Returns if Val is an input or output of this IRInputOutput instance
  bool hasInput(const Val* const input) const;
  bool hasOutput(const Val* const output) const;

  // Input/output accessors
  void addInputAt(std::deque<Val*>::size_type pos, Val* input) {
    inputs_.insert(inputs_.begin() + pos, input);
  }

  void addOutputAt(std::deque<Val*>::size_type pos, Val* output) {
    outputs_.insert(outputs_.begin() + pos, output);
  }

  const std::deque<Val*>& inputs() const noexcept {
    return inputs_;
  }
  const std::deque<Val*>& outputs() const noexcept {
    return outputs_;
  }

  Val* input(std::deque<Val*>::size_type idx) const {
    return inputs_[idx];
  }
  Val* output(std::deque<Val*>::size_type idx) const {
    return outputs_[idx];
  }

  void addInput(Val* input) {
    inputs_.push_back(input);
  }
  void addOutput(Val* output) {
    outputs_.push_back(output);
  }

  void removeOutput(Val* val);

  std::deque<Val*>::size_type nInputs() const noexcept {
    return inputs_.size();
  }
  std::deque<Val*>::size_type nOutputs() const noexcept {
    return outputs_.size();
  }

 protected:
  std::deque<Val*> inputs_;
  std::deque<Val*> outputs_;
};

/*
 * A Expr represents a "computation." These are functions that may take inputs
 * and produce outputs.
 *
 * Exprs are unique and immutable. Conceptually, Exprs could always be
 * manipulated using unique pointers, and we could add this later. However, for
 * now Exprs can be replaced in a fusion, but they cannot be modified in place.
 *
 * Note: Registering an Expr with a Fusion is actually 2 parts, one part is done
 * in the Expr constructor, so that should be called on anything that inherits
 * Expr. The issue with having registration in Expr's constructor, is that the
 * constructor of an Expr will set ouputs and inputs. This information is
 * important for registration with Fuser, so it can track the dependency chain.
 */
struct TORCH_API Expr : public Statement, IRInputOutput {
 public:
  virtual ~Expr() = 0;
  Expr() = delete;
  Expr(ExprType _type);

  Expr(const Expr& other) = delete;
  Expr& operator=(const Expr& other) = delete;

  Expr(Expr&& other) = delete;
  Expr& operator=(Expr&& other) = delete;

  c10::optional<ExprType> getExprType() const noexcept override {
    return type_;
  }
  ExprType type() const noexcept {
    return type_;
  }

  virtual bool same_as(const Expr* const other) const {
    if (getExprType() != other->getExprType())
      return false;
    if (inputs().size() != other->inputs().size() ||
        outputs().size() != other->outputs().size())
      return false;
    for (int i = 0; i < inputs().size(); i++) {
      if (!input(i)->same_as(other->input(i)))
        return false;
    }
    return true;
  }

  // Dispatch functions, definitions in dispatch.cpp
  template <typename T>
  static void dispatch(T handler, Expr*);

  template <typename T>
  static void const_dispatch(T handler, const Expr* const);

  template <typename T>
  static Statement* mutator_dispatch(T mutator, Expr*);

 private:
  ExprType type_;
};

} // namespace fuser
} // namespace jit
} // namespace torch
