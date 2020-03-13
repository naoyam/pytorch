#include <test/cpp/jit/test_base.h>

// #include <torch/csrc/jit/fuser/interface.h>
#include <torch/csrc/jit/codegen/cuda/ir_interface_nodes.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/iriostream.h>
#include <torch/csrc/jit/codegen/cuda/mutator.h>
#include <torch/csrc/jit/codegen/cuda/tensor.h>
#include <torch/csrc/jit/codegen/cuda/tensor_meta.h>
#include <torch/csrc/jit/codegen/cuda/transform_replay.h>
#include <torch/csrc/jit/codegen/cuda/code_write.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

// fuser and IR parser
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/ir/irparser.h>

#include <iostream>

// Tests go in torch::jit
namespace torch {
namespace jit {

using namespace torch::jit::fuser;

// 1. Test cases are void() functions.
// 2. They start with the prefix `test`

void testGPU_FusionDispatch() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f = new Float{2.f};

  std::cout << "Dispatch 2.f by Float reference: " << f << std::endl;

  std::cout << "Dispatch 2.f by Val reference: " << static_cast<Val*>(f)
            << std::endl;

  std::cout << "Dispatch 2.f by Statement reference: "
            << static_cast<Statement*>(f) << std::endl;
}

void testGPU_FusionSimpleArith() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f1 = new Float(1.f);
  Float* f2 = new Float{2.f};

  auto f3 = add(f1, f2);
  std::cout << "Explicit add construction of 1.f + 2.f: " << fusion
            << std::endl;
}

void testGPU_FusionContainer() {
  Fusion fusion1;
  FusionGuard fg(&fusion1);

  Float* f1 = new Float(1.f);
  Float* f2 = new Float(2.f);
  auto f3 = add(f1, f2);
  std::cout << "Implicit add construction of 1.f + 2.f : " << fusion1
            << std::endl;
  
  Fusion fusion2;
  {
    FusionGuard fg2(&fusion2);
    Float* f3 = new Float(1.f);
    Float* f4 = new Float(2.f);
    auto f5 = add(f3, f4);
    TORCH_CHECK(
       FusionGuard::getCurFusion()->used(f3)
    && FusionGuard::getCurFusion()->used(f4)
    && !FusionGuard::getCurFusion()->used(f5));

    TORCH_CHECK(FusionGuard::getCurFusion() == &fusion2);
  }

  TORCH_CHECK(FusionGuard::getCurFusion() == &fusion1);
}

void testGPU_FusionSimpleTypePromote() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f4 = new Float{4.f};
  Int* i1 = new Int{3};
  auto f5 = add(f4, i1);

  TORCH_CHECK(f5->getDataType() == DataType::Float);
}

void testGPU_FusionCastOp() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f3_test = new Float{3.f};
  Int* i3 = new Int{3};
  auto f3 = castOp(DataType::Float, i3);

  TORCH_CHECK(f3->getDataType().value() == f3_test->getDataType().value());
}

class ZeroMutator : public OptOutMutator {
 public:
  Statement* mutate(Float* f) {
    if (f->isConst() && *(f->value()) == 1.0)
      return new Float(0.0);
    return f;
  }
  void mutate(Fusion* f){
    OptOutMutator::mutate(f);
  }
};

void testGPU_FusionMutator() {
  
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f4 = new Float{1.f};
  Int* i1 = new Int{3};
  Val* f5 = binaryOp(BinaryOpType::Add, f4, i1);
  
  ZeroMutator mutator;
  mutator.mutate(&fusion);
  Val* lhs = static_cast<BinaryOp*>(fusion.origin(f5))->lhs();
  TORCH_CHECK(lhs->getValType().value() == ValType::Scalar && lhs->getDataType().value() == DataType::Float);
  Float* flhs = static_cast<Float *>( lhs );
  
  TORCH_CHECK(flhs->value().value() == 0.f);
  
}

void testGPU_FusionRegister() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  Float* v1 = new Float{1.f};
  Float* v2 = new Float{2.f};
  Val* v3 = binaryOp(BinaryOpType::Add, v1, v2);
  Val* v4 = binaryOp(BinaryOpType::Add, v1, v2);
  TORCH_CHECK(v1->name() + 1 == v2->name());
  TORCH_CHECK(v2->name() + 1 == v3->name());
  TORCH_CHECK(v3->name() + 1 == v4->name());
  TORCH_CHECK(fusion.origin(v3)->name() + 1 == fusion.origin(v4)->name());
}

// dummy expr with 2 outputs only for toposort test.
struct TORCH_API DummyExpr : public Expr {
  ~DummyExpr() = default;
  DummyExpr(Val* _outlhs, Val* _outrhs, Val* _lhs, Val* _rhs)
      : Expr(ExprType::BinaryOp) // Not terribly safe...
  {
    addOutput(_outlhs);
    addOutput(_outrhs); addInput(_lhs);
    addInput(_rhs);
    this->name_ = FusionGuard::getCurFusion()->registerExpr(this);
  }
  DummyExpr(const DummyExpr& other) = delete;
  DummyExpr& operator=(const DummyExpr& other) = delete;
  DummyExpr(DummyExpr&& other) = delete;
  DummyExpr& operator=(DummyExpr&& other) = delete;
};

void testGPU_FusionTopoSort() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // e0: v3, v2 = dummy(v1, v0)
  // e1: v4     =   add(v3, v2)
  // e2: v5     =   add(v2, v4)
  // e3: v6     =   add(v5, v5)
  Float* v0 = new Float{1.f};
  Float* v1 = new Float{2.f};
  Float* v2 = new Float();
  Float* v3 = new Float();
  Float* v4 = new Float();
  Float* v5 = new Float();
  Float* v6 = new Float();

  Expr* e0 = new DummyExpr(v3, v2, v1, v0);
  Expr* e1 = new BinaryOp(BinaryOpType::Add, v4, v3, v2);
  Expr* e2 = new BinaryOp(BinaryOpType::Add, v5, v2, v4);
  Expr* e3 = new BinaryOp(BinaryOpType::Add, v6, v5, v5);

  std::vector<Expr*> exprs = fusion.exprs();

  TORCH_CHECK(exprs.size() == 4);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);
  TORCH_CHECK(exprs[3] == e3);

  fusion.addOutput(v2);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs.size() == 1);
  TORCH_CHECK(exprs[0] == e0);

  fusion.addOutput(v5);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v4);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v3);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);

  fusion.addOutput(v6);
  exprs = fusion.exprs(true);
  TORCH_CHECK(exprs.size() == 4);
  TORCH_CHECK(exprs[0] == e0);
  TORCH_CHECK(exprs[1] == e1);
  TORCH_CHECK(exprs[2] == e2);
  TORCH_CHECK(exprs[3] == e3);

  TORCH_CHECK(fusion.origin(v2)->name() == 0);
  TORCH_CHECK(fusion.origin(v3)->name() == 0);
  TORCH_CHECK(fusion.origin(v4)->name() == 1);
  TORCH_CHECK(fusion.origin(v5)->name() == 2);
  TORCH_CHECK(fusion.origin(v6)->name() == 3);
}

void testGPU_FusionTensor() {
  auto tensor = at::randn({2, 3, 4, 5}, at::kCUDA);
  auto sizes = tensor.sizes().vec();
  auto tensor_type = TensorType::create(tensor);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto fuser_tensor = new Tensor(tensor_type);
  TORCH_CHECK(fuser_tensor->getDataType().value() == DataType::Float);
  TORCH_CHECK(fuser_tensor->domain() != nullptr);

  auto fuser_null_tensor = new Tensor(DataType::Int);
  TORCH_CHECK(fuser_null_tensor->getDataType().value() == DataType::Int);
  TORCH_CHECK(fuser_null_tensor->domain() == nullptr);
}

void testGPU_FusionTensorContiguity() {
  {
    // NCHW memory layout
    auto tensor = at::randn({2, 3, 4, 5});
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK(t_c.rank() == 4);
    TORCH_CHECK(t_c.getBroadcastDims().size() == 0);
    for (int i = 0; i < 4; i++) {
      TORCH_CHECK(!t_c.isBroadcastDim(i));
      if (i < 3) {
        TORCH_CHECK(t_c.canCollapseToHigher(i));
      }
    }
  }

  {
    // NHWC memory layout
    TensorContiguity t_c({2, 3, 4, 5}, {60, 1, 15, 3});
    TORCH_CHECK(t_c.rank() == 4);
    TORCH_CHECK(t_c.getBroadcastDims().size() == 0);
    for (int i = 0; i < 4; i++) {
      TORCH_CHECK(!t_c.isBroadcastDim(i));
      if (i < 3) {
        TORCH_CHECK((t_c.canCollapseToHigher(i) ^ (i != 2)));
      }
    }
  }

  {
    // NHWC memory layout with broadcast
    TensorContiguity t_c({2, 3, 4, 5}, {120, 0, 30, 3});
    TORCH_CHECK(t_c.rank() == 4);
    auto b_dims = t_c.getBroadcastDims();
    TORCH_CHECK(b_dims.size() == 1 && b_dims[0] == 1);
    for (int i = 0; i < 4; i++) {
      TORCH_CHECK(!(t_c.isBroadcastDim(i)) ^ (i == 1));
      if (i < 3) {
        TORCH_CHECK(!(t_c.canCollapseToHigher(i)));
      }
    }
  }

  {
    // contiguity across size-1 dimension
    auto tensor = at::randn({4, 1, 4});
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    auto dim = sizes.size();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK(t_c.rank() == sizes.size());
    auto b_dims = t_c.getBroadcastDims();
    TORCH_CHECK(b_dims.size() == 0);
    TORCH_CHECK(t_c.getFCD() == 2);
    TORCH_CHECK(t_c.hasContiguousFCD());
    for (int i = 0; i < dim; i++) {
      TORCH_CHECK(!t_c.isBroadcastDim(i));
      if (i < dim - 1) {
        TORCH_CHECK(t_c.canCollapseToHigher(i));
      }
    }
  }

  {
    // no contiguity across size-1 dimension
    auto tensor = at::randn({4, 4, 4}).split(1, 1)[0];
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK(!(t_c.canCollapseToHigher(0)));
    TORCH_CHECK((t_c.canCollapseToHigher(1)));
  }

  {
    // no contiguity across size-1 dimension
    auto tensor = at::randn({4, 1, 8}).split(4, 2)[0];
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK((t_c.canCollapseToHigher(0)));
    TORCH_CHECK((!t_c.canCollapseToHigher(1)));
  }

  {
    // no contiguity across size-1 dimension
    auto tensor = at::randn({8, 1, 4}).split(4, 0)[0];
    auto sizes = tensor.sizes().vec();
    auto strides = tensor.strides().vec();
    TensorContiguity t_c(sizes, strides);
    TORCH_CHECK((t_c.canCollapseToHigher(0)));
    TORCH_CHECK((t_c.canCollapseToHigher(1)));
  }

  {
    // test merge
    TensorContiguity t_c_l({4, 4, 4}, {16, 4, 1});
    TensorContiguity t_c_r({4, 4, 4}, {16, 4, 1});
    t_c_l.merge(t_c_r);
    TORCH_CHECK((t_c_l.isIdentical(t_c_r)));
  }

  {
    TensorContiguity t_c_l({4, 4, 4, 4}, {16, 0, 4, 1});
    TensorContiguity t_c_r({4, 4, 4, 4}, {64, 16, 4, 1});
    t_c_l.merge(t_c_r);
    TORCH_CHECK(t_c_l.getFCD() == 3);
    TORCH_CHECK(t_c_l.getAxisByStride(0) == 0);
  }

  {
    // NHWC + NCHW
    TensorContiguity t_c_l({4, 4, 4, 4}, {64, 16, 4, 1});
    TensorContiguity t_c_r({4, 4, 4, 4}, {64, 1, 16, 4});
    t_c_l.merge(t_c_r);
    TORCH_CHECK(!t_c_l.hasContiguousFCD());
    TORCH_CHECK(t_c_l.getFCD() == -1);
    TORCH_CHECK(t_c_l.getAxisByStride(0) == 0);
    TORCH_CHECK(t_c_l.getAxisByStride(1) == -1);
    TORCH_CHECK(t_c_l.getAxisByStride(2) == -1);
    TORCH_CHECK(t_c_l.getAxisByStride(3) == -1);
  }

  {
    // NCHW + NCHW with broadcasting
    TensorContiguity t_c_l({4, 4, 4, 4}, {4, 1, 4, 0});
    TensorContiguity t_c_r({4, 4, 4, 4}, {64, 1, 16, 4});
    t_c_l.merge(t_c_r);
    TORCH_CHECK(t_c_l.getFCD() == 1);
    TORCH_CHECK(t_c_l.getAxisByStride(0) == 0);
  }
}

void testGPU_FusionTVSplit() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = new TensorView(Tensor::MakeDummyTensor(3));

  tv = tv->split(2, 2);
  std::cout << "Split: " << tv << std::endl;

  std::cout << "Split fusion output: " << fusion << std::endl;
}

void testGPU_FusionTVMerge() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* tv = new TensorView(Tensor::MakeDummyTensor(3));

  tv = tv->merge(1);

  std::cout << "Merge fusion output: " << fusion << std::endl;
}

void testGPU_FusionTVReorder() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Tensor* dummyTensor = Tensor::MakeDummyTensor(3);

  std::unordered_map<int, int> shift_right{{-1, 0}};

  std::unordered_map<int, int> shift_left{{0, -1}};

  std::unordered_map<int, int> shift_left_2{{0, -1}, {1, 0}, {2, 1}};

  std::unordered_map<int, int> swap{{0, 2}, {2, 0}};
  TensorView* ref = new TensorView(dummyTensor);
  TensorView* tv = new TensorView(dummyTensor);

  TensorView* s_leftl = tv->reorder(shift_left);
  for (int i = 0; i < tv->nDims(); i++)
    TORCH_CHECK(ref->axis(i) == s_leftl->axis(i - 1));

  tv = new TensorView(dummyTensor);
  TensorView* s_left2 = tv->reorder(shift_left);
  for (int i = 0; i < tv->nDims(); i++)
    TORCH_CHECK(ref->axis(i) == s_left2->axis(i - 1));

  tv = new TensorView(dummyTensor);
  TensorView* s_right = tv->reorder(shift_right);
  for (int i = 0; i < tv->nDims(); i++)
    TORCH_CHECK(ref->axis(i - 1) == s_right->axis(i));

  tv = new TensorView(dummyTensor);
  TensorView* rswap = tv->reorder(swap);
  TORCH_CHECK(ref->axis(0) == rswap->axis(2));
  TORCH_CHECK(ref->axis(2) == rswap->axis(0));
  TORCH_CHECK(ref->axis(1) == rswap->axis(1));
}

void testGPU_FusionEquality() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* fval1 = new Float();
  Float* fval1_copy = fval1;
  Float* fval2 = new Float();
  Float* fone = new Float(1.0);

  TORCH_CHECK(fval1->sameAs(fval1_copy));
  TORCH_CHECK(!fval1->sameAs(fval2));
  TORCH_CHECK(!fone->sameAs(fval1));
  TORCH_CHECK(fone->sameAs(new Float(1.0)));

  Int* ival1 = new Int();
  Int* ival1_copy = ival1;
  Int* ival2 = new Int();
  Int* ione = new Int(1);

  TORCH_CHECK(ival1->sameAs(ival1_copy));
  TORCH_CHECK(!ival1->sameAs(ival2));
  TORCH_CHECK(!ione->sameAs(ival1));
  TORCH_CHECK(ione->sameAs(new Int(1)));

  BinaryOp* add1 = new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  BinaryOp* add1_copy =
      new BinaryOp(BinaryOpType::Add, new Float(), fval1, ival1);
  BinaryOp* sub1 = new BinaryOp(BinaryOpType::Sub, new Float(), fval1, ival1);

  UnaryOp* neg1 = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);
  UnaryOp* neg2 = new UnaryOp(UnaryOpType::Neg, new Float(), fval2);
  UnaryOp* neg1_copy = new UnaryOp(UnaryOpType::Neg, new Float(), fval1);

  TORCH_CHECK(add1->sameAs(add1_copy));
  TORCH_CHECK(!add1->sameAs(sub1));

  TORCH_CHECK(neg1->sameAs(neg1_copy));
  TORCH_CHECK(!static_cast<Expr*>(neg1)->sameAs(add1));
  TORCH_CHECK(!neg1->sameAs(neg2));
}

void testGPU_FusionReplaceAll() {

  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f0 = new Float();
  Float* f1 = new Float{1.f};
  Float* f2 = new Float{2.f};
  Float* f3 = new Float();
  Float* f4 = static_cast<Float*>(add(f1, f0));

  // replace the output f4 with f3
  ReplaceAll::instancesOf(f4, f3);

  // f3 should now have an origin function
  TORCH_CHECK(fusion.origin(f3) != nullptr);

  // Should have removed f4 completely so we shouldn't have any other expr than
  // f3 construction
  TORCH_CHECK(fusion.exprs().size() == 1);

  // Replace constant Float's of value 1.f with 2.f
  ReplaceAll::instancesOf(f1, f2);
  BinaryOp* bop = static_cast<BinaryOp*>(fusion.origin(f3));
  // make sure the binary op (origin of f3) actually changed to 2.f
  TORCH_CHECK(static_cast<Float*>(bop->lhs())->sameAs(new Float{2.f}));

}

void testGPU_FusionComputeAt() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<IterDomain*> dom;
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int(), ParallelType::Serial, true));
  dom.push_back(new IterDomain(new Int()));

  TensorDomain* td = new TensorDomain(dom);
  //TensorView* tv0 = new TensorView(new Tensor(DataType::Float, td));
  TensorView* tv0 = new TensorView(td, DataType::Float);
  new BinaryOp(BinaryOpType::Add, tv0, new Float(0.0), new Float(1.0));
  TensorView* tv1 = static_cast<TensorView*>(add(tv0, new Float(2.0)));
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(3.0)));

  ASSERT_ANY_THROW(tv0->computeAt(tv2, 3));

  //[I0, I1, I3]
  tv2 = tv2->split(0, 4);
  //[I0o, I0i{4}, I1, I3]
  tv2 = tv2->merge(1);
  //[I0o, I0i{4}*I1, I3]
  tv2 = tv2->split(-1, 2);
  //[I0o, I0i{4}*I1, I3o, I3i{2}]
  tv2 = tv2->reorder({{0, 1}, {1, 0}, {3, 2}});
  //[I0i{4}*I1, I0o, I3i, I3o{2}]
  std::cout << "Replaying: " << td << "\n-> " << tv2 << "\n on " << tv0
            << " and " << tv1 << "\nwith \'compute_at(2)\' produces:\n"
            << tv0->computeAt(tv2, 2)
            << "\nWhich should along the lines of:"
            << "\n[I0i{4}*I1, I0o, R2, I3]\n"
            << tv1 << "\nshould be along the lines of: "
            << "\n[I0i{4}*I1, I0o, I3]" << std::endl;
  
 }

void testGPU_FusionComputeAt2() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<IterDomain*> dom;
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));

  TensorDomain* td = new TensorDomain(dom);
  TensorView* tv0 = new TensorView(new Tensor(DataType::Float, td));
  new BinaryOp(BinaryOpType::Add, tv0, new Float(0.0), new Float(1.0));

  TensorView* tv1 = static_cast<TensorView*>(add(tv0, new Float(1.0)));
  
  //[I0, I1, I2, I3]
  tv1 = tv1->split(-1, 4);
  //[I0, I1, I2, I3o, I3i{4}]
  tv1 = tv1->reorder({{3, 0}, {0, 3}, {1, 4}, {4, 1}});
  //[I3o, I3i{4}, I2, I0, I1]
  tv1 = tv1->split(3, 2);
  //[I3o, I3i{4}, I2, I0o, I0i{2}, I1]
  tv1 = tv1->reorder(
      {
          {3, 0}, {4, 1}, {5, 2}, {2, 3}
          //{0, 4} //doesn't need to be specified
          //{1, 5} //doesn't need to be specified
      });
  //[I0o, I0i{2}, I1, I2, I3o, I3i{4}]

  std::cout << "Replaying: " << td << "\n -> " << tv1 << "\n on " << tv0
            << "\n with \'compute_at(2)\' produces: \n"
            << tv0->computeAt(tv1, 2) << std::endl;
  std::cout << "Which should be along the lines of:";
  std::cout << "[I0o, I0i{2}, I1, I2, I3]" << std::endl;
  
}

void testGPU_FusionComputeAt3() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<IterDomain*> dom;
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));

  TensorDomain* td = new TensorDomain(dom);
  TensorView* tv0 = new TensorView(new Tensor(DataType::Float, td));
  
  new BinaryOp(BinaryOpType::Add, tv0, new Float(0.0), new Float(1.0));
  TensorView* tv1 = static_cast<TensorView*>(add(tv0, new Float(2.0)));
  TensorView* tv2 = static_cast<TensorView*>(add(tv0, new Float(3.0)));
  TensorView* tv3 = static_cast<TensorView*>(add(tv2, new Float(4.0)));
  
  //tv0 =   0 + 1
  //tv1 = tv0 + 2
  //tv2 = tv0 + 3
  //tv3 = tv2 + 4
  std::cout << "Replaying " << tv3 << "->";
  //[I0, I1]
  tv3 = tv3->split(0, 4);
  //[I0o, I0i{4}, I1]
  tv3 = tv3->reorder({{2, 0}});
  //[I1, I0o, I0i{4}]
  tv3 = tv3->split(0, 2);
  //[I1o, I1i{2} I0o, I0i{4}]
  tv3 = tv3->reorder( { {0, 2}, {1, 3} } );
  //[I0o, I0i{4}, I1o, I1i{2}]

  std::cout << tv3 <<std::endl;
  tv0->computeAt(tv3, 1);

  std::cout << "on to:\n" << tv0 << "\n" << tv2 << "\nand\n" << tv1 << std::endl;
  std::cout << "These domains should approximately be: [I0o, I0i{4}, I1]" << std::endl;
}

void testGPU_FusionParser() {
  /*
  auto g = std::make_shared<Graph>();
  const auto graph0_string = R"IR(
    graph(%0 : Float(2, 3, 4),
          %1 : Float(2, 3, 4)):
      %c0 : Float(2, 3, 4) = aten::mul(%0, %1)
      %d0 : Float(2, 3, 4) = aten::mul(%c0, %0)
      return (%d0))IR";
  torch::jit::script::parseIR(graph0_string, g.get());

  // strides are not yet supported in the irparser.
  for (auto val : g->block()->inputs()) {
    if (val->isCompleteTensor())
      val->setType(val->type()->cast<TensorType>()->contiguous());
  }
  for (auto node : g->block()->nodes()) {
    for (auto val : node->outputs()) {
      if (val->isCompleteTensor())
        val->setType(val->type()->cast<TensorType>()->contiguous());
    }
  }

  Fusion fusion;
  FusionGuard fg(&fusion);
  fuser::cuda::parseJitIR(g, fusion);
  
  CodeWrite cw(std::cout);
  cw.traverse(&fusion);
  */
}

void testGPU_FusionDependency() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  Float* f0 = new Float(0.f);
  Float* f1 = new Float(1.f);
  auto f2 = add(f0, f1);

  auto f3 = add(f2, f2);

  Float* f4 = new Float(4.f);
  Float* f5 = new Float(5.f);
  auto f6 = add(f4, f5);

  Float* f7 = new Float(7.f);
  Float* f8 = new Float(8.f);
  auto f9 = add(f7, f8);

  auto f10 = add(f6, f9);

  auto f11 = add(f3, f10);

  TORCH_CHECK(DependencyCheck::isDependencyOf(f0, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f1, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f2, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f3, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f6, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f9, f11));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f0, f2));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f2, f3));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f4, f6));
  TORCH_CHECK(DependencyCheck::isDependencyOf(f8, f10));

  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f0));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f1));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f2));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f3));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f4));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f11, f5));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f2, f0));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f3, f2));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f6, f4));
  TORCH_CHECK(!DependencyCheck::isDependencyOf(f10, f8));

  std::stack<Val*> dep_chain = DependencyCheck::getDependencyChain(f0, f11);
  TORCH_CHECK(dep_chain.top() == f11);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f3);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f2);
  dep_chain.pop();

  dep_chain = DependencyCheck::getDependencyChain(f6, f11);
  TORCH_CHECK(dep_chain.top() == f11);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f10);
  dep_chain.pop();

  dep_chain = DependencyCheck::getDependencyChain(f4, f11);
  TORCH_CHECK(dep_chain.top() == f11);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f10);
  dep_chain.pop();
  TORCH_CHECK(dep_chain.top() == f6);
  dep_chain.pop();

  dep_chain = DependencyCheck::getDependencyChain(f11, f2);
  TORCH_CHECK(dep_chain.empty());
}

void testGPU_FusionTwoAdds() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // This is the beginning of an example where two Adds are fused where their computation
  // is unrolled and vectorized per thread.

  /**** Tensor Storage       ****/

  // All Tensors have TensorDomain Shapes of [16]
  // T3 is notably the only intermediate that is not I/O
  const auto T0  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T1  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T2  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T3  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T4  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));

  fusion.addInput(T0);
  fusion.addInput(T1);
  fusion.addInput(T2);
  fusion.addOutput(T4);

  auto TV0 = new TensorView(T0);
  auto TV1 = new TensorView(T1);
  auto TV2 = new TensorView(T2);
  auto TV3 = new TensorView(T3);
  auto TV4 = new TensorView(T4);
  
  /**** Operator Expressions ****/ 

  new BinaryOp(BinaryOpType::Add, TV3, T0, T1);
  new BinaryOp(BinaryOpType::Add, TV4, TV3, T2);

  /**** Tensor Expressions   ****/ 
 
  // [x] -> [16/4=4, 4]
  TV4 = TV4->split(-1, 4);
  // [x/4, 4] -> [16/4=4, 4/2=2, 2]
  TV4 = TV4->split(-1, 2); 

  // Compute T3 at inner loop of T4 but allow vectorization.
  TV3->computeAt(TV4, 1);
  
  fusion.print();
}


void testGPU_FusionCodeGen() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  std::vector<IterDomain*> dom;
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int()));
  dom.push_back(new IterDomain(new Int(), ParallelType::Serial, true));

  TensorDomain* td = new TensorDomain(dom);
  TensorView* tv0 = new TensorView(td, DataType::Float);
  new BinaryOp(BinaryOpType::Add, tv0, new Float(0.0), new Float(1.0));
  TensorView* tv1 = static_cast<TensorView*>(add(tv0, new Float(2.0)));
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(3.0)));

  //[I0, I1, I2]
  tv2 = tv2->split(0, 4);
  //[I0o, I0i{4}, I1, I2]
  tv2 = tv2->merge(1);
  //[I0o, I0i{4}*I1, I2]
  tv2 = tv2->split(-1, 2);
  //[I0o, I0i{4}*I1, I2o, I2i{2}]
  tv2 = tv2->reorder( {{0, 1}, {1, 0}, {3, 2}} );
  //[I0i{4}*I1, I0o, I2i{2}, I2o]
  fusion.addOutput(tv2);

  tv0->computeAt(tv2, 1);
  
  //std::cout<<fusion<<std::endl;

  std::cout
  << "Code gen-ing:\n"
  << "%TV0[ I0i{4} * I1, I0o, I2, R3] compute_at( %TV1, 1 ) = 0f + 1f\n"
  << "%TV1[ I0i{4} * I1, I0o, I2] compute_at( %TV2, 1 ) = %TV0 + 2f\n"
  << "%TV2[ I0i{4} * I1, I0o, I2i{2}, I2o]              = %TV1 + 3f\n"
  << ":::::::" << std::endl;

  std::stringstream ref;
  ref << "__global__ void kernel(Tensor<float> T2){\n";
  ref << "  float T0[( ( ( 1 * ( ceilDiv(T2.size[0], 4) ) ) * T2.size[2] ) * i3 )];\n";
  ref << "  for( size_t i27 = 0; i27 < ( 4 * T2.size[1] ); ++i27 ) {\n";
  ref << "    for( size_t i29 = 0; i29 < ( ceilDiv(T2.size[0], 4) ); ++i29 ) {\n";
  ref << "      for( size_t i31 = 0; i31 < T2.size[2]; ++i31 ) {\n";
  ref << "        for( size_t i33 = 0; i33 < i3; ++i33 ) {\n";
  ref << "          if( ( ( ( i29 * 4 ) + ( i27 / T2.size[1] ) ) < T2.size[0] ) && ( ( i27 % T2.size[1] ) < T2.size[1] ) ) {\n";
  ref << "            T0[i29 * T2.size[2] * i3 + i31 * i3 + i33]\n";
  ref << "              = float(0)\n";
  ref << "              + float(1);\n";
  ref << "          }\n";
  ref << "        }\n";
  ref << "      }\n";
  ref << "    }\n";
  ref << "    float T1[( ( 1 * ( ceilDiv(T2.size[0], 4) ) ) * T2.size[2] )];\n";
  ref << "    for( size_t i54 = 0; i54 < ( ceilDiv(T2.size[0], 4) ); ++i54 ) {\n";
  ref << "      for( size_t i56 = 0; i56 < T2.size[2]; ++i56 ) {\n";
  ref << "        if( ( ( ( i54 * 4 ) + ( i27 / T2.size[1] ) ) < T2.size[0] ) && ( ( i27 % T2.size[1] ) < T2.size[1] ) ) {\n";
  ref << "          T1[i54 * T2.size[2] + i56]\n";
  ref << "            = T0[i54 * T2.size[2] + i56]\n";
  ref << "            + float(2);\n";
  ref << "        }\n";
  ref << "      }\n";
  ref << "    }\n";
  ref << "    for( size_t i81 = 0; i81 < ( ceilDiv(T2.size[0], 4) ); ++i81 ) {\n";
  ref << "      for( size_t i83 = 0; i83 < 2; ++i83 ) {\n";
  ref << "        for( size_t i85 = 0; i85 < ( ceilDiv(T2.size[2], 2) ); ++i85 ) {\n";
  ref << "          if( ( ( ( i81 * 4 ) + ( i27 / T2.size[1] ) ) < T2.size[0] ) && ( ( i27 % T2.size[1] ) < T2.size[1] ) && ( ( ( i85 * 2 ) + i83 ) < T2.size[2] ) ) {\n";
  ref << "            T2[( ( i81 * 4 ) + ( i27 / T2.size[1] ) ) * T2.stride[0] + ( i27 % T2.size[1] ) * T2.stride[1] + ( ( i85 * 2 ) + i83 ) * T2.stride[2]]\n";
  ref << "              = T1[i81 * 2 * ( ceilDiv(T2.size[2], 2) ) + i83 * ( ceilDiv(T2.size[2], 2) ) + i85]\n";
  ref << "              + float(3);\n";
  ref << "          }\n";
  ref << "        }\n";
  ref << "      }\n";
  ref << "    }\n";
  ref << "  }\n";
  ref << "}\n";

  std::cout << "\nREFERENCE: ------------------------------------------------------------\n";
  std::cout << ref.str();

  //Generate the kernel, it gets sent to what ever stream you give it
  std::cout << "\nCODEGEN:   ------------------------------------------------------------\n";
  std::stringstream cdg;
  CodeWrite cw(cdg);
  cw.traverse(&fusion);
  std::cout << cdg.str();

  std::cout << "SIZE: " << ref.str().size() << " " << cdg.str().size() << "\n";
  TORCH_CHECK(ref.str().size() == cdg.str().size());
  /**** Start Debug code ****/

  for(int i = 0; i < ref.str().size(); i++) {
	if(ref.str()[i] != cdg.str()[i]) {
      std::cout << "BADCHAR: " << i << " " << ref.str().substr(i,1) << " " << cdg.str().substr(i,1) << "\n";
    }
  }

  int mismatch = ref.str().compare(cdg.str());
  if(mismatch != 0) {
	std::cout << "MISMATCH! " << mismatch << std::endl;
  }
  /**** End   Debug code ****/

  // TODO: enable when non-deterministic Tensor size usage is fixed.
  TORCH_CHECK(ref.str().compare(cdg.str()) == 0);
}

void testGPU_FusionCodeGen2() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  int nDims = 3;
  std::vector<IterDomain*> dom;
  for(int i=0; i<nDims; i++)
    dom.push_back(new IterDomain(new Int()));

  TensorView* tv0 = new TensorView(new TensorDomain(dom), DataType::Float);
  TensorView* tv1 = new TensorView(new TensorDomain(dom), DataType::Float);
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(2.0)));
  TensorView* tv3 = static_cast<TensorView*>(add(tv0, tv2));

  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv3);

  //[I0, I1, I2]
  tv3->reorder({{0, 2}, {2, 0}});
  //[I2, I1, I0]
  tv3->split(-1, 4);
  //[I2, I1, I0o, I0i{4}]
  tv3->reorder({{2, 0}, {3, 1}, {0, 3}});
  //I0o, I0i{4}, I1, I2]


  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);
  
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  

  //std::cout<<fusion<<std::endl;
  std::cout
  << "%T3[ iS{( ceilDiv(%i0, 4) )}, iS{4}, iS{%i1}, iS{%i2} ] compute_at( %T5, -1 ) = %T1 + 2f\n"
  << "%T5[ iS{( ceilDiv(%i0, 4) )}, iS{4}, iS{%i1}, iS{%i2} ] = %T0 + %T3\n"
  << "::::::::::::" << std::endl;
  
  std::stringstream ref;
  ref << "__global__ void kernel(Tensor<float> T0, Tensor<float> T1, Tensor<float> T3){\n";
  ref << "  float T2[1];\n";
  ref << "  for( size_t i12 = 0; i12 < 4; ++i12 ) {\n";
  ref << "    for( size_t i14 = 0; i14 < T0.size[1]; ++i14 ) {\n";
  ref << "      if( ( ( ( blockIdx.x * 4 ) + i12 ) < T0.size[0] ) ) {\n";
  ref << "        T2[0]\n";
  ref << "          = T1[( ( blockIdx.x * 4 ) + i12 ) * T1.stride[0] + i14 * T1.stride[1] + threadIdx.x * T1.stride[2]]\n";
  ref << "          + float(2);\n";
  ref << "      }\n";
  ref << "      if( ( ( ( blockIdx.x * 4 ) + i12 ) < T0.size[0] ) ) {\n";
  ref << "        T3[( ( blockIdx.x * 4 ) + i12 ) * T3.stride[0] + i14 * T3.stride[1] + threadIdx.x * T3.stride[2]]\n";
  ref << "          = T0[( ( blockIdx.x * 4 ) + i12 ) * T0.stride[0] + i14 * T0.stride[1] + threadIdx.x * T0.stride[2]]\n";
  ref << "          + T2[0];\n";
  ref << "      }\n";
  ref << "    }\n";
  ref << "  }\n";
  ref << "}\n";

  std::cout << "\nREFERENCE: ------------------------------------------------------------\n";
  std::cout << ref.str();

  //Generate the kernel, it gets sent to what ever stream you give it
  std::cout << "\nCODEGEN:   ------------------------------------------------------------\n";
  std::stringstream cdg;
  CodeWrite cw(cdg);
  cw.traverse(&fusion);
  std::cout << cdg.str();

  std::cout << "SIZE: " << ref.str().size() << " " << cdg.str().size() << "\n";
  TORCH_CHECK(ref.str().size() == cdg.str().size());
  /**** Start Debug code ****/
  
  for(int i = 0; i < ref.str().size(); i++) {
	if(ref.str()[i] != cdg.str()[i]) {
      std::cout << "BADCHAR: " << i << " " << ref.str().substr(i,1) << " " << cdg.str().substr(i,1) << "\n";
    }
  }

  int mismatch = ref.str().compare(cdg.str());
  if(mismatch != 0) {
	std::cout << "MISMATCH! " << mismatch << std::endl;
  }
  /**** End   Debug code ****/
  
  TORCH_CHECK(ref.str() == cdg.str()); 
  
  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(4);
  prog.block(8);

  auto options =
  at::TensorOptions()
    .dtype(at::kFloat)
    .device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({16,8,8}, options);
  at::Tensor input2 = at::randn_like(input1);;
  at::Tensor output = at::empty_like(input1);
  std::vector<at::Tensor> inputs{{input1, input2}};
  std::vector<at::Tensor> outputs{{output}};

  torch::jit::fuser::cuda::compileKernel(fusion, prog);
  torch::jit::fuser::cuda::runTestKernel(prog, inputs, outputs);
  
  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}


void testGPU_FusionSimplePWise() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  //dimensionality of the problem
  int nDims = 3; 

  //Set up symbolic sizes for the axes should be dimensionality of the problem
  std::vector<IterDomain*> dom;
  for(int i=0; i<nDims; i++)
    dom.push_back(new IterDomain(new Int()));

  //Set up your input tensor views
  TensorView* tv0 = new TensorView(new TensorDomain(dom), DataType::Float);
  TensorView* tv1 = new TensorView(new TensorDomain(dom), DataType::Float);

  //Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  //Do math with it, it returns a `Val*` but can be static_casted back to TensorView
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(2.0)));
  TensorView* tv3 = static_cast<TensorView*>(add(tv0, tv2));

  //Register your outputs
  fusion.addOutput(tv3);

  // Do transformations, remember, transformations are outputs to inputs
  // This doesn't have to be in this order
  tv3->merge(1);
  tv3->merge(0);
  
  // Split by n_threads
  tv3->split(-1, 128*2);
  tv3->split(-1, 128);

  //For all inputs, computeAt the output inline, temporaries should be squeezed between them
  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);

  //Parallelize TV3  
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-2)->parallelize(ParallelType::TIDy);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  

  std::cout
  << "%T3[ iS{( ceilDiv(%i0, 4) )}, iS{4}, iS{%i1}, iS{%i2} ] compute_at( %T5, 1 ) = %T1 + 2f\n"
  << "%T5[ iS{( ceilDiv(%i0, 4) )}, iS{4}, iS{%i1}, iS{%i2} ] = %T0 + %T3\n"
  << "::::::::::::" << std::endl;

  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(64);     //   1 CTA
  prog.block(128,2); // 256 Threads

  auto options =
  at::TensorOptions()
    .dtype(at::kFloat)
    .device(at::kCUDA, 0);

  at::Tensor input1 = at::randn({64,2,128}, options);
  at::Tensor input2 = at::randn_like(input1);;
  at::Tensor output = at::empty_like(input1);
  std::vector<at::Tensor> inputs{{input1, input2}};
  std::vector<at::Tensor> outputs{{output}};

  torch::jit::fuser::cuda::compileKernel(fusion, prog);
  torch::jit::fuser::cuda::runTestKernel(prog, inputs, outputs);
  
  at::Tensor tv2_ref = input2 + 2.0;
  at::Tensor output_ref = input1 + tv2_ref;

  TORCH_CHECK(output_ref.equal(output));
}

void testGPU_FusionExecKernel() {
  Fusion fusion;
  FusionGuard fg(&fusion);
  //dimensionality of the problem
  int nDims = 2; 

  //Set up symbolic sizes for the axes should be dimensionality of the problem
  std::vector<IterDomain*> dom;
  for(int i=0; i<nDims; i++)
    dom.push_back(new IterDomain(new Int()));

  //Set up your input tensor views
  TensorView* tv0 = new TensorView(new TensorDomain(dom), DataType::Float);
  TensorView* tv1 = new TensorView(new TensorDomain(dom), DataType::Float);

  //Register your inputs
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  //Do math with it, it returns a `Val*` but can be static_casted back to TensorView
  TensorView* tv2 = static_cast<TensorView*>(add(tv1, new Float(2.0)));
  TensorView* tv3 = static_cast<TensorView*>(add(tv0, tv2));

  //Register your outputs
  fusion.addOutput(tv3);

  //For all inputs, computeAt the output inline, temporaries should be squeezed between them
  tv0->computeAt(tv3, -1);
  tv1->computeAt(tv3, -1);

  //Parallelize TV3  
  tv3->axis(0)->parallelize(ParallelType::BIDx);
  tv3->axis(-1)->parallelize(ParallelType::TIDx);
  
  torch::jit::fuser::cuda::CudaKernel prog;
  prog.device_ = 0;
  prog.grid(1);    // 1 CTA
  prog.block(128); // 128 Threads

  auto options =
  at::TensorOptions()
    .dtype(at::kFloat)
    .device(at::kCUDA, 0);

  at::Tensor input1 = at::ones({1,128}, options);
  at::Tensor input2 = at::ones_like(input1);;
  at::Tensor output = at::empty_like(input1);
  std::vector<at::Tensor> inputs{{input1, input2}};
  std::vector<at::Tensor> outputs{{output}};

  torch::jit::fuser::cuda::compileKernel(fusion, prog);
  torch::jit::fuser::cuda::runTestKernel(prog, inputs, outputs);
  
  at::Tensor check = at::full({1,128}, 4, options);;
  TORCH_CHECK(output.equal(check));
}

void testGPU_FusionForLoop() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const auto T0  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T1  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T2  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));

  fusion.addInput(T0);
  fusion.addInput(T1);
  fusion.addOutput(T2);

  auto ID0 = new IterDomain(new Int(8));

  auto TV2 = new TensorView(T2);
  
  BinaryOp* op = new BinaryOp(BinaryOpType::Add, TV2, T0, T1);
  ForLoop*  fl = new ForLoop(new Int(), ID0, {op});

  std::cout << fl;
}

void testGPU_FusionIfThenElse() {
  Fusion fusion;
  FusionGuard fg(&fusion);

  const auto T0  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T1  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T2  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T3  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T4  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));
  const auto T5  = new Tensor(DataType::Float, new TensorDomain({new IterDomain(new Int(16))}));

  fusion.addInput(T0);
  fusion.addInput(T1);
  fusion.addOutput(T2);
  fusion.addInput(T3);
  fusion.addInput(T4);
  fusion.addOutput(T5);

  auto TV2 = new TensorView(T2);
  auto TV5 = new TensorView(T5);
  
  BinaryOp*   op1 = new BinaryOp(BinaryOpType::Add, TV2, T0, T1);
  BinaryOp*   op2 = new BinaryOp(BinaryOpType::Add, TV5, T3, T4);
  IfThenElse* ite = new IfThenElse(new Int(0), {op1}, {op2});

  std::cout << ite;
}

void testGPU_Fusion() {}

} // namespace jit
} // namespace torch
