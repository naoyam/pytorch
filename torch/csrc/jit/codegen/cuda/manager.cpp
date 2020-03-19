#include <torch/csrc/jit/codegen/cuda/manager.h>
#include <torch/csrc/jit/codegen/cuda/parser.h>
#include <torch/csrc/jit/codegen/cuda/kernel.h>

#include <torch/csrc/jit/codegen/cuda/tensor.h>
#include <torch/csrc/jit/codegen/cuda/fusion.h>
#include <torch/csrc/jit/codegen/cuda/utils.h>
#include <torch/csrc/jit/passes/canonicalize.h>
#include <torch/csrc/jit/passes/shape_analysis.h>

#include <unordered_map>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

namespace {

// CudaFusionManager holds compiled `CudaKernel` and handles all interfacing
// including compilation and execution.
//
// We cache two maps here:
//   a. string of graph -> kernel_id
//   b. kernel_id -> CudaKernel
//
// This allows CudaKernel reuse across nodes;
class CudaFusionManager {
public:
  static CudaFusionManager& getManager() {
    static CudaFusionManager cuda_fusion_manager_;
    return cuda_fusion_manager_;
  };

  // TODO: I'm assuming we have stride information in `graph->toString`
  //       We need to make sure stride information is in the final string, as we
  //       want to AVOID kernel reuse between different fusion_node, unless they
  //       have identical contiguity information! (So identical stride + shape
  //       is even more restricting in a good way)
  int32_t registerOrGetCacheId(std::shared_ptr<Graph>& graph) {
    std::lock_guard<std::mutex> guard(mutex_);
    // prepare graph for lowering;
    Canonicalize(graph, false);
    //EraseShapeInformation(graph);
    auto repr = graph->toString(false);

    // create new graph_cache_ entry;
    if (graph_cache_.count(repr) == 0) {
      int32_t kernel_id = getNextUniqueID();

      graph_cache_[repr] = kernel_id;

      Fusion fusion;
      // lower torch::jit::Graph to torch::jit::fuser::cuda::fusion
      parseJitIR(graph, fusion);

      // default constructor via accessing empty key;
      compileKernel(fusion, kernel_cache_[kernel_id]);

      return kernel_id;
    } else {
      return graph_cache_[repr];
    }
  };

  void runFusionNode(
      int32_t kernel_id,
      const at::ArrayRef<IValue> inputs,
      std::vector<at::Tensor> outputs) {
    TORCH_CHECK(kernel_cache_.count(kernel_id) != 0, "kernel id not recognized");

    CudaKernel& cuda_kernel_entry = kernel_cache_[kernel_id];

    runKernel(cuda_kernel_entry, inputs, outputs);
  }

protected:

  std::mutex mutex_;

  void runCudaKernel(
    int32_t key,
    const std::vector<int>& contiguity_tag,
    const c10::Device) {
  };

  int32_t getNextUniqueID() {
    return next_unique_id_++;
  };

  std::unordered_map<std::string, int32_t> graph_cache_;
  std::unordered_map<int64_t, CudaKernel> kernel_cache_;

private:

  int32_t next_unique_id_ = 0;
};


} // namespace

void compileCudaFusionGroup(Node* fusion_node) {
  TORCH_CHECK(
      fusion_node->kind() == prim::CudaFusionGroup,
      "Only prim::CudaFusionGroup can be compiled");
  if (fusion_node->hasAttribute(attr::cache_id)) {
    TORCH_WARN("Double registration of CudaFusionGroup on CudaFusionManager");
  }
  int32_t fusion_cache_id = 
      CudaFusionManager::getManager().registerOrGetCacheId(fusion_node->g(attr::Subgraph));
  fusion_node->i_(attr::cache_id, fusion_cache_id);
}

void runCudaFusionGroup(const Node* const fusion_node, Stack& stack) {
  TORCH_CHECK(
      fusion_node->kind() == prim::CudaFusionGroup,
      "prim::CudaFusionGroup expected");
  // TODO: should we support runtime compilation with updated dynamic shape;
  //       shape inference would be needed so we can allocate output;
  TORCH_CHECK(
      fusion_node->hasAttribute(attr::cache_id),
      "node prim::CudaFusionGroup has not been compiled yet");
  int32_t kernel_id = fusion_node->i(attr::cache_id);

  // Currently we just construct I/O tensors for static graph;
  std::shared_ptr<Graph> graph = fusion_node->g(attr::Subgraph);
  const auto nInputs = graph->inputs().size();
  at::ArrayRef<IValue> inputs = last(stack, nInputs);

  // we need to construct outputs;
  std::vector<at::Tensor> outputs;

  bool matched_static_inputs = true;
  for (int i = 0; i < nInputs; i++) {
    auto& static_input = graph->inputs()[i];
    auto& dynamic_input = inputs[i]; // this is FILO stack
    if ((*dynamic_input.type()) != (*static_input->type())) {
      matched_static_inputs  = false;
      break;
    }
    if (dynamic_input.isTensor()) {
      // TODO: we could relax on a bunch of checks here, like strides & gradient
      at::Tensor inp_tensor = dynamic_input.toTensor();
      if (!static_input->type()->cast<TensorType>()
          ->isCompatibleWithInCurrentExecutionContext(inp_tensor)) {
        matched_static_inputs  = false;
        break;
      }
    }
  }

  // TODO: expose the API to populate shape inference. This allows separate CI
  // tests
  // matched_static_inputs = false;
  if (!matched_static_inputs) {
    // update shape information per the new inputs;
    // shape inference done through PyTorch JIT shape propagation;
    EraseShapeInformation(graph);
    for (int i = 0; i < nInputs; i++) {
      graph->inputs()[i]->setType(inputs[i].type());
    }
    // shape inference
    PropagateInputShapes(graph);
  }

  for (const auto* const output : graph->outputs()) {
    auto type = output->type()->expect<TensorType>();
    
    // Expect output to be tensor;
    // we don't use type->isComplete() because strides technically should not be 
    // populated during shape inference;
    TORCH_CHECK(
        type && type->device() && type->sizes().isComplete() && type->scalarType(),
        "Complete TensorType for output is expected.");

    const auto device = *(type->device());
    const auto scalar_type = *(type->scalarType());

    auto options = at::TensorOptions()
      .dtype(scalar_type)
      .layout(at::kStrided)
      .device(device)
      .requires_grad(type->requires_grad());

    // TODO: We should infer output shape from `inputs`
    const auto sizes = extractSizes(type);
    const auto strides= extractStrides(type);

    auto tensor = at::empty_strided(sizes, strides, options);
    outputs.push_back(tensor);
  }

  CudaFusionManager::getManager().runFusionNode(kernel_id, inputs, outputs);

  drop(stack, inputs.size());
  stack.insert(stack.end(),
      std::make_move_iterator(outputs.begin()),
      std::make_move_iterator(outputs.end()));
}

}}}} // namespace torch::jit::fuser::cuda
