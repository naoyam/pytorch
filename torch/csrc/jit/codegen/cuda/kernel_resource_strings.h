namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

// IO data structure for kernel code;
static auto code_template_tensor_struct = R"(
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef short int  int16_t;
typedef long long int int64_t;

template<typename T, int N>
struct Tensor {
  T& operator[](int64_t ind) {
    return data[ind];
  };

  T* data;
  int64_t size[N];
  int64_t stride[N];
};
)";

static auto code_template_block_reduction = R"(
// [Z,Y,X]_THREADS is the number of participating threads in the z, y, x
// dimension of the block. If set to 0 it means that dimension doesn't
// participate, otherwise it is the number of threads. We could start with warp
// reductions, then reduce the warps, this could save some shared memory, but
// may actually be slower.
template<int X_THREADS, int Y_THREADS, int Z_THREADS, typename T, typename Func>
__inline__ __device__
void blockReduce(const T inp_val, T& out, Func reduction_op) {

  static constexpr int X_STRIDE = (X_THREADS > 0 ? X_THREADS: 1);
  static constexpr int Y_STRIDE = (Y_THREADS > 0 ? Y_THREADS: 1);
  static constexpr int Z_STRIDE = (Z_THREADS > 0 ? Z_THREADS: 1);

  static constexpr int numel = X_STRIDE * Y_STRIDE * Z_STRIDE;

  __shared__ T shared_mem[numel];

  unsigned int reduction_size = 1;
  unsigned int linear_tid = 0;

  if(X_THREADS > 0){
    linear_tid += threadIdx.x;
    reduction_size *= X_STRIDE;
  }
  if(Y_THREADS > 0){
    linear_tid += threadIdx.y * X_STRIDE;
    reduction_size *= Y_STRIDE;
  }
  if(Z_THREADS > 0){
    linear_tid += threadIdx.z * Y_STRIDE * X_STRIDE;
    reduction_size *= Z_STRIDE;
  }

  // how many threads in inner most contig reduction, i.e. if this is >32 we can
  // do warp shuffles. We could do some template magic to make this a constexpr
  // value.
  int contig_threads = X_STRIDE;
  if(Y_THREADS > 0){
    contig_threads*=Y_THREADS;
    if(Z_THREADS>0)
      contig_threads*=Z_THREADS;
  }

  // Round contig_threads down to nearest power of 2
  contig_threads = 1 << (31 - __clz(contig_threads));
  // If greater than a warp round down to a warp
  contig_threads = contig_threads > 32 ? 32 : contig_threads;

  shared_mem[linear_tid] = inp_val;
  __syncthreads();
  // Reduce down to nearest power of 2:
  int np2 =  1 << (31 - __clz(reduction_size));

  if( linear_tid < np2 ){
    if( linear_tid + np2 < reduction_size){
      reduction_op( shared_mem[linear_tid], shared_mem[linear_tid + np2] );
    }
  }
  __syncthreads();
  for (int factor = np2/2; factor >= contig_threads; factor>>=1) {
    if (linear_tid < factor) {
      reduction_op( shared_mem[linear_tid], shared_mem[linear_tid + factor] );
    }
    __syncthreads();
  }

  unsigned int mask = 0;
  mask = ~mask; // flip all bits to 1
  mask >>= (32 - contig_threads); // Move bits right

  T val = shared_mem[linear_tid];
  if( linear_tid < contig_threads / 2){
     reduction_op(val, shared_mem[linear_tid + contig_threads / 2] );
    for (int offset = contig_threads/2; offset > 0; offset /= 2){
      reduction_op(val, __shfl_down_sync(mask, val, offset));
    }
  }

  if(linear_tid == 0)
    out = val;
}

)";

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch