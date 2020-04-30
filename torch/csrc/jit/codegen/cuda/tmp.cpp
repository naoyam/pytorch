== == == == ==
    T2[iS{(ceilDiv((ceilDiv((((i9 * i11) * i13) * i15), 128)), 4))},
       iU{4},
       ithreadIdx.x{128}] compute_at(T3, 1) =
    T1[iS{(ceilDiv((ceilDiv((((i9 * i11) * i13) * i15), 128)), 4))},
       iU{4},
       ithreadIdx.x{128}] compute_at(T2, 1) *
    float(0.979361);

T3[iS{(ceilDiv((ceilDiv((((i1 * i3) * i5) * i7), 128)), 4))},
   iU{4},
   ithreadIdx.x{128}] =
    T2[iS{(ceilDiv((ceilDiv((((i9 * i11) * i13) * i15), 128)), 4))},
       iU{4},
       ithreadIdx.x{128}] compute_at(T3, 1) *
    T0[iS{(ceilDiv((ceilDiv((((i1 * i3) * i5) * i7), 128)), 4))},
       iU{4},
       ithreadIdx.x{128}] compute_at(T3, 1);

== == == == == __device__ int ceilDiv(const int a, const int b) {
  return (a + b - 1) / b;
}

__global__ void CUDAGeneratedKernel(
    Tensor<float, 4> T0,
    Tensor<float, 4> T1,
    Tensor<float, 4> T3) {
  for (size_t i193 = 0; i193 <
       (ceilDiv(
           (ceilDiv(
               (((T3.size[0] * T3.size[1]) * T3.size[2]) * T3.size[3]), 128)),
           4));
       ++i193) {
    float T2[4];
    if ((((((((((((i193 * 4) + (4 - 1)) * 128) + threadIdx.x) / T1.size[3]) /
              T1.size[2]) /
             T1.size[1]) < T1.size[0]) &&
           ((((((((i193 * 4) + (4 - 1)) * 128) + threadIdx.x) / T1.size[3]) /
              T1.size[2]) %
             T1.size[1]) < T1.size[1])) &&
          (((((((i193 * 4) + (4 - 1)) * 128) + threadIdx.x) / T1.size[3]) %
            T1.size[2]) < T1.size[2])) &&
         ((((((i193 * 4) + (4 - 1)) * 128) + threadIdx.x) % T1.size[3]) <
          T1.size[3]))) {
      for (size_t i194 = 0; i194 < 4; ++i194) {
        T2[i194] =
            T1[((((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) /
                  T1.size[2]) /
                 T1.size[1]) *
                T1.stride[0]) +
               ((((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) /
                  T1.size[2]) %
                 T1.size[1]) *
                T1.stride[1]) +
               (((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) %
                 T1.size[2]) *
                T1.stride[2]) +
               ((((((i193 * 4) + i194) * 128) + threadIdx.x) % T1.size[3]) *
                T1.stride[3])] *
            float(0.979361);
      }
    } else {
      for (size_t i194 = 0; i194 < 4; ++i194) {
        if ((((((((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) /
                  T1.size[2]) /
                 T1.size[1]) < T1.size[0]) &&
               ((((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) /
                  T1.size[2]) %
                 T1.size[1]) < T1.size[1])) &&
              (((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) %
                T1.size[2]) < T1.size[2])) &&
             ((((((i193 * 4) + i194) * 128) + threadIdx.x) % T1.size[3]) <
              T1.size[3]))) {
          T2[i194] =
              T1[((((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) /
                    T1.size[2]) /
                   T1.size[1]) *
                  T1.stride[0]) +
                 ((((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) /
                    T1.size[2]) %
                   T1.size[1]) *
                  T1.stride[1]) +
                 (((((((i193 * 4) + i194) * 128) + threadIdx.x) / T1.size[3]) %
                   T1.size[2]) *
                  T1.stride[2]) +
                 ((((((i193 * 4) + i194) * 128) + threadIdx.x) % T1.size[3]) *
                  T1.stride[3])] *
              float(0.979361);
        }
      }
    }
    if ((((((((((((i193 * 4) + (4 - 1)) * 128) + threadIdx.x) / T3.size[3]) /
              T3.size[2]) /
             T3.size[1]) < T3.size[0]) &&
           ((((((((i193 * 4) + (4 - 1)) * 128) + threadIdx.x) / T3.size[3]) /
              T3.size[2]) %
             T3.size[1]) < T3.size[1])) &&
          (((((((i193 * 4) + (4 - 1)) * 128) + threadIdx.x) / T3.size[3]) %
            T3.size[2]) < T3.size[2])) &&
         ((((((i193 * 4) + (4 - 1)) * 128) + threadIdx.x) % T3.size[3]) <
          T3.size[3]))) {
      for (size_t i195 = 0; i195 < 4; ++i195) {
        T3[((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
              T3.size[2]) /
             T3.size[1]) *
            T3.stride[0]) +
           ((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
              T3.size[2]) %
             T3.size[1]) *
            T3.stride[1]) +
           (((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) %
             T3.size[2]) *
            T3.stride[2]) +
           ((((((i193 * 4) + i195) * 128) + threadIdx.x) % T3.size[3]) *
            T3.stride[3])] = T2[i195] *
            T0[((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
                  T3.size[2]) /
                 T3.size[1]) *
                T0.stride[0]) +
               ((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
                  T3.size[2]) %
                 T3.size[1]) *
                T0.stride[1]) +
               (((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) %
                 T3.size[2]) *
                T0.stride[2]) +
               ((((((i193 * 4) + i195) * 128) + threadIdx.x) % T3.size[3]) *
                T0.stride[3])];
      }
    } else {
      for (size_t i195 = 0; i195 < 4; ++i195) {
        if ((((((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
                  T3.size[2]) /
                 T3.size[1]) < T3.size[0]) &&
               ((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
                  T3.size[2]) %
                 T3.size[1]) < T3.size[1])) &&
              (((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) %
                T3.size[2]) < T3.size[2])) &&
             ((((((i193 * 4) + i195) * 128) + threadIdx.x) % T3.size[3]) <
              T3.size[3]))) {
          T3[((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
                T3.size[2]) /
               T3.size[1]) *
              T3.stride[0]) +
             ((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
                T3.size[2]) %
               T3.size[1]) *
              T3.stride[1]) +
             (((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) %
               T3.size[2]) *
              T3.stride[2]) +
             ((((((i193 * 4) + i195) * 128) + threadIdx.x) % T3.size[3]) *
              T3.stride[3])] = T2[i195] *
              T0[((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
                    T3.size[2]) /
                   T3.size[1]) *
                  T0.stride[0]) +
                 ((((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) /
                    T3.size[2]) %
                   T3.size[1]) *
                  T0.stride[1]) +
                 (((((((i193 * 4) + i195) * 128) + threadIdx.x) / T3.size[3]) %
                   T3.size[2]) *
                  T0.stride[2]) +
                 ((((((i193 * 4) + i195) * 128) + threadIdx.x) % T3.size[3]) *
                  T0.stride[3])];
        }
      }
    }
  }
}
