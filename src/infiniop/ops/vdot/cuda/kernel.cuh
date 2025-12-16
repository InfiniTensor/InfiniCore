#ifndef __VDOT_CUDA_KERNEL_CUH__
#define __VDOT_CUDA_KERNEL_CUH__

#include <cub/block/block_reduce.cuh>

namespace op::vdot::cuda {

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void vdotKernel(Tcompute *out, const Tdata *a, const Tdata *b,
                           size_t length, ptrdiff_t a_stride,
                           ptrdiff_t b_stride) {

  Tcompute dot = 0;

  // Each thread computes its partial dot product
  for (size_t i = threadIdx.x; i < length; i += BLOCK_SIZE) {
    Tcompute a_val = Tcompute(a[i * a_stride]);
    Tcompute b_val = Tcompute(b[i * b_stride]);
    dot += a_val * b_val;
  }

  // Use CUB block-level reduction
  using BlockReduce = cub::BlockReduce<Tcompute, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  Tcompute block_dot = BlockReduce(temp_storage).Sum(dot);

  // Thread 0 writes the result
  if (threadIdx.x == 0) {
    *out = block_dot;
  }
}

} // namespace op::vdot::cuda

#endif // __VDOT_CUDA_KERNEL_CUH__
