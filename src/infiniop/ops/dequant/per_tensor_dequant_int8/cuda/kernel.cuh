#ifndef __PER_TENSOR_DEQUANT_INT8_KERNEL_CUH__
#define __PER_TENSOR_DEQUANT_INT8_KERNEL_CUH__

template <typename Tin, typename Tout>
__device__ void perTensorDequantI8SymKernel(
    Tout *x, const Tin *x_packed, const float *x_scale,
    int num_elements) {

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int grid_size = blockDim.x * gridDim.x;
    float x_scale_val = x_scale[0];
    for (int i = gid; i < num_elements; i += grid_size) {
        float val = static_cast<float>(x_packed[i]) * x_scale_val;
        x[i] = static_cast<Tout>(val);
    }
}

#endif // __PER_TENSOR_DEQUANT_INT8_KERNEL_CUH__
