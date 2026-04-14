#include <algorithm>

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "convert_to_f32_nvidia.cuh"

namespace {

template <typename Tin>
INFINIOP_CUDA_KERNEL FastConvertToF32Kernel(size_t n, float *output, const Tin *input) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    op::convert_to_f32::cuda::ConvertToF32Op op{};
    for (; idx < n; idx += stride) {
        output[idx] = op.template operator()<float, Tin>(input[idx]);
    }
}

template <typename Tin>
infiniStatus_t launchFastConvertToF32Kernel(size_t numel,
                                            void *output,
                                            const std::vector<const void *> &inputs,
                                            void *stream) {
    if (numel == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    constexpr int block = 256;
    int grid = static_cast<int>((numel + block - 1) / block);
    grid = std::min(grid, 65535);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    FastConvertToF32Kernel<Tin><<<grid, block, 0, cuda_stream>>>(
        numel,
        reinterpret_cast<float *>(output),
        reinterpret_cast<const Tin *>(inputs[0]));
    auto err = cudaGetLastError();
    return err == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace

namespace op::convert_to_f32::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto out_dtype = out_desc->dtype();
    const auto &x_desc = input_desc_vec.at(0);
    auto in_dtype = x_desc->dtype();

    const auto &y_shape = out_desc->shape();
    const auto &x_shape = x_desc->shape();

    CHECK_DTYPE(out_dtype, INFINI_DTYPE_F32);
    CHECK_DTYPE(in_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

    CHECK_SAME_SHAPE(y_shape, x_shape);

    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, in_dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    bool fast_path = _info.isOutputContiguous();
    if (fast_path) {
        fast_path = _info.getInputContiguous()[0] && !_info.getInputBroadcasted()[0];
    }

    if (fast_path) {
        size_t numel = _info.getOutputSize();
        switch (_dtype) {
        case INFINI_DTYPE_F16:
            return launchFastConvertToF32Kernel<half>(numel, output, inputs, stream);
        case INFINI_DTYPE_BF16:
            return launchFastConvertToF32Kernel<cuda_bfloat16>(numel, output, inputs, stream);
        case INFINI_DTYPE_F32:
            return launchFastConvertToF32Kernel<float>(numel, output, inputs, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::ConvertToF32Op, float, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::ConvertToF32Op, float, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::ConvertToF32Op, float, float>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::convert_to_f32::nvidia
