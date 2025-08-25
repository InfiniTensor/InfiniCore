#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "crossentropyloss_backward_nvidia.cuh"

namespace op::CrossEntropyLossBackWard::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // create CUDA elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto shape = _info.getAllInputShapes();
    auto dim = _info.getNdim();
    size_t N = std::accumulate(shape, shape + dim - 1, 1ull, std::multiplies<size_t>());
    float fN = static_cast<float>(N);

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackWardOp, half>(_info, workspace, output, inputs, stream, std::move(fN));
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackWardOp, cuda_bfloat16>(_info, workspace, output, inputs, stream, std::move(fN));
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackWardOp, float>(_info, workspace, output, inputs, stream, std::move(fN));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::CrossEntropyLossBackWard::nvidia
