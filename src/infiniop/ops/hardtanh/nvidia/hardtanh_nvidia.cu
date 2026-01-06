#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "hardtanh_nvidia.cuh"

namespace op::hardtanh::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec,
    float min_val,   // 新增参数
    float max_val) { // 新增参数

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &input_desc = input_desc_vec.at(0);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    CHECK_SAME_SHAPE(output_shape, input_shape);

    // 调用宏创建 CUDA 逐元素描述符
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    // 将 HardTanh 特有的参数存入 Descriptor 实例
    auto desc = *desc_ptr;
    desc->min_val = min_val;
    desc->max_val = max_val;

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

    // 构造带参数的 CUDA Op 实例
    // 注意：这里的 Op 实例会被拷贝到 GPU 常量区或通过参数传递给 Kernel
    cuda::HardTanhOp op(this->min_val, this->max_val);

    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::HardTanhOp, cuda_bfloat16>(_info, workspace, output, inputs, stream, op);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::HardTanhOp, half>(_info, workspace, output, inputs, stream, op);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::HardTanhOp, float>(_info, workspace, output, inputs, stream, op);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::HardTanhOp, double>(_info, workspace, output, inputs, stream, op);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::hardtanh::nvidia