#include "sum_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::sum::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t *dim, // todo 目前 pybind 的 sum.hpp里 dim 有可能是vector，有可能是 单个的整数值！！！
    bool keepdim,
    size_t dim_size
    // std::vector<infiniopTensorDescriptor_t> input_desc_vec
    ) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SumInfo::create(output_desc, input_desc, dim, keepdim);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handel->device, handel->device_id);
    return INFINI_STATUS_SUCCESS;
}


// 在最外围计算reduce_tensor output_tensor的shape
// 假设reduce dim为 dim1 dim2 dim3... 其他dim记为 other1, other2, other3...  均为有序
// reduce_tesnor = input_tensor.permute(other1, other2, other3...dim1, dim2, dim3...)
// 计算有多少个值需要 reduce     reduce_num = shape[dim1] * shape[dim2] * shape[dim3] * ...
// 然后这个时候reduce_tensor的shape strides都变过来了
// for auto output_index  : output_size (整数)
// 计算output_index 对应的 output_offset 其实就是output_index
// 计算reduce_num个input_offset 相加 得到tempSum
// for size_t i : output_size
// convert(i * reduce_num) .... convert((i+1) * reduce_num - 1)
// indexToOffset 来进行计算 就行
template<typename T>
infiniStatus_t calculateSum(
    const SumInfo *info,
    T *output,
    const T *input,
    size_t *dim,
    bool keepdim,
    size_t dim_size,
){
    std::vector<T> temp_sum(, 0);
    if (dim_size == info->in_shape.size()){
        T tempSum = 0;
        for(size_t index = 0; index < info->input_size; index++){
            size_t index_offset = op::common_cpu::indexToOffset(index, info->input_ndim, info->input_shape.data(), info->input_strides.data());
            tempSum += input[index_offset];
        }
        output[0] = tempSum;
        return INFINI_STATUS_SUCCESS;
    }
// todo 完成对应的计算逻辑 参考Any 和 adaptive_avg_pool3d
    for (size_t i = 0; i < output_size; i++) {
        size_t output_offset = op::common_cpu::indexToOffset(i, info->out_ndim, info->out_shape.data(), info->out_strides.data());
        T tempSum = 0;
        for(size_t j = 0; j < reduce_num; j++){
            size_t input_offset = op::common_cpu::indexToOffset(j + i * reduce_num, info->in_ndim, info->in_shape.data(), info->in_strides.data());
            tempSum += input[input_offset];
        }
        output[output_offset] = tempSum;
    }
    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *input, // todo 确保合理转化
    // std::vector<const void *> inputs,
    void *stream) const {
    // todo 搞懂这里的_device_info  对应的elementwise 以及 对应的info，处理逻辑等
    switch (_info._dtype) {
    case INFINI_DTYPE_F16:
        return calculateSum<fp16_t>(_info, reinterpret_cast<fp16_t *>(input));
    case INFINI_DTYPE_F32:
        return calculateSum<float>(_info, reinterpret_cast<float *>(input));
    case INFINI_DTYPE_F64:
        return calculateSum<double>(_info, reinterpret_cast<double *>(input));
    case INFINI_DTYPE_BF16:
        return calculateSum<bf16_t>(_info, reinterpret_cast<bf16_t *>(input));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::sum::cpu
