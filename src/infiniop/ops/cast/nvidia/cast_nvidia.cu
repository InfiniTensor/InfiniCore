#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "cast_nvidia.cuh"

namespace op::cast::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto output_dtype = out_desc->dtype();
    auto input_dtype = input_desc_vec.at(0)->dtype();

    const auto &out_shape = out_desc->shape();
    const auto &in_shape = input_desc_vec.at(0)->shape();

    CHECK_SAME_SHAPE(out_shape, in_shape);
    CHECK_DTYPE(output_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);
    CHECK_DTYPE(input_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    // ✅ 使用 ElementwiseInfo::create 而不是构造函数
    auto info_result = op::elementwise::ElementwiseInfo::create(out_desc, input_desc_vec);
    CHECK_RESULT(info_result);  // 检查是否创建成功
    
    auto info = info_result.take();                                                           
    auto workspace_size = info.getMetaMemSize() + info.getInputSize() * sizeof(void *);       
                                                                                              
    auto device_impl_result = op::elementwise::nvidia::DeviceImpl::create(handle->internal()); 
    CHECK_RESULT(device_impl_result);                                                         
                                                                                              
    *desc_ptr = new Descriptor(                                                               
        output_dtype,
        input_dtype,                                                                                 
        std::move(info),                                                                      
        std::move(device_impl_result.take()),                                                 
        workspace_size,                                                                       
        handle->device,                                                                       
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // 简化类型映射宏
    #define DISPATCH_CAST(SRC, DST) \
        return _device_info->calculate<256, cuda::CastOp, DST, SRC>(_info, workspace, output, inputs, stream);

    // dispatch by _output_dtype (目标类型)
    switch (_output_dtype) {
    case INFINI_DTYPE_F16:
        switch (_input_dtype) {
        case INFINI_DTYPE_F32:     DISPATCH_CAST(float, half);
        case INFINI_DTYPE_F64:     DISPATCH_CAST(double, half);
        case INFINI_DTYPE_I32:   DISPATCH_CAST(int32_t, half);
        case INFINI_DTYPE_I64:   DISPATCH_CAST(int64_t, half);
        case INFINI_DTYPE_U32:  DISPATCH_CAST(uint32_t, half);
        case INFINI_DTYPE_U64:  DISPATCH_CAST(uint64_t, half);
        case INFINI_DTYPE_F16:     DISPATCH_CAST(half, half);
        default: break;
        }
        break;

    case INFINI_DTYPE_F32:
        switch (_input_dtype) {
        case INFINI_DTYPE_F16:     DISPATCH_CAST(half, float);
        case INFINI_DTYPE_F64:     DISPATCH_CAST(double, float);
        case INFINI_DTYPE_I32:   DISPATCH_CAST(int32_t, float);
        case INFINI_DTYPE_I64:   DISPATCH_CAST(int64_t, float);
        case INFINI_DTYPE_U32:  DISPATCH_CAST(uint32_t, float);
        case INFINI_DTYPE_U64:  DISPATCH_CAST(uint64_t, float);
        case INFINI_DTYPE_F32:     DISPATCH_CAST(float, float);
        default: break;
        }
        break;

    case INFINI_DTYPE_F64:
        switch (_input_dtype) {
        case INFINI_DTYPE_F16:     DISPATCH_CAST(half, double);
        case INFINI_DTYPE_F32:     DISPATCH_CAST(float, double);
        case INFINI_DTYPE_I32:   DISPATCH_CAST(int32_t, double);
        case INFINI_DTYPE_I64:   DISPATCH_CAST(int64_t, double);
        case INFINI_DTYPE_U32:  DISPATCH_CAST(uint32_t, double);
        case INFINI_DTYPE_U64:  DISPATCH_CAST(uint64_t, double);
        case INFINI_DTYPE_F64:     DISPATCH_CAST(double, double);
        default: break;
        }
        break;

    case INFINI_DTYPE_I32:
        switch (_input_dtype) {
        case INFINI_DTYPE_I64:   DISPATCH_CAST(int64_t, int32_t);
        case INFINI_DTYPE_U32:  DISPATCH_CAST(uint32_t, int32_t);
        case INFINI_DTYPE_U64:  DISPATCH_CAST(uint64_t, int32_t);
        case INFINI_DTYPE_I32:   DISPATCH_CAST(int32_t, int32_t);
        default: break;
        }
        break;

    case INFINI_DTYPE_I64:
        switch (_input_dtype) {
        case INFINI_DTYPE_I32:   DISPATCH_CAST(int32_t, int64_t);
        case INFINI_DTYPE_U32:  DISPATCH_CAST(uint32_t, int64_t);
        case INFINI_DTYPE_U64:  DISPATCH_CAST(uint64_t, int64_t);
        case INFINI_DTYPE_I64:   DISPATCH_CAST(int64_t, int64_t);
        default: break;
        }
        break;

    case INFINI_DTYPE_U32:
        switch (_input_dtype) {
        case INFINI_DTYPE_I32:   DISPATCH_CAST(int32_t, uint32_t);
        case INFINI_DTYPE_I64:   DISPATCH_CAST(int64_t, uint32_t);
        case INFINI_DTYPE_U64:  DISPATCH_CAST(uint64_t, uint32_t);
        case INFINI_DTYPE_U32:  DISPATCH_CAST(uint32_t, uint32_t);
        default: break;
        }
        break;

    case INFINI_DTYPE_U64:
        switch (_input_dtype) {
        case INFINI_DTYPE_I32:   DISPATCH_CAST(int32_t, uint64_t);
        case INFINI_DTYPE_I64:   DISPATCH_CAST(int64_t, uint64_t);
        case INFINI_DTYPE_U32:  DISPATCH_CAST(uint32_t, uint64_t);
        case INFINI_DTYPE_U64:  DISPATCH_CAST(uint64_t, uint64_t);
        default: break;
        }
        break;

    default:
        break;
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::cast::nvidia