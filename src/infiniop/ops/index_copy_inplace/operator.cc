#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/index_copy_inplace.h"

#ifdef ENABLE_CPU_API
#include "cpu/index_copy_inplace_cpu.h"//待创建
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/index_copy_inplace_nvidia.cuh"//待创建
#endif
#ifdef ENABLE_METAX_API
#include "metax/index_copy_inplace_metax.h"//待创建
#endif

__C infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(
    infiniopHandle_t handle,
    infiniopIndexCopyInplaceDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int dim,
    infiniopTensorDescriptor_t index_desc) {

#define CREATE(CASE, NAMESPACE)                                            \
    case CASE:                                                             \
        /*op::index_copy_inplace::NAMESPACE::Descriptor 需要在平台头文件中定义类*/ \
        return op::index_copy_inplace::NAMESPACE::Descriptor::create(                     \
            handle,                                                        \
            reinterpret_cast<op::index_copy_inplace::NAMESPACE::Descriptor **>(desc_ptr), \
            input_desc, /*模仿rope对应文件的写法，参数扁平化直接传递，这里不模仿add了*/                                                       \
            output_desc,                                                               \
            dim,                                                       \
            index_desc)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);// 假设天数也复用 NVIDIA 的实现
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

// __C infiniStatus_t infiniopGetAddWorkspaceSize(infiniopAddDescriptor_t desc, size_t *size) {

// #define GET(CASE, NAMESPACE)                                                               \
//     case CASE:                                                                             \
//         *size = reinterpret_cast<op::add::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
//         return INFINI_STATUS_SUCCESS

//     switch (desc->device_type) {
// #ifdef ENABLE_CPU_API
//         GET(INFINI_DEVICE_CPU, cpu);
// #endif
// #ifdef ENABLE_NVIDIA_API
//         GET(INFINI_DEVICE_NVIDIA, nvidia);
// #endif
// #ifdef ENABLE_ILUVATAR_API
//         GET(INFINI_DEVICE_ILUVATAR, nvidia);
// #endif
// #ifdef ENABLE_METAX_API
//         GET(INFINI_DEVICE_METAX, metax);
// #endif
//     default:
//         return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
//     }
// #undef GET

//     return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
// }

__C infiniStatus_t infiniopIndexCopyInplace(
    infiniopIndexCopyInplaceDescriptor_t desc,
    //void *workspace,
    //size_t workspace_size,
    const void *input,
    void *output,
    const void *index,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                            \
    case CASE:                                                                \
        return reinterpret_cast<const op::index_copy_inplace::NAMESPACE::Descriptor *>(desc) \
            ->calculate(input, output, index, stream)/*这里不需要dim参数，因为在创建描述符CreateDescriptor时已经提供*/
            /*参数顺序需要和这里匹配*/

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyIndexCopyInplaceDescriptor(infiniopIndexCopyInplaceDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        delete reinterpret_cast<const op::index_copy_inplace::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
//----------------------------抛弃模仿 rope 的、复杂的面向对象封装----------------------------------------
// #include "infiniop/index_copy_inplace.h" // C-API 声明
// #include "../../tensor.h"
// #include "../../utils.h"
// #include <vector>

// // 引入平台特定的【头文件】
// #ifdef ENABLE_CPU_API
// #include "cpu/index_copy_inplace_cpu.h"
// #endif
// // ... 其他平台的 #ifdef ...


// // Info 结构体的【定义】放在这里，因为它是平台无关的
// struct IndexCopyInplaceInfo {
//     // Info 类的成员变量
//     infiniDtype_t data_type;
//     int dim;
//     std::vector<size_t> output_shape;
//     std::vector<ptrdiff_t> output_strides;
//     std::vector<size_t> input_shape;
//     std::vector<ptrdiff_t> input_strides;
//     size_t index_size;
//     int64_t slice_size;

//     // Info 类的 create 方法，负责所有验证
//     static utils::Result<IndexCopyInplaceInfo> create(
//         const infiniopTensorDescriptor_t input_desc,
//         const infiniopTensorDescriptor_t output_desc,
//         int dim_val,
//         const infiniopTensorDescriptor_t index_desc) {
        
//         CHECK_OR_RETURN(
//             input_desc != nullptr && output_desc != nullptr && index_desc != nullptr,
//             INFINI_STATUS_NULL_POINTER);

//         const infiniDtype_t dtype = output_desc->dtype();

//         CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16, INFINI_DTYPE_F64);
//         CHECK_OR_RETURN(dtype == input_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
//         CHECK_OR_RETURN(index_desc->dtype() == INFINI_DTYPE_I64, INFINI_STATUS_BAD_TENSOR_DTYPE);

//         CHECK_OR_RETURN(output_desc->ndim() == input_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
//         CHECK_OR_RETURN(index_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
        
//         if (output_desc->ndim() == 0) {
//             CHECK_OR_RETURN(dim_val == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
//         } else {
//             CHECK_OR_RETURN(dim_val >= 0 && static_cast<size_t>(dim_val) < output_desc->ndim(), INFINI_STATUS_BAD_TENSOR_SHAPE);
//         }
        
//         for (size_t i = 0; i < output_desc->ndim(); ++i) {
//             if (i != static_cast<size_t>(dim_val)) {
//                 CHECK_OR_RETURN(output_desc->dim(i) == input_desc->dim(i), INFINI_STATUS_BAD_TENSOR_SHAPE);
//             }
//         }
//         if (output_desc->ndim() > 0) {
//             CHECK_OR_RETURN(input_desc->dim(dim_val) == index_desc->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
//         }

//         int64_t current_slice_size = 1;
//         if (output_desc->ndim() > 0) {
//             for (size_t i = 0; i < output_desc->ndim(); ++i) {
//                 if (i != static_cast<size_t>(dim_val)) {
//                     current_slice_size *= output_desc->dim(i);
//                 }
//             }
//         }
        
//         return utils::Result<IndexCopyInplaceInfo>(IndexCopyInplaceInfo{
//             dtype, dim_val, output_desc->shape(), output_desc->strides(),
//             input_desc->shape(), input_desc->strides(),
//             index_desc->numel(), current_slice_size,
//         });
//     }
// };

// // C-API 的实现
// extern "C" {

// infiniStatus_t infiniopCreateIndexCopyInplaceDescriptor(
//     infiniopHandle_t handle,
//     infiniopIndexCopyInplaceDescriptor_t *desc_ptr,
//     infiniopTensorDescriptor_t input,
//     infiniopTensorDescriptor_t output,
//     int dim,
//     infiniopTensorDescriptor_t index) {
    
//     auto info_result = IndexCopyInplaceInfo::create(input, output, dim, index);
//     CHECK_RESULT(info_result);
    
//     // 我们仍然使用 Info 指针作为不透明描述符
//     auto info = new IndexCopyInplaceInfo(info_result.take());
//     // 【关键】在 Info 中保存设备类型，以便后续分发
//     // (假设 Info 结构体中增加了 infiniDevice_t device; 成员)
//     info->device = handle->device; 
//     *desc_ptr = reinterpret_cast<infiniopIndexCopyInplaceDescriptor_t>(info);
    
//     return INFINI_STATUS_SUCCESS;
// }

// infiniStatus_t infiniopIndexCopyInplace(
//     infiniopIndexCopyInplaceDescriptor_t desc,
//     const void *input, void *output, const void *index, void *stream) {
    
//     auto info = reinterpret_cast<const IndexCopyInplaceInfo *>(desc);

//     // 【关键】在这里进行平台分发
//     switch (info->device) {
//         #ifdef ENABLE_CPU_API
//         case INFINI_DEVICE_CPU:
//             // 调用 CPU 专属的内核启动函数
//             return index_copy_inplace_kernel_cpu(*info, input, output, index, stream);
//         #endif
//         // ... 其他平台的 case ...
//         default:
//             return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
//     }
// }

// infiniStatus_t infiniopDestroyIndexCopyInplaceDescriptor(
//     infiniopIndexCopyInplaceDescriptor_t desc) {
//     delete reinterpret_cast<const IndexCopyInplaceInfo *>(desc);
//     return INFINI_STATUS_SUCCESS;
// }

// } // extern "C"