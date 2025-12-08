#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/hypot.h"

// --- 后端实现头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/hypot_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/hypot_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/hypot_metax.h"
#endif

extern "C" {

// =======================================================================
// 1. 创建算子描述符
// =======================================================================
__C infiniStatus_t infiniopCreateHypotDescriptor(
    infiniopHandle_t handle,
    infiniopHypotDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input_a,
    infiniopTensorDescriptor_t input_b) {

    // 【修改点】Create 接收 input_a 和 input_b，并以列表形式 {input_a, input_b} 传递给后端
    #define CREATE(CASE, NAMESPACE)                                                 \
        case CASE:                                                                  \
            return op::hypot::NAMESPACE::Descriptor::create(                        \
                handle,                                                             \
                reinterpret_cast<op::hypot::NAMESPACE::Descriptor **>(desc_ptr),    \
                output,                                                             \
                {input_a, input_b})

    switch (handle->device) {
    #ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    #ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
    #endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef CREATE
}

// =======================================================================
// 2. 获取 Workspace 大小
// =======================================================================
__C infiniStatus_t infiniopGetHypotWorkspaceSize(infiniopHypotDescriptor_t desc, size_t *size) {

    #define GET(CASE, NAMESPACE)                                                             \
        case CASE:                                                                           \
            *size = reinterpret_cast<op::hypot::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
            return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    /* 保持与参考代码一致的注释状态，如有需要请取消注释
    #ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
    #endif 
    #ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
    #endif*/
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

// =======================================================================
// 3. 执行计算 (Calculate)
// =======================================================================
__C infiniStatus_t infiniopHypot(
    infiniopHypotDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input_a,
    const void *input_b,
    void *stream) {

    // 【修改点】calculate 接收 input_a 和 input_b，并以列表形式 {input_a, input_b} 传递
    #define CALCULATE(CASE, NAMESPACE)                                          \
        case CASE:                                                              \
            return reinterpret_cast<const op::hypot::NAMESPACE::Descriptor *>(desc) \
                ->calculate(workspace, workspace_size, output, {input_a, input_b}, stream)

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    /* 保持与参考代码一致的注释状态，如有需要请取消注释
    #ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
    #endif 
    #ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
    #endif*/
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef CALCULATE
}

// =======================================================================
// 4. 销毁描述符
// =======================================================================
__C infiniStatus_t infiniopDestroyHypotDescriptor(infiniopHypotDescriptor_t desc) {

    #define DELETE(CASE, NAMESPACE)                                            \
        case CASE:                                                             \
            delete reinterpret_cast<const op::hypot::NAMESPACE::Descriptor *>(desc); \
            return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
    #ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
    #endif
    #ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
    #endif
    /* 保持与参考代码一致的注释状态，如有需要请取消注释
    #ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
    #endif
    #ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
    #endif
    #ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
    #endif 
    #ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
    #endif*/
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
    #undef DELETE
}

} // extern "C"