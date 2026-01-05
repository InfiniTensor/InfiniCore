#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/hardswish.h"

// 引入各后端头文件
#ifdef ENABLE_CPU_API
#include "cpu/hardswish_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/hardswish_nvidia.cuh"
#endif
// 如果有其他后端支持，也可在此添加
// #ifdef ENABLE_METAX_API
// #include "metax/hardswish_metax.h"
// #endif


// ==================================================================
// 1. Create 函数
// ==================================================================
__C infiniStatus_t infiniopCreateHardSwishDescriptor(
    infiniopHandle_t handle,
    infiniopHardSwishDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc) {

#define CREATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        return op::hardswish::NAMESPACE::Descriptor::create(                        \
            handle,                                                                 \
            reinterpret_cast<op::hardswish::NAMESPACE::Descriptor **>(desc_ptr),    \
            output_desc,                                                            \
            {input_desc}) /* 注意：Elementwise 基类通常接受 vector<Desc>，所以这里用 {} 包裹 */

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
// 其他后端的 Case ...

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

// ==================================================================
// 2. GetWorkspaceSize 函数
// ==================================================================
__C infiniStatus_t infiniopGetHardSwishWorkspaceSize(infiniopHardSwishDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                        \
    case CASE:                                                                                      \
        *size = reinterpret_cast<op::hardswish::NAMESPACE::Descriptor *>(desc)->workspaceSize();    \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

// ==================================================================
// 3. Calculate 函数
// ==================================================================
__C infiniStatus_t infiniopHardSwish(
    infiniopHardSwishDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                          \
        return reinterpret_cast<const op::hardswish::NAMESPACE::Descriptor *>(desc)     \
            ->calculate(workspace, workspace_size, output, {input}, stream)             \
            /* 注意：同样将单个 input 指针包装成 vector/initializer_list 传入 */

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

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

// ==================================================================
// 4. Destroy 函数
// ==================================================================
__C infiniStatus_t infiniopDestroyHardSwishDescriptor(infiniopHardSwishDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                          \
        delete reinterpret_cast<const op::hardswish::NAMESPACE::Descriptor *>(desc);    \
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

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}