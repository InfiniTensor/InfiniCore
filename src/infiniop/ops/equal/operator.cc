#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/equal.h"

// 引入各后端头文件
#ifdef ENABLE_CPU_API
#include "cpu/equal_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/equal_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/equal_metax.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/equal_kunlun.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/equal_bang.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/equal_moore.h"
#endif

// =======================================================================
// 1. Create 函数实现
// =======================================================================
__C infiniStatus_t infiniopCreateEqualDescriptor(
    infiniopHandle_t handle,
    infiniopEqualDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc, // Output
    infiniopTensorDescriptor_t a_desc, // Input A
    infiniopTensorDescriptor_t b_desc) // Input B
{

#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::equal::NAMESPACE::Descriptor::create(                              \
            handle,                                                                   \
            reinterpret_cast<op::equal::NAMESPACE::Descriptor **>(desc_ptr),          \
            c_desc,                                                                   \
            {a_desc, b_desc}) /* 注意：这里将两个输入打包成 vector 传入 */

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
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

// =======================================================================
// 2. GetWorkspaceSize 函数实现
// =======================================================================
__C infiniStatus_t infiniopGetEqualWorkspaceSize(infiniopEqualDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                   \
    case CASE:                                                                                 \
        *size = reinterpret_cast<op::equal::NAMESPACE::Descriptor *>(desc)->workspaceSize();   \
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
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

// =======================================================================
// 3. Execute (Calculate) 函数实现
// =======================================================================
__C infiniStatus_t infiniopEqual(
    infiniopEqualDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,           // Output data
    const void *a,     // Input A data
    const void *b,     // Input B data
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                        \
        return reinterpret_cast<const op::equal::NAMESPACE::Descriptor *>(desc)       \
            ->calculate(workspace, workspace_size, c, {a, b}, stream)

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
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

// =======================================================================
// 4. Destroy 函数实现
// =======================================================================
__C infiniStatus_t
infiniopDestroyEqualDescriptor(infiniopEqualDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                          \
        delete reinterpret_cast<const op::equal::NAMESPACE::Descriptor *>(desc);        \
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
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        DELETE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}