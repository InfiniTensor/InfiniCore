#include "../../operator.h" // 必须包含：定义了 InfiniopDescriptor 基类
#include "../../handle.h"
#include "infiniop/ops/affine_grid.h"

// --- 后端实现头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/affine_grid_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/affine_grid_nvidia.cuh"
#endif

// 其他后端暂省略
// #ifdef ENABLE_METAX_API
// #include "metax/affine_grid_metax.h"
// #endif

__C infiniStatus_t infiniopCreateAffineGridDescriptor(
    infiniopHandle_t handle,
    infiniopAffineGridDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    uint8_t align_corners) { 

#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::affine_grid::NAMESPACE::Descriptor::create(                       \
            handle,                                                                  \
            reinterpret_cast<op::affine_grid::NAMESPACE::Descriptor **>(desc_ptr),   \
            output_desc,                                                             \
            input_desc,                                                              \
            align_corners) 

    switch (handle->device) { // handle->device 是可以直接访问的，不需要转换

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
/*
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
*/
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetAffineGridWorkspaceSize(infiniopAffineGridDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                              \
    case CASE:                                                                            \
        *size = reinterpret_cast<op::affine_grid::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    // 【修正】必须强制转换为 InfiniopDescriptor* 才能访问 device_type
    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
/*
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
*/
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopAffineGrid(
    infiniopAffineGridDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input, 
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                        \
        return reinterpret_cast<const op::affine_grid::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, stream)

    // 【修正】必须强制转换为 InfiniopDescriptor* 才能访问 device_type
    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
/*
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
*/

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyAffineGridDescriptor(infiniopAffineGridDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                             \
        delete reinterpret_cast<const op::affine_grid::NAMESPACE::Descriptor *>(desc);     \
        return INFINI_STATUS_SUCCESS

    // 【修正】必须强制转换为 InfiniopDescriptor* 才能访问 device_type
    switch (reinterpret_cast<InfiniopDescriptor *>(desc)->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
/*
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif
*/

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}