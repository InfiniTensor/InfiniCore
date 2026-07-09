#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dsv4_sglang_mega_moe_pre_dispatch.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/dsv4_sglang_mega_moe_pre_dispatch_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateDsv4SglangMegaMoePreDispatchDescriptor(infiniopHandle_t handle, infiniopDsv4SglangMegaMoePreDispatchDescriptor_t *desc_ptr, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t topk_idx_desc, infiniopTensorDescriptor_t topk_weights_desc, infiniopTensorDescriptor_t buf_x_desc, infiniopTensorDescriptor_t buf_x_sf_desc, infiniopTensorDescriptor_t buf_topk_idx_desc, infiniopTensorDescriptor_t buf_topk_weights_desc) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::dsv4_sglang_mega_moe_pre_dispatch::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::dsv4_sglang_mega_moe_pre_dispatch::NAMESPACE::Descriptor **>(desc_ptr), x_desc, topk_idx_desc, topk_weights_desc, buf_x_desc, buf_x_sf_desc, buf_topk_idx_desc, buf_topk_weights_desc)
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}
__INFINI_C infiniStatus_t infiniopGetDsv4SglangMegaMoePreDispatchWorkspaceSize(infiniopDsv4SglangMegaMoePreDispatchDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                             \
    case CASE:                                                                                                           \
        *size = reinterpret_cast<op::dsv4_sglang_mega_moe_pre_dispatch::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}
__INFINI_C infiniStatus_t infiniopDsv4SglangMegaMoePreDispatch(infiniopDsv4SglangMegaMoePreDispatchDescriptor_t desc, void *workspace, size_t workspace_size, const void *x, const void *topk_idx, const void *topk_weights, void *buf_x, void *buf_x_sf, void *buf_topk_idx, void *buf_topk_weights, void *stream) {
#define CALCULATE(CASE, NAMESPACE) \
    case CASE:                     \
        return reinterpret_cast<op::dsv4_sglang_mega_moe_pre_dispatch::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, x, topk_idx, topk_weights, buf_x, buf_x_sf, buf_topk_idx, buf_topk_weights, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}
__INFINI_C infiniStatus_t infiniopDestroyDsv4SglangMegaMoePreDispatchDescriptor(infiniopDsv4SglangMegaMoePreDispatchDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                       \
    case CASE:                                                                                         \
        delete reinterpret_cast<op::dsv4_sglang_mega_moe_pre_dispatch::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
