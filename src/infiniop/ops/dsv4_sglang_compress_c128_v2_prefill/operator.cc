#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dsv4_sglang_compress_c128_v2_prefill.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/dsv4_sglang_compress_c128_v2_prefill_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateDsv4SglangCompressC128V2PrefillDescriptor(infiniopHandle_t handle, infiniopDsv4SglangCompressC128V2PrefillDescriptor_t *desc_ptr, infiniopTensorDescriptor_t kv_buffer_desc, infiniopTensorDescriptor_t kv_input_desc, infiniopTensorDescriptor_t kv_output_desc, infiniopTensorDescriptor_t ape_desc, infiniopTensorDescriptor_t plan_c_desc, infiniopTensorDescriptor_t plan_w_desc) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::dsv4_sglang_compress_c128_v2_prefill::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::dsv4_sglang_compress_c128_v2_prefill::NAMESPACE::Descriptor **>(desc_ptr), kv_buffer_desc, kv_input_desc, kv_output_desc, ape_desc, plan_c_desc, plan_w_desc)
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
__INFINI_C infiniStatus_t infiniopGetDsv4SglangCompressC128V2PrefillWorkspaceSize(infiniopDsv4SglangCompressC128V2PrefillDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                                \
    case CASE:                                                                                                              \
        *size = reinterpret_cast<op::dsv4_sglang_compress_c128_v2_prefill::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
__INFINI_C infiniStatus_t infiniopDsv4SglangCompressC128V2Prefill(infiniopDsv4SglangCompressC128V2PrefillDescriptor_t desc, void *workspace, size_t workspace_size, const void *kv_buffer, const void *kv_input, void *kv_output, const void *ape, const void *plan_c, const void *plan_w, void *stream) {
#define CALCULATE(CASE, NAMESPACE) \
    case CASE:                     \
        return reinterpret_cast<op::dsv4_sglang_compress_c128_v2_prefill::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, kv_buffer, kv_input, kv_output, ape, plan_c, plan_w, stream)
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
__INFINI_C infiniStatus_t infiniopDestroyDsv4SglangCompressC128V2PrefillDescriptor(infiniopDsv4SglangCompressC128V2PrefillDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                            \
        delete reinterpret_cast<op::dsv4_sglang_compress_c128_v2_prefill::NAMESPACE::Descriptor *>(desc); \
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
