#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/dsv4_add_rmsnorm_quant.h"
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/dsv4_add_rmsnorm_quant_nvidia.cuh"
#endif
__INFINI_C infiniStatus_t infiniopCreateDsv4AddRMSNormQuantDescriptor(infiniopHandle_t handle, infiniopDsv4AddRMSNormQuantDescriptor_t *desc_ptr, infiniopTensorDescriptor_t res_desc, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t weight_desc, float epsilon) {
#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return op::dsv4_add_rmsnorm_quant::NAMESPACE::Descriptor::create(handle, reinterpret_cast<op::dsv4_add_rmsnorm_quant::NAMESPACE::Descriptor **>(desc_ptr), res_desc, q_desc, scale_desc, x_desc, weight_desc, epsilon)
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
__INFINI_C infiniStatus_t infiniopGetDsv4AddRMSNormQuantWorkspaceSize(infiniopDsv4AddRMSNormQuantDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                                  \
    case CASE:                                                                                                \
        *size = reinterpret_cast<op::dsv4_add_rmsnorm_quant::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
__INFINI_C infiniStatus_t infiniopDsv4AddRMSNormQuant(infiniopDsv4AddRMSNormQuantDescriptor_t desc, void *workspace, size_t workspace_size, void *res, void *q, void *scale, const void *x, const void *weight, void *stream) {
#define CALC(CASE, NAMESPACE) \
    case CASE:                \
        return reinterpret_cast<op::dsv4_add_rmsnorm_quant::NAMESPACE::Descriptor *>(desc)->calculate(workspace, workspace_size, res, q, scale, x, weight, stream)
    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALC(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALC(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALC(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALC(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_ALI_API
        CALC(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALC
}
__INFINI_C infiniStatus_t infiniopDestroyDsv4AddRMSNormQuantDescriptor(infiniopDsv4AddRMSNormQuantDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                            \
    case CASE:                                                                              \
        delete reinterpret_cast<op::dsv4_add_rmsnorm_quant::NAMESPACE::Descriptor *>(desc); \
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
