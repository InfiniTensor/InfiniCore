#ifndef DEEPSEEK_V4_SWA_DECODE_H
#define DEEPSEEK_V4_SWA_DECODE_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE) \
    namespace op::deepseek_v4_swa_decode::NAMESPACE { \
    class Descriptor final : public InfiniopDescriptor { \
        DeepseekV4SwaDecodeInfo _info; \
        size_t _workspace_size; \
        Descriptor(DeepseekV4SwaDecodeInfo info, size_t workspace_size, infiniDevice_t device_type, int device_id) : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size) {} \
    public: \
        size_t workspaceSize() const { return _workspace_size; } \
        const DeepseekV4SwaDecodeInfo &info() const { return _info; } \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t attn_sink_desc, float softmax_scale); \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *y, const void *q, const void *k, const void *attn_sink, void *stream) const; \
    }; \
    }

#endif
