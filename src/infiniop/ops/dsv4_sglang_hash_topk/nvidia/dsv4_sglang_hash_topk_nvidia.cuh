#ifndef DSV4_SGLANG_HASH_TOPK_NVIDIA_CUH
#define DSV4_SGLANG_HASH_TOPK_NVIDIA_CUH

#include "../dsv4_sglang_hash_topk.h"

namespace op::dsv4_sglang_hash_topk::nvidia {

class Descriptor final : public op::dsv4_sglang_hash_topk::Descriptor {
public:
    Descriptor(Info info, size_t workspace_size, infiniDevice_t device_type, int device_id)
        : op::dsv4_sglang_hash_topk::Descriptor(info, workspace_size, device_type, device_id) {}

    static infiniStatus_t create(infiniopHandle_t handle,
                                 Descriptor **desc_ptr,
                                 infiniopTensorDescriptor_t router_logits_desc,
                                 infiniopTensorDescriptor_t input_ids_desc,
                                 infiniopTensorDescriptor_t tid2eid_desc,
                                 infiniopTensorDescriptor_t topk_weights_desc,
                                 infiniopTensorDescriptor_t topk_ids_desc,
                                 float routed_scaling_factor);

    infiniStatus_t calculate(void *workspace,
                             size_t workspace_size,
                             const void *router_logits,
                             const void *input_ids,
                             const void *tid2eid,
                             void *topk_weights,
                             void *topk_ids,
                             void *stream) const;
};

} // namespace op::dsv4_sglang_hash_topk::nvidia

#endif
