#include "../../../operator.h"
#include "../flash_attention.h"

namespace op::flash_attention::ascend {
class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;
    FlashAttentionInfo _info;
    size_t _min_workspace_size;

    Descriptor(
        FlashAttentionInfo info,
        size_t workspace_size_,
        Opaque *opaque,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _opaque(opaque),
          _info(info),
          _min_workspace_size(workspace_size_) {}

public:
    ~Descriptor();
    size_t workspaceSize() const { return _min_workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t k_desc,
        infiniopTensorDescriptor_t v_desc,
        infiniopTensorDescriptor_t mask);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *out,
        const void *q,
        const void *k,
        const void *v,
        void *mask,
        void *stream) const;
};
} // namespace op::flash_attention::ascend