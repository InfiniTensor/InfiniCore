#ifndef DSV4_FUSED_ROPE_H
#define DSV4_FUSED_ROPE_H
#include "../../operator.h"
#include "info.h"
#define DESCRIPTOR(NAMESPACE)                                                                                                                                                                                                                                \
    namespace op::dsv4_fused_rope::NAMESPACE {                                                                                                                                                                                                               \
    class Descriptor final : public InfiniopDescriptor {                                                                                                                                                                                                     \
        Info _info;                                                                                                                                                                                                                                          \
        size_t _workspace_size;                                                                                                                                                                                                                              \
        Descriptor(Info info, size_t workspace_size, infiniDevice_t device_type, int device_id) : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size) {}                                                                \
                                                                                                                                                                                                                                                             \
    public:                                                                                                                                                                                                                                                  \
        size_t workspaceSize() const { return _workspace_size; }                                                                                                                                                                                             \
        const Info &info() const { return _info; }                                                                                                                                                                                                           \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t k_desc, infiniopTensorDescriptor_t freq_real_desc, infiniopTensorDescriptor_t freq_imag_desc, int has_k); \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q, void *k, const void *freq_real, const void *freq_imag, void *stream) const;                                                                                                \
    };                                                                                                                                                                                                                                                       \
    }
#endif
