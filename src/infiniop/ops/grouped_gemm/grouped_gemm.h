#ifndef __GROUPED_GEMM_H__
#define __GROUPED_GEMM_H__

#include "../../operator.h"
#include "info.h"

// See `ops/gemm/gemm.h` for the rationale of the PImpl-style DESCRIPTOR macro.
#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::grouped_gemm::NAMESPACE {                      \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        GroupedGemmInfo _info;                                   \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            GroupedGemmInfo info,                                \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t c_desc,                   \
            infiniopTensorDescriptor_t a_desc,                   \
            infiniopTensorDescriptor_t b_desc,                   \
            infiniopTensorDescriptor_t group_sizes_desc);        \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *c,                                             \
            const void *a,                                       \
            const void *b,                                       \
            const void *group_sizes,                             \
            const void *group_sizes_host,                        \
            float alpha,                                         \
            float beta,                                          \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __GROUPED_GEMM_H__
