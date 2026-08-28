#ifndef __FP8_BLOCKWISE_GEMM_H__
#define __FP8_BLOCKWISE_GEMM_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                 \
    namespace op::fp8_blockwise_gemm::NAMESPACE {                             \
    class Descriptor final : public InfiniopDescriptor {                      \
        struct Opaque;                                                        \
        Opaque *_opaque;                                                      \
        Fp8BlockwiseGemmInfo _info;                                           \
                                                                              \
        Descriptor(Opaque *opaque, Fp8BlockwiseGemmInfo info,                 \
                   infiniDevice_t device_type, int device_id)                 \
            : InfiniopDescriptor{device_type, device_id},                     \
              _opaque(opaque), _info(info) {}                                 \
                                                                              \
    public:                                                                   \
        ~Descriptor();                                                        \
        size_t workspaceSize() const { return 0; }                            \
                                                                              \
        static infiniStatus_t create(                                         \
            infiniopHandle_t handle, Descriptor **desc_ptr,                   \
            infiniopTensorDescriptor_t out_desc,                              \
            infiniopTensorDescriptor_t a_desc,                                \
            infiniopTensorDescriptor_t q_desc,                                \
            infiniopTensorDescriptor_t scales_desc);                          \
                                                                              \
        infiniStatus_t calculate(                                             \
            void *workspace, size_t workspace_size, void *out,                \
            const void *a, const void *q, const void *scales,                 \
            void *stream) const;                                              \
    };                                                                        \
    }

#endif
