#ifndef __I8GEMM_H__
#define __I8GEMM_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                    \
                                                                                 \
    namespace op::i8gemm::NAMESPACE {                                            \
    class Descriptor final : public InfiniopDescriptor {                         \
        struct Opaque;                                                           \
        Opaque *_opaque;                                                         \
        size_t _workspace_size;                                                  \
        I8GemmInfo _info;                                                        \
        infiniDtype_t _out_dtype;                                                \
                                                                                 \
        Descriptor(Opaque *opaque, I8GemmInfo info,                              \
                   size_t workspace_size,                                        \
                   infiniDtype_t out_dtype,                                      \
                   infiniDevice_t device_type, int device_id)                    \
            : InfiniopDescriptor{device_type, device_id}, _out_dtype(out_dtype), \
              _opaque(opaque), _info(info), _workspace_size(workspace_size) {}   \
                                                                                 \
    public:                                                                      \
        ~Descriptor();                                                           \
                                                                                 \
        size_t minWorkspaceSize() const { return _workspace_size; }              \
                                                                                 \
        static infiniStatus_t create(                                            \
            infiniopHandle_t handle, Descriptor **desc_ptr,                      \
            infiniopTensorDescriptor_t out_desc,                                 \
            infiniopTensorDescriptor_t bias_desc,                                \
            infiniopTensorDescriptor_t a_desc,                                   \
            infiniopTensorDescriptor_t a_scale_desc,                             \
            infiniopTensorDescriptor_t b_desc,                                   \
            infiniopTensorDescriptor_t b_scale_desc);                            \
                                                                                 \
        infiniStatus_t calculate(                                                \
            void *workspace, size_t workspace_size,                              \
            void *out, const void *bias, const void *a,                          \
            const void *a_scale, const void *b,                                  \
            const void *b_scale, void *stream) const;                            \
    };                                                                           \
    }

#endif // __I8GEMM_H__
