#ifndef __LINEAR_H__
#define __LINEAR_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                        \
                                                                                                                     \
    namespace op::linear::NAMESPACE {                                                                                \
    class Descriptor final : public InfiniopDescriptor {                                                             \
        struct Opaque;                                                                                               \
        Opaque *_opaque;                                                                                             \
        LinearInfo _info;                                                                                            \
        size_t _workspace_size;                                                                                      \
                                                                                                                     \
        Descriptor(Opaque *opaque, LinearInfo info,                                                                  \
                   size_t workspace_size,                                                                            \
                   infiniDevice_t device_type, int device_id)                                                        \
            : InfiniopDescriptor{device_type, device_id},                                                            \
              _opaque(opaque), _info(info), _workspace_size(workspace_size) {}                                       \
                                                                                                                     \
    public:                                                                                                          \
        ~Descriptor();                                                                                               \
                                                                                                                     \
        size_t minWorkspaceSize() const { return _workspace_size; }                                                  \
                                                                                                                     \
        static infiniStatus_t create(                                                                                \
            infiniopHandle_t handle, Descriptor **desc_ptr,                                                          \
            infiniopTensorDescriptor_t d_desc,                                                                       \
            infiniopTensorDescriptor_t c_desc,                                                                       \
            infiniopTensorDescriptor_t bias_desc,                                                                    \
            infiniopTensorDescriptor_t x_desc,                                                                       \
            infiniopTensorDescriptor_t x_scale_desc,                                                                 \
            infiniopTensorDescriptor_t x_zero_desc,                                                                  \
            infiniopTensorDescriptor_t weights_desc,                                                                 \
            infiniopTensorDescriptor_t weights_scale_desc,                                                           \
            infiniopTensorDescriptor_t weights_zero_desc,                                                            \
            float alpha,                                                                                             \
            float beta);                                                                                             \
        template <unsigned int BLOCK_SIZE, typename Tdata>                                                           \
        infiniStatus_t launchKernel(const LinearInfo &info, Tdata *y,                                                \
                                    const Tdata *c, const Tdata *bias, const int8_t *x_packed,                       \
                                    const Tdata *x_scale, const Tdata *x_zero, const int8_t *w_packed,               \
                                    const Tdata *w_scale, const Tdata *w_zero, void *stream, void *workspace) const; \
                                                                                                                     \
        infiniStatus_t calculate(                                                                                    \
            void *workspace, size_t workspace_size,                                                                  \
            void *d, const void *c, const void *bias, const void *x,                                                 \
            const void *x_scale, const void *x_zero, const void *weights,                                            \
            const void *weights_scale, const void *weights_zero, void *stream) const;                                \
    };                                                                                                               \
    }

#endif // __LINEAR_H__