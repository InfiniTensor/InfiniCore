#ifndef TIME_MIXING_H
#define TIME_MIXING_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                             \
                                                                                                          \
    namespace op::time_mixing::NAMESPACE {                                                                \
    class Descriptor final : public InfiniopDescriptor {                                                  \
        TimeMixingInfo _info;                                                                             \
        size_t _workspace_size;                                                                           \
                                                                                                          \
        Descriptor(TimeMixingInfo info, size_t workspace_size, infiniDevice_t device_type, int device_id) \
            : InfiniopDescriptor{device_type, device_id}, _info(info), _workspace_size(workspace_size) {} \
                                                                                                          \
    public:                                                                                               \
        ~Descriptor();                                                                                    \
        size_t workspaceSize() const { return _workspace_size; }                                          \
        static infiniStatus_t create(                                                                     \
            infiniopHandle_t handle,                                                                      \
            Descriptor **desc_ptr,                                                                        \
            infiniopTensorDescriptor_t y_desc,                                                            \
            infiniopTensorDescriptor_t r_desc,                                                            \
            infiniopTensorDescriptor_t w_desc,                                                            \
            infiniopTensorDescriptor_t k_desc,                                                            \
            infiniopTensorDescriptor_t v_desc,                                                            \
            infiniopTensorDescriptor_t a_desc,                                                            \
            infiniopTensorDescriptor_t b_desc);                                                           \
                                                                                                          \
        infiniStatus_t calculate(                                                                         \
            void *workspace, size_t workspace_size,                                                       \
            void *y,                                                                                      \
            const void *r,                                                                                \
            const void *w,                                                                                \
            const void *k,                                                                                \
            const void *v,                                                                                \
            const void *a,                                                                                \
            const void *b,                                                                                \
            void *stream) const;                                                                          \
    };                                                                                                    \
    }

#endif // TIME_MIXING_H