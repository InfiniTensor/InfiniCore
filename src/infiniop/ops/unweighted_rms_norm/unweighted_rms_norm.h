#ifndef UNWEIGHTED_RMS_NORM_H
#define UNWEIGHTED_RMS_NORM_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                             \
                                                                          \
    namespace op::unweighted_rms_norm::NAMESPACE {                        \
    class Descriptor final : public InfiniopDescriptor {                  \
        struct Opaque;                                                    \
        Opaque *_opaque;                                                  \
        UnweightedRMSNormInfo _info;                                      \
        size_t _workspace_size;                                           \
                                                                          \
        Descriptor(                                                       \
            Opaque *opaque,                                               \
            UnweightedRMSNormInfo info,                                   \
            size_t workspace_size,                                        \
            infiniDevice_t device_type,                                   \
            int device_id)                                                \
            : InfiniopDescriptor{device_type, device_id},                 \
              _opaque(opaque),                                            \
              _info(info),                                                \
              _workspace_size(workspace_size) {}                          \
                                                                          \
    public:                                                               \
        ~Descriptor();                                                    \
                                                                          \
        size_t workspaceSize() const { return _workspace_size; }          \
                                                                          \
        static infiniStatus_t create(                                     \
            infiniopHandle_t handle,                                      \
            Descriptor **desc_ptr,                                        \
            infiniopTensorDescriptor_t y_desc,                            \
            infiniopTensorDescriptor_t x_desc,                            \
            float epsilon);                                               \
                                                                          \
        infiniStatus_t calculate(                                         \
            void *workspace, size_t workspace_size,                       \
            void *y,                                                      \
            const void *x,                                                \
            void *stream) const;                                          \
    };                                                                    \
    }

#endif // UNWEIGHTED_RMS_NORM_H
