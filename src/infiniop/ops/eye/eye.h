#ifndef __EYE_H__
#define __EYE_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "info.h"

#define EYE_DESCRIPTOR(NAMESPACE)                                               \
    namespace op::eye::NAMESPACE {                                              \
    class Descriptor final : public InfiniopDescriptor {                        \
        op::eye::EyeInfo _info;                                                 \
                                                                                \
        Descriptor(op::eye::EyeInfo info, infiniDevice_t device_type, int device_id) \
            : InfiniopDescriptor{device_type, device_id}, _info(info) {}        \
                                                                                \
    public:                                                                     \
        ~Descriptor() = default;                                                \
                                                                                \
        size_t workspaceSize() const { return 0; }                              \
                                                                                \
        static infiniStatus_t create(                                          \
            infiniopHandle_t handle,                                            \
            Descriptor **desc_ptr,                                               \
            infiniopTensorDescriptor_t y_desc);                                  \
                                                                                \
        infiniStatus_t calculate(                                               \
            void *workspace, size_t workspace_size,                             \
            void *y, void *stream) const;                                      \
    };                                                                          \
    }

#endif
