#ifndef __W4A8_MOE_SHUFFLE_H__
#define __W4A8_MOE_SHUFFLE_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::w4a8_moe_shuffle::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {                 \
        W4A8MoeShuffleInfo _info;                                        \
                                                                         \
        Descriptor(W4A8MoeShuffleInfo info, infiniDevice_t device_type,  \
                   int device_id)                                        \
            : InfiniopDescriptor{device_type, device_id}, _info(info) {} \
                                                                         \
    public:                                                              \
        static infiniStatus_t create(                                    \
            infiniopHandle_t handle, Descriptor **desc_ptr,              \
            infiniopTensorDescriptor_t output_desc,                      \
            infiniopTensorDescriptor_t input_desc);                      \
                                                                         \
        infiniStatus_t calculate(void *output, const void *input,        \
                                 void *stream) const;                    \
    };                                                                   \
    }

#endif
