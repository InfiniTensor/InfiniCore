#ifndef __FP8_INDEXER_QUANT_H__
#define __FP8_INDEXER_QUANT_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                             \
    namespace op::fp8_indexer_quant::NAMESPACE {          \
    class Descriptor final : public InfiniopDescriptor {  \
        size_t _num_groups;                               \
        size_t _head_dim;                                 \
        size_t _threads;                                  \
        infiniDtype_t _input_dtype;                       \
                                                          \
        Descriptor(                                       \
            size_t num_groups,                            \
            size_t head_dim,                              \
            size_t threads,                               \
            infiniDtype_t input_dtype,                    \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _num_groups(num_groups),                    \
              _head_dim(head_dim),                        \
              _threads(threads),                          \
              _input_dtype(input_dtype) {}                \
                                                          \
    public:                                               \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t q_fp8_desc,        \
            infiniopTensorDescriptor_t weights_fp32_desc, \
            infiniopTensorDescriptor_t q_desc,            \
            infiniopTensorDescriptor_t weights_desc);     \
                                                          \
        infiniStatus_t calculate(                         \
            void *q_fp8,                                  \
            void *weights_fp32,                           \
            const void *q,                                \
            const void *weights,                          \
            void *stream) const;                          \
    };                                                    \
    }

#endif
