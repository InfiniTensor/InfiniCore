#ifndef __SCALED_MM_W8A8_H__
#define __SCALED_MM_W8A8_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                             \
    namespace op::scaled_mm_w8a8::NAMESPACE {             \
    class Descriptor final : public InfiniopDescriptor {  \
        size_t _m;                                        \
        size_t _n;                                        \
        size_t _k;                                        \
        infiniDtype_t _out_dtype;                         \
        bool _trans_weight;                               \
                                                          \
        Descriptor(                                       \
            size_t m,                                     \
            size_t n,                                     \
            size_t k,                                     \
            infiniDtype_t out_dtype,                      \
            bool trans_weight,                            \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _m(m),                                      \
              _n(n),                                      \
              _k(k),                                      \
              _out_dtype(out_dtype),                      \
              _trans_weight(trans_weight) {}              \
                                                          \
    public:                                               \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t out_desc,          \
            infiniopTensorDescriptor_t a_desc,            \
            infiniopTensorDescriptor_t b_desc,            \
            infiniopTensorDescriptor_t a_scales_desc,     \
            infiniopTensorDescriptor_t b_scales_desc,     \
            infiniopTensorDescriptor_t bias_desc,         \
            bool trans_weight);                           \
                                                          \
        infiniStatus_t calculate(                         \
            void *out,                                    \
            const void *a,                                \
            const void *b,                                \
            const void *a_scales,                         \
            const void *b_scales,                         \
            const void *bias,                             \
            void *stream) const;                          \
    };                                                    \
    }

#endif
