#ifndef __TIMESTEP_EMBEDDING_H__
#define __TIMESTEP_EMBEDDING_H__

#include "../../../utils.h"
#include "../../operator.h"

#define TIMESTEP_EMBEDDING_DESCRIPTOR(NAMESPACE)          \
                                                          \
    namespace op::timestep_embedding::NAMESPACE {         \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
        size_t _num_timesteps;                            \
        size_t _embedding_dim;                            \
        infiniDtype_t _input_dtype;                       \
                                                          \
        Descriptor(                                       \
            size_t num_timesteps,                         \
            size_t embedding_dim,                         \
            infiniDtype_t input_dtype,                    \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _num_timesteps(num_timesteps),              \
              _embedding_dim(embedding_dim),              \
              _input_dtype(input_dtype) {}                \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t output_desc,       \
            infiniopTensorDescriptor_t timestep_desc);    \
                                                          \
        infiniStatus_t calculate(                         \
            void *output,                                 \
            const void *timestep,                         \
            float max_period,                             \
            void *stream) const;                          \
    };                                                    \
    }

#endif
