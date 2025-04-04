#ifndef __RANDOM_SAMPLE_H__
#define __RANDOM_SAMPLE_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                             \
                                                          \
    namespace op::random_sample::NAMESPACE {              \
    class Descriptor final : public InfiniopDescriptor {  \
        struct Opaque;                                    \
        Opaque *_opaque;                                  \
                                                          \
        infiniDtype_t _dt_i, _dt_p;                       \
        size_t _n, _min_workspace_size;                   \
                                                          \
        Descriptor(                                       \
            infiniDtype_t dt_i,                           \
            infiniDtype_t dt_p,                           \
            size_t n,                                     \
            size_t min_workspace_size,                    \
            Opaque *opaque,                               \
            infiniDevice_t device_type,                   \
            int device_id)                                \
            : InfiniopDescriptor{device_type, device_id}, \
              _opaque(opaque),                            \
              _dt_i(dt_i),                                \
              _dt_p(dt_p),                                \
              _n(n),                                      \
              _min_workspace_size(min_workspace_size) {}  \
                                                          \
    public:                                               \
        ~Descriptor();                                    \
                                                          \
        static infiniStatus_t create(                     \
            infiniopHandle_t handle,                      \
            Descriptor **desc_ptr,                        \
            infiniopTensorDescriptor_t result_desc,       \
            infiniopTensorDescriptor_t probs_desc);       \
                                                          \
        size_t minWorkspaceSize() const;                  \
                                                          \
        infiniStatus_t calculate(                         \
            void *workspace,                              \
            size_t workspace_size,                        \
            void *result,                                 \
            const void *probs,                            \
            float random_val,                             \
            float topp,                                   \
            int topk,                                     \
            float temperature,                            \
            void *stream) const;                          \
    };                                                    \
    }

#endif // __RANDOM_SAMPLE_H__
