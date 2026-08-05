#ifndef __SELECT_LAST_TOKEN_HIDDEN_H__
#define __SELECT_LAST_TOKEN_HIDDEN_H__

#include "../../../utils.h"
#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                               \
    namespace op::select_last_token_hidden::NAMESPACE {     \
    class Descriptor final : public InfiniopDescriptor {    \
        size_t _num_requests;                               \
        size_t _total_tokens;                               \
        size_t _row_bytes;                                  \
                                                            \
        Descriptor(                                         \
            size_t num_requests,                            \
            size_t total_tokens,                            \
            size_t row_bytes,                               \
            infiniDevice_t device_type,                     \
            int device_id)                                  \
            : InfiniopDescriptor{device_type, device_id},   \
              _num_requests(num_requests),                  \
              _total_tokens(total_tokens),                  \
              _row_bytes(row_bytes) {}                      \
                                                            \
    public:                                                 \
        static infiniStatus_t create(                       \
            infiniopHandle_t handle,                        \
            Descriptor **desc_ptr,                          \
            infiniopTensorDescriptor_t output_desc,         \
            infiniopTensorDescriptor_t hidden_states_desc,  \
            infiniopTensorDescriptor_t input_offsets_desc); \
                                                            \
        infiniStatus_t calculate(                           \
            void *output,                                   \
            const void *hidden_states,                      \
            const void *input_offsets,                      \
            void *stream) const;                            \
    };                                                      \
    }

#endif
