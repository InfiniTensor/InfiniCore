#ifndef RWKV5_WKV_H
#define RWKV5_WKV_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                 \
    namespace op::rwkv5_wkv::NAMESPACE {                                                      \
    class Descriptor final : public InfiniopDescriptor {                                      \
        struct Opaque;                                                                        \
        Opaque *_opaque;                                                                      \
        Rwkv5WkvInfo _info;                                                                   \
        size_t _workspace_size;                                                               \
                                                                                              \
        Descriptor(Opaque *opaque, Rwkv5WkvInfo info, size_t workspace_size,                  \
                   infiniDevice_t device_type, int device_id)                                 \
            : InfiniopDescriptor{device_type, device_id}, _opaque(opaque),                    \
              _info(info), _workspace_size(workspace_size) {}                                 \
                                                                                              \
    public:                                                                                   \
        ~Descriptor();                                                                        \
        size_t workspaceSize() const { return _workspace_size; }                              \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,          \
                                     infiniopTensorDescriptor_t out_desc,                     \
                                     infiniopTensorDescriptor_t receptance_desc,              \
                                     infiniopTensorDescriptor_t key_desc,                     \
                                     infiniopTensorDescriptor_t value_desc,                   \
                                     infiniopTensorDescriptor_t time_decay_desc,              \
                                     infiniopTensorDescriptor_t time_faaaa_desc,              \
                                     infiniopTensorDescriptor_t state_desc);                  \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *out,           \
                                 const void *receptance, const void *key, const void *value,  \
                                 const void *time_decay, const void *time_faaaa, void *state, \
                                 void *stream) const;                                         \
    };                                                                                        \
    }

#endif
