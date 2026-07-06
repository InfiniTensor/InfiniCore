#ifndef __DEEPSEEK_V4_ROUTER_H__
#define __DEEPSEEK_V4_ROUTER_H__

#include "../../operator.h"
#include "info.h"

#define TOPK_DESCRIPTOR(NAMESPACE)                                                                                   \
    namespace op::deepseek_v4_router::NAMESPACE {                                                                    \
    class TopkRouterDescriptor final : public InfiniopDescriptor {                                                    \
        struct Opaque;                                                                                               \
        Opaque *_opaque;                                                                                             \
        DeepseekV4TopkRouterInfo _info;                                                                              \
        size_t _workspace_size;                                                                                      \
        TopkRouterDescriptor(Opaque *opaque, DeepseekV4TopkRouterInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}       \
                                                                                                                     \
    public:                                                                                                          \
        ~TopkRouterDescriptor();                                                                                     \
        size_t workspaceSize() const { return _workspace_size; }                                                     \
        static infiniStatus_t create(infiniopHandle_t handle, TopkRouterDescriptor **desc_ptr,                       \
                                     infiniopTensorDescriptor_t topk_weights_desc,                                   \
                                     infiniopTensorDescriptor_t topk_indices_desc,                                   \
                                     infiniopTensorDescriptor_t logits_desc,                                         \
                                     infiniopTensorDescriptor_t bias_desc,                                           \
                                     bool renormalize);                                                              \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *topk_weights, void *topk_indices,     \
                                 const void *logits, const void *bias, void *stream) const;                         \
    };                                                                                                               \
    }

#define HASH_DESCRIPTOR(NAMESPACE)                                                                                   \
    namespace op::deepseek_v4_router::NAMESPACE {                                                                    \
    class HashRouterDescriptor final : public InfiniopDescriptor {                                                    \
        struct Opaque;                                                                                               \
        Opaque *_opaque;                                                                                             \
        DeepseekV4HashRouterInfo _info;                                                                              \
        size_t _workspace_size;                                                                                      \
        HashRouterDescriptor(Opaque *opaque, DeepseekV4HashRouterInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}       \
                                                                                                                     \
    public:                                                                                                          \
        ~HashRouterDescriptor();                                                                                     \
        size_t workspaceSize() const { return _workspace_size; }                                                     \
        static infiniStatus_t create(infiniopHandle_t handle, HashRouterDescriptor **desc_ptr,                       \
                                     infiniopTensorDescriptor_t topk_weights_desc,                                   \
                                     infiniopTensorDescriptor_t topk_indices_desc,                                   \
                                     infiniopTensorDescriptor_t logits_desc,                                         \
                                     infiniopTensorDescriptor_t input_ids_desc,                                      \
                                     infiniopTensorDescriptor_t tid2eid_desc,                                        \
                                     bool renormalize);                                                              \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *topk_weights, void *topk_indices,     \
                                 const void *logits, const void *input_ids, const void *tid2eid, void *stream) const;\
    };                                                                                                               \
    }

#endif
