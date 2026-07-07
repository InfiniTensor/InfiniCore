#ifndef __DEEPSEEK_V4_INDEXER_H__
#define __DEEPSEEK_V4_INDEXER_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                        \
    namespace op::deepseek_v4_indexer::NAMESPACE {                                  \
    class Descriptor final : public InfiniopDescriptor {                            \
        struct Opaque;                                                              \
        Opaque *_opaque;                                                            \
        DeepseekV4IndexerInfo _info;                                                \
        size_t _workspace_size;                                                     \
        Descriptor(Opaque *opaque, DeepseekV4IndexerInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}       \
                                                                                    \
    public:                                                                         \
        ~Descriptor();                                                              \
        size_t workspaceSize() const { return _workspace_size; }                    \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, \
                                     infiniopTensorDescriptor_t indices_desc,       \
                                     infiniopTensorDescriptor_t q_desc,             \
                                     infiniopTensorDescriptor_t weights_desc,       \
                                     infiniopTensorDescriptor_t compressed_desc,    \
                                     infiniopTensorDescriptor_t positions_desc,     \
                                     size_t query_start,                            \
                                     size_t compress_ratio);                        \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *indices, \
                                 const void *q, const void *weights, const void *compressed, \
                                 const void *positions, void *stream) const;        \
    };                                                                              \
    }

#endif
