#ifndef __DEEPSEEK_V4_COMPRESSOR_H__
#define __DEEPSEEK_V4_COMPRESSOR_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                        \
    namespace op::deepseek_v4_compressor::NAMESPACE {                               \
    class Descriptor final : public InfiniopDescriptor {                            \
        struct Opaque;                                                              \
        Opaque *_opaque;                                                            \
        DeepseekV4CompressorInfo _info;                                             \
        size_t _workspace_size;                                                     \
        Descriptor(Opaque *opaque, DeepseekV4CompressorInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}       \
                                                                                    \
    public:                                                                         \
        ~Descriptor();                                                              \
        size_t workspaceSize() const { return _workspace_size; }                    \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, \
                                     infiniopTensorDescriptor_t out_desc,           \
                                     infiniopTensorDescriptor_t kv_desc,            \
                                     infiniopTensorDescriptor_t score_desc,         \
                                     infiniopTensorDescriptor_t ape_desc,           \
                                     infiniopTensorDescriptor_t norm_weight_desc,   \
                                     size_t compress_ratio,                         \
                                     float epsilon);                                \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *out, \
                                 const void *kv, const void *score, const void *ape, \
                                 const void *norm_weight, void *stream) const;      \
    };                                                                              \
    }

#endif
