#ifndef __DEEPSEEK_V4_MHC_H__
#define __DEEPSEEK_V4_MHC_H__

#include "../../operator.h"
#include "info.h"

#define PARAMS_DESCRIPTOR(NAMESPACE)                                                                                   \
    namespace op::deepseek_v4_mhc::NAMESPACE {                                                                         \
    class ParamsDescriptor final : public InfiniopDescriptor {                                                          \
        struct Opaque;                                                                                                  \
        Opaque *_opaque;                                                                                                \
        DeepseekV4MHCParamsInfo _info;                                                                                  \
        size_t _workspace_size;                                                                                         \
        ParamsDescriptor(Opaque *opaque, DeepseekV4MHCParamsInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}          \
                                                                                                                        \
    public:                                                                                                             \
        ~ParamsDescriptor();                                                                                            \
        size_t workspaceSize() const { return _workspace_size; }                                                        \
        static infiniStatus_t create(infiniopHandle_t handle, ParamsDescriptor **desc_ptr,                              \
                                     infiniopTensorDescriptor_t pre_desc,                                                \
                                     infiniopTensorDescriptor_t post_desc,                                               \
                                     infiniopTensorDescriptor_t comb_desc,                                               \
                                     infiniopTensorDescriptor_t mixes_desc,                                              \
                                     infiniopTensorDescriptor_t base_desc,                                               \
                                     infiniopTensorDescriptor_t scale_desc,                                              \
                                     size_t sinkhorn_iters, float epsilon);                                              \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *pre, void *post, void *comb,             \
                                 const void *mixes, const void *base, const void *scale, void *stream) const;           \
    };                                                                                                                  \
    }

#define PRE_COLLAPSE_DESCRIPTOR(NAMESPACE)                                                                                         \
    namespace op::deepseek_v4_mhc::NAMESPACE {                                                                                     \
    class PreCollapseDescriptor final : public InfiniopDescriptor {                                                                 \
        struct Opaque;                                                                                                              \
        Opaque *_opaque;                                                                                                            \
        DeepseekV4MHCPreCollapseInfo _info;                                                                                         \
        size_t _workspace_size;                                                                                                     \
        PreCollapseDescriptor(Opaque *opaque, DeepseekV4MHCPreCollapseInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}                      \
                                                                                                                                     \
    public:                                                                                                                         \
        ~PreCollapseDescriptor();                                                                                                   \
        size_t workspaceSize() const { return _workspace_size; }                                                                    \
        static infiniStatus_t create(infiniopHandle_t handle, PreCollapseDescriptor **desc_ptr,                                     \
                                     infiniopTensorDescriptor_t collapsed_desc,                                                      \
                                     infiniopTensorDescriptor_t post_desc,                                                           \
                                     infiniopTensorDescriptor_t comb_desc,                                                           \
                                     infiniopTensorDescriptor_t x_desc,                                                              \
                                     infiniopTensorDescriptor_t mixes_desc,                                                          \
                                     infiniopTensorDescriptor_t base_desc,                                                           \
                                     infiniopTensorDescriptor_t scale_desc,                                                          \
                                     size_t sinkhorn_iters, float epsilon);                                                          \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *collapsed, void *post, void *comb,                    \
                                 const void *x, const void *mixes, const void *base, const void *scale, void *stream) const;         \
    };                                                                                                                              \
    }

#define SCALE_MIXES_DESCRIPTOR(NAMESPACE)                                                                                         \
    namespace op::deepseek_v4_mhc::NAMESPACE {                                                                                     \
    class ScaleMixesDescriptor final : public InfiniopDescriptor {                                                                  \
        struct Opaque;                                                                                                              \
        Opaque *_opaque;                                                                                                            \
        DeepseekV4MHCScaleMixesInfo _info;                                                                                          \
        size_t _workspace_size;                                                                                                     \
        ScaleMixesDescriptor(Opaque *opaque, DeepseekV4MHCScaleMixesInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {}                      \
                                                                                                                                     \
    public:                                                                                                                         \
        ~ScaleMixesDescriptor();                                                                                                    \
        size_t workspaceSize() const { return _workspace_size; }                                                                    \
        static infiniStatus_t create(infiniopHandle_t handle, ScaleMixesDescriptor **desc_ptr,                                      \
                                     infiniopTensorDescriptor_t scaled_desc,                                                         \
                                     infiniopTensorDescriptor_t x_desc,                                                              \
                                     infiniopTensorDescriptor_t raw_mixes_desc,                                                      \
                                     float epsilon);                                                                                 \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *scaled,                                               \
                                 const void *x, const void *raw_mixes, void *stream) const;                                          \
    };                                                                                                                              \
    }

#endif
