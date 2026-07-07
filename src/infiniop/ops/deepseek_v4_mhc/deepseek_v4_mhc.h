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

#endif
