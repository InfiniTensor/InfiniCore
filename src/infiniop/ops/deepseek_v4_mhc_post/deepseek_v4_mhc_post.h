#ifndef __DEEPSEEK_V4_MHC_POST_H__
#define __DEEPSEEK_V4_MHC_POST_H__

#include "../../operator.h"
#include "info.h"

#define MHC_POST_DESCRIPTOR(NAMESPACE)                                         \
    namespace op::deepseek_v4_mhc_post::NAMESPACE {                            \
    class PostDescriptor final : public InfiniopDescriptor {                    \
        struct Opaque;                                                          \
        Opaque *_opaque;                                                        \
        DeepseekV4MHCPostInfo _info;                                            \
        size_t _workspace_size;                                                 \
        PostDescriptor(Opaque *opaque, DeepseekV4MHCPostInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {} \
                                                                                \
    public:                                                                     \
        ~PostDescriptor();                                                      \
        size_t workspaceSize() const { return _workspace_size; }                \
        static infiniStatus_t create(infiniopHandle_t handle, PostDescriptor **desc_ptr, \
                                     infiniopTensorDescriptor_t y_desc,         \
                                     infiniopTensorDescriptor_t new_x_desc,     \
                                     infiniopTensorDescriptor_t residual_desc,  \
                                     infiniopTensorDescriptor_t post_desc,      \
                                     infiniopTensorDescriptor_t comb_desc);     \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *y, \
                                 const void *new_x, const void *residual, const void *post, const void *comb, void *stream) const; \
    };                                                                          \
    }

#endif
