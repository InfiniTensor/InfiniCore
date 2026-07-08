#ifndef __DEEPSEEK_V4_MHC_HEAD_H__
#define __DEEPSEEK_V4_MHC_HEAD_H__

#include "../../operator.h"
#include "info.h"

#define HEAD_COLLAPSE_DESCRIPTOR(NAMESPACE)                                    \
    namespace op::deepseek_v4_mhc_head::NAMESPACE {                            \
    class HeadCollapseDescriptor final : public InfiniopDescriptor {            \
        struct Opaque;                                                          \
        Opaque *_opaque;                                                        \
        DeepseekV4MHCHeadCollapseInfo _info;                                    \
        size_t _workspace_size;                                                 \
        HeadCollapseDescriptor(Opaque *opaque, DeepseekV4MHCHeadCollapseInfo info, size_t workspace_size, infiniDevice_t device, int id) \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {} \
                                                                                \
    public:                                                                     \
        ~HeadCollapseDescriptor();                                              \
        size_t workspaceSize() const { return _workspace_size; }                \
        static infiniStatus_t create(infiniopHandle_t handle, HeadCollapseDescriptor **desc_ptr, \
                                     infiniopTensorDescriptor_t y_desc,         \
                                     infiniopTensorDescriptor_t x_desc,         \
                                     infiniopTensorDescriptor_t mixes_desc,     \
                                     infiniopTensorDescriptor_t base_desc,      \
                                     infiniopTensorDescriptor_t scale_desc,     \
                                     float epsilon);                            \
        infiniStatus_t calculate(void *workspace, size_t workspace_size, void *y, \
                                 const void *x, const void *mixes, const void *base, const void *scale, void *stream) const; \
    };                                                                          \
    }

#endif
