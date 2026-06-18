#ifndef __MOE_SUM_H__
#define __MOE_SUM_H__

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                                                  \
                                                                                                               \
    namespace op::moe_sum::NAMESPACE {                                                                         \
    class Descriptor final : public InfiniopDescriptor {                                                       \
        struct Opaque;                                                                                         \
        Opaque *_opaque;                                                                                       \
        MoeSumInfo _info;                                                                                      \
        size_t _workspace_size;                                                                                \
                                                                                                               \
        Descriptor(Opaque *opaque, MoeSumInfo info, size_t workspace_size, infiniDevice_t device, int id)      \
            : InfiniopDescriptor{device, id}, _opaque(opaque), _info(info), _workspace_size(workspace_size) {} \
                                                                                                               \
    public:                                                                                                    \
        ~Descriptor();                                                                                         \
                                                                                                               \
        size_t workspaceSize() const { return _workspace_size; }                                               \
                                                                                                               \
        static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,                           \
                                     infiniopTensorDescriptor_t output_desc,                                   \
                                     infiniopTensorDescriptor_t input_desc);                                   \
                                                                                                               \
        infiniStatus_t calculate(void *workspace, size_t workspace_size,                                       \
                                 void *output, const void *input, void *stream) const;                         \
    };                                                                                                         \
    }

#endif
