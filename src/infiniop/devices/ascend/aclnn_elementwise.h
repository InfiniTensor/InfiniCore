#ifndef __INFINIOP_ACLNN_ELEMENTWISE_H__
#define __INFINIOP_ACLNN_ELEMENTWISE_H__

#include "../../operator.h"
#include "../../tensor.h"

#include <vector>

#define ACLNN_ELEMENTWISE_DESCRIPTOR(OP)                           \
    namespace op::OP::ascend {                                     \
    class Descriptor final : public InfiniopDescriptor {           \
        struct Opaque;                                             \
        Opaque *_opaque;                                           \
        size_t _workspace_size;                                    \
                                                                   \
        Descriptor(Opaque *opaque, size_t workspace_size,          \
                   infiniDevice_t device_type, int device_id)      \
            : InfiniopDescriptor{device_type, device_id},          \
              _opaque(opaque), _workspace_size(workspace_size) {}  \
                                                                   \
    public:                                                        \
        ~Descriptor();                                             \
        size_t workspaceSize() const { return _workspace_size; }   \
        static infiniStatus_t create(                              \
            infiniopHandle_t handle, Descriptor **desc_ptr,        \
            infiniopTensorDescriptor_t output_desc,                \
            std::vector<infiniopTensorDescriptor_t> input_descs);  \
        infiniStatus_t calculate(                                  \
            void *workspace, size_t workspace_size, void *output,  \
            std::vector<const void *> inputs, void *stream) const; \
    };                                                             \
    }

#endif // __INFINIOP_ACLNN_ELEMENTWISE_H__
