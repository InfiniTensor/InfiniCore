#ifndef __CROSS_ENTROPY_LOSS_BACKWARD_H__
#define __CROSS_ENTROPY_LOSS_BACKWARD_H__

#include "../../elementwise/elementwise.h"

#define CROSS_ENTROPY_LOSS_BACKWARD_ELEMENTWISE_DESCRIPTOR(OP, NAMESPACE)     \
                                                                              \
    namespace op::OP::NAMESPACE {                                             \
    class Descriptor final : public InfiniopDescriptor {                      \
        infiniDtype_t _dtype;                                                 \
        op::elementwise::ElementwiseInfo _info;                               \
        std::unique_ptr<op::elementwise::NAMESPACE::DeviceImpl> _device_info; \
        size_t _workspace_size;                                               \
        size_t _batch_size;                                                   \
        Descriptor(                                                           \
            infiniDtype_t dtype,                                              \
            op::elementwise::ElementwiseInfo info,                            \
            op::elementwise::NAMESPACE::DeviceImpl *device_info,              \
            size_t workspace_size,                                            \
            infiniDevice_t device_type,                                       \
            int device_id,                                                    \
            size_t batch_size)                                                \
            : InfiniopDescriptor{device_type, device_id},                     \
              _dtype(dtype),                                                  \
              _info(std::move(info)),                                         \
              _device_info(std::move(device_info)),                           \
              _workspace_size(workspace_size),                                \
              _batch_size(batch_size){}                                       \
                                                                              \
    public:                                                                   \
        ~Descriptor();                                                        \
                                                                              \
        size_t workspaceSize() const { return _workspace_size; }              \
                                                                              \
        static infiniStatus_t create(                                         \
            infiniopHandle_t handle,                                          \
            Descriptor **desc_ptr,                                            \
            infiniopTensorDescriptor_t output_desc,                           \
            std::vector<infiniopTensorDescriptor_t> input_descs);             \
                                                                              \
        infiniStatus_t calculate(                                             \
            void *workspace, size_t workspace_size,                           \
            void *output,                                                     \
            std::vector<const void *> inputs,                                 \
            void *stream) const;                                              \
    };                                                                        \
    }

#endif // __CROSS_ENTROPY_LOSS_BACKWARD_H__