#ifndef __MUL_SCALAR_H__
#define __MUL_SCALAR_H__

#include "../../operator.h"
#include "info.h"

#include <memory>

#define MUL_SCALAR_DESCRIPTOR(NAMESPACE)                                      \
                                                                              \
    namespace op::mul_scalar::NAMESPACE {                                     \
    class Descriptor final : public InfiniopDescriptor {                      \
        MulScalarInfo _info;                                                  \
        std::unique_ptr<op::elementwise::NAMESPACE::DeviceImpl> _device_info; \
        size_t _workspace_size;                                               \
                                                                              \
        Descriptor(                                                           \
            MulScalarInfo info,                                               \
            op::elementwise::NAMESPACE::DeviceImpl *device_info,              \
            size_t workspace_size_,                                           \
            infiniDevice_t device_type,                                       \
            int device_id)                                                    \
            : InfiniopDescriptor{device_type, device_id},                     \
              _info(std::move(info)),                                         \
              _device_info(std::move(device_info)),                           \
              _workspace_size(workspace_size_) {}                             \
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
            infiniopTensorDescriptor_t input_desc);                           \
                                                                              \
        infiniStatus_t calculate(                                             \
            void *workspace,                                                  \
            size_t workspace_size,                                            \
            void *output,                                                     \
            const void *input,                                                \
            double alpha,                                                     \
            void *stream) const;                                              \
    };                                                                        \
    }

#endif // __MUL_SCALAR_H__
