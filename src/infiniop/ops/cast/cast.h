#ifndef __CAST_H__
#define __CAST_H__

#include "../../operator.h"

#define DESCRIPTOR(NAMESPACE)                                    \
namespace op::cast::NAMESPACE {                                  \
class Descriptor final : public InfiniopDescriptor {             \
    infiniDtype_t _output_dtype;                                 \
    infiniDtype_t _input_dtype;                                  \
    op::elementwise::ElementwiseInfo _info;                      \
    std::unique_ptr<op::elementwise::NAMESPACE::DeviceImpl> _device_info; \
    size_t _workspace_size;                                      \
    Descriptor(                                                  \
        infiniDtype_t _out_dtype,                                \
        infiniDtype_t _input_dtype,                              \
        op::elementwise::ElementwiseInfo info,                   \
        op::elementwise::NAMESPACE::DeviceImpl *device_info,           \
        size_t workspace_size_,                                  \
        infiniDevice_t device_type,                              \
        int device_id)                                           \
        : InfiniopDescriptor{device_type, device_id},            \
          _output_dtype(_out_dtype),                             \
          _input_dtype(_input_dtype),                            \
          _info(std::move(info)),                                \
          _device_info(std::move(device_info)),                  \
          _workspace_size(workspace_size_) {}                    \
public:                                                          \
    ~Descriptor();                                               \
    size_t workspaceSize() const { return _workspace_size; }     \
    static infiniStatus_t create(                                \
        infiniopHandle_t handle,                                 \
        Descriptor **desc_ptr,                                   \
        infiniopTensorDescriptor_t output_desc,                  \
        std::vector<infiniopTensorDescriptor_t> input_descs);    \
    infiniStatus_t calculate(                                    \
        void *workspace, size_t workspace_size,                  \
        void *output,                                            \
        std::vector<const void *> inputs,                        \
        void *stream) const;                                     \
};                                                               \
}


#endif 

