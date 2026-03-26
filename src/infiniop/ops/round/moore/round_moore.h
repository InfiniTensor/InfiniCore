#ifndef __ROUND_MOORE_API_H__
#define __ROUND_MOORE_API_H__

#include "../../../elementwise/moore/elementwise_moore_api.h"

namespace op::round::moore {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    op::elementwise::ElementwiseInfo _info;
    std::unique_ptr<op::elementwise::moore::DeviceImpl> _device_info;
    size_t _workspace_size;
    int _decimals;

    Descriptor(infiniDtype_t dtype,
               op::elementwise::ElementwiseInfo info,
               op::elementwise::moore::DeviceImpl *device_info,
               size_t workspace_size,
               infiniDevice_t device_type,
               int device_id,
               int decimals);

public:
    ~Descriptor();

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t out_desc,
        std::vector<infiniopTensorDescriptor_t> input_desc_vec,
        int decimals);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *output,
        std::vector<const void *> inputs,
        void *stream) const;
};

} // namespace op::round::moore

#endif // __ROUND_MOORE_API_H__