#ifndef __REDUCE_MEAN_METAX_H__
#define __REDUCE_MEAN_METAX_H__

#include "../../../operator.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include <vector>

namespace op::reduce_mean::metax {

struct ReduceMeanInfo {
    size_t ndim;
    size_t reduce_dim;
    infiniDtype_t dtype;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    size_t input_size;
    size_t output_size;
};

class Descriptor final : public InfiniopDescriptor {
    ReduceMeanInfo _info;
    size_t _workspace_size;

    Descriptor(infiniDevice_t device, int device_id, ReduceMeanInfo info, size_t workspace_size)
        : InfiniopDescriptor{device, device_id}, _info(std::move(info)), _workspace_size(workspace_size) {}

public:
    ~Descriptor() = default;

    size_t workspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        size_t dim);

    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *output,
        const void *input,
        void *stream) const;

private:
    template <typename T>
    infiniStatus_t reduceMeanMetax(
        void *output_data,
        const void *input_data,
        void *stream) const;
};

} // namespace op::reduce_mean::metax

#endif // __REDUCE_MEAN_METAX_H__