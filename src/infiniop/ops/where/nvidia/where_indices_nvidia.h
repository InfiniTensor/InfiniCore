#ifndef __WHERE_INDICES_NVIDIA_H__
#define __WHERE_INDICES_NVIDIA_H__

#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::where::nvidia {

class IndicesDescriptor final : public InfiniopDescriptor {
    size_t _numel;
    int _ndim;
    size_t *_shape;
    ptrdiff_t *_strides;
    std::shared_ptr<device::nvidia::Handle::Internal> _internal;

public:
    IndicesDescriptor(
        size_t numel,
        int ndim,
        const size_t *shape,
        const ptrdiff_t *strides,
        std::shared_ptr<device::nvidia::Handle::Internal> internal,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _numel(numel),
          _ndim(ndim),
          _internal(internal) {
        _shape = new size_t[ndim];
        _strides = new ptrdiff_t[ndim];
        for (int i = 0; i < ndim; ++i) {
            _shape[i] = shape[i];
            _strides[i] = strides[i];
        }
    }

    ~IndicesDescriptor() {
        delete[] _shape;
        delete[] _strides;
    }

    size_t workspaceSize() const;

    static infiniStatus_t create(
        infiniopHandle_t handle,
        IndicesDescriptor **desc_ptr,
        infiniopTensorDescriptor_t cond_desc);

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void **outputs, // NDIM 个输出张量的指针数组
        const void *cond,
        void *stream,
        size_t *num_true) const; // 输出：True 元素的数量
};

} // namespace op::where::nvidia

#endif // __WHERE_INDICES_NVIDIA_H__
