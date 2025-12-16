#ifndef __WHERE_INDICES_METAX_H__
#define __WHERE_INDICES_METAX_H__

#include "../../../devices/metax/metax_handle.h"
#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::where::metax {

class IndicesDescriptor final : public InfiniopDescriptor {
    size_t _numel;
    int _ndim;
    size_t *_shape;
    ptrdiff_t *_strides;

public:
    IndicesDescriptor(
        size_t numel,
        int ndim,
        const size_t *shape,
        const ptrdiff_t *strides,
        infiniDevice_t device_type,
        int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _numel(numel),
          _ndim(ndim) {
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
        void **outputs,
        const void *cond,
        void *stream,
        size_t *num_true) const;
};

} // namespace op::where::metax

#endif // __WHERE_INDICES_METAX_H__
