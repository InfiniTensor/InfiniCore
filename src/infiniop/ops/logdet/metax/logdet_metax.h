#ifndef __LOGDET_METAX_H__
#define __LOGDET_METAX_H__

#include "../../../operator.h"
#include "../../../devices/metax/metax_common.h"

namespace op::logdet::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t matrix_size;
    size_t input_size;

    Descriptor(infiniDtype_t dtype, size_t matrix_size, size_t input_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          matrix_size(matrix_size),
          input_size(input_size) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc);

    size_t workspaceSize() const { return matrix_size * matrix_size * sizeof(double) * 2; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::logdet::metax

#endif // __LOGDET_METAX_H__
