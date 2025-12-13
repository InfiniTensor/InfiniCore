#ifndef __HINGE_EMBEDDING_LOSS_METAX_H__
#define __HINGE_EMBEDDING_LOSS_METAX_H__

#include "../../../operator.h"
#include "../../../devices/metax/metax_common.h"

namespace op::hinge_embedding_loss::metax {

enum class Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t input_size;
    double margin;
    Reduction reduction;

    Descriptor(infiniDtype_t dtype, size_t input_size, double margin, Reduction reduction,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          input_size(input_size),
          margin(margin),
          reduction(reduction) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        double margin,
        int reduction);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *input,
        const void *target,
        void *stream) const;
};

} // namespace op::hinge_embedding_loss::metax

#endif // __HINGE_EMBEDDING_LOSS_METAX_H__
