#ifndef __GAUSSIAN_NLL_LOSS_MOORE_H__
#define __GAUSSIAN_NLL_LOSS_MOORE_H__

#include "../../../operator.h"
#include "../../../devices/moore/moore_common.h"

namespace op::gaussian_nll_loss::moore {

enum class Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t input_size;
    int full;
    double eps;
    Reduction reduction;

    Descriptor(infiniDtype_t dtype, size_t input_size, int full, double eps, Reduction reduction,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          input_size(input_size),
          full(full),
          eps(eps),
          reduction(reduction) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        infiniopTensorDescriptor_t var_desc,
        int full,
        double eps,
        int reduction);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *input,
        const void *target,
        const void *var,
        void *stream) const;
};

} // namespace op::gaussian_nll_loss::moore

#endif // __GAUSSIAN_NLL_LOSS_MOORE_H__
