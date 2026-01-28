#ifndef __CROSS_ENTROPY_LOSS_INFO_H__
#define __CROSS_ENTROPY_LOSS_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::cross_entropy_loss {

class CrossEntropyInfo {
public:
    CrossEntropyInfo() = default;
    size_t batch = 0;
    size_t num_classes = 0;
    infiniDtype_t dtype;

    static utils::Result<CrossEntropyInfo> create(
        infiniopTensorDescriptor_t loss,
        infiniopTensorDescriptor_t logits,
        infiniopTensorDescriptor_t target) {

        if (logits->ndim() != 2 || loss->ndim() != 1 || target->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        CrossEntropyInfo info;
        info.batch = logits->dim(0);
        info.num_classes = logits->dim(1);
        info.dtype = logits->dtype();
        return utils::Result<CrossEntropyInfo>(std::move(info));
    }
};

} // namespace op::cross_entropy_loss

#endif // __CROSS_ENTROPY_LOSS_INFO_H__
