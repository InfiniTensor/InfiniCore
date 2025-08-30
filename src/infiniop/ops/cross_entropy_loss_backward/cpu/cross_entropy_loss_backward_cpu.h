#ifndef __CROSS_ENTROPY_LOSS_BACKWARD_CPU_H__
#define __CROSS_ENTROPY_LOSS_BACKWARD_CPU_H__
#include "../cross_entropy_loss_backward.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

CROSS_ENTROPY_LOSS_BACKWARD_ELEMENTWISE_DESCRIPTOR(cross_entropy_loss_backward, cpu)

#define CREATE_CROSS_ENTROPY_LOSS_BACKWARD_ELEMENTWISE_CPU_DESCRIPTOR(HANDLE, DTYPE, OUT_DESC, INPUT_DESC_VEC, BATCH_SIZE)         \
    auto info_result = op::elementwise::ElementwiseInfo::create(OUT_DESC, INPUT_DESC_VEC); \
    CHECK_RESULT(info_result);                                                             \
    *desc_ptr = new Descriptor(                                                            \
        DTYPE,                                                                             \
        info_result.take(),                                                                \
        nullptr,                                                                           \
        0,                                                                                 \
        HANDLE->device,                                                                    \
        HANDLE->device_id,                                                                 \
        BATCH_SIZE);

        
namespace op::cross_entropy_loss_backward::cpu {
typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    // float batch_size_f;
    // CrossEntropyLossBackwardOp(size_t n) :batch_size_f(utils::cast<float>(n)){};
    template <typename T>
    T operator()(const T& probs, const T& target, size_t batch_size) const {
        return utils::cast<float>(probs - target) / batch_size;
    }
} CrossEntropyLossBackwardOp;
} // namespace op::cross_entropy_loss_backward::cpu


#endif // __CROSS_ENTROPY_LOSS_BACKWARD_CPU_H__
