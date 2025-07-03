#include "softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::softmax::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    int axis) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = y->dtype();

    const auto &x_shape = x->shape();
    const auto &y_shape = y->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

    CHECK_SAME_SHAPE(y_shape, x_shape);

    auto result = SoftmaxInfo::create(y, x, axis);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        dtype,
        result.take(),
        0,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void softmax_cpu(const SoftmaxInfo &info,
                 const void *x, void *y, int axis) {
    int dim_size = info.dim_size;
    int stride = info.stride;
    int other_size = info.other_size;
    auto input = reinterpret_cast<const T *>(x);
    auto output = reinterpret_cast<T *>(y);

    auto compute_softmax = [&](int i) {
        int tid = i % stride + (i - i % stride) * dim_size;

        float max_data = -INFINITY;
        for (int j = 0; j < dim_size; j++) {
            int index = tid + j * stride;
            max_data = fmax(max_data, utils::cast<float>(input[index]));
        }

        float sum_data = 0.0f;
        for (int j = 0; j < dim_size; j++) {
            int index = tid + j * stride;
            sum_data += std::exp(utils::cast<float>(input[index]) - max_data);
        }

        for (int j = 0; j < dim_size; j++) {
            int index = tid + j * stride;
            float result = std::exp(utils::cast<float>(input[index]) - max_data) / sum_data;
            output[index] = utils::cast<T>(result);
        }
    };
#pragma omp parallel for
    for (int i = 0; i < other_size; i++) {
        compute_softmax(i);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream_) const {
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        softmax_cpu<fp16_t>(_info, x, y, _info.axis);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        softmax_cpu<float>(_info, x, y, _info.axis);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}
} // namespace op::softmax::cpu
