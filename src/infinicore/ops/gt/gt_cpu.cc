#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/gt.hpp"
#include <omp.h>

namespace infinicore::op::gt_impl::cpu {

template <typename T>
void calculate_gt_cpu(Tensor input, Tensor other, Tensor output) {
    auto in_ptr = reinterpret_cast<const T *>(input->data());
    auto other_ptr = reinterpret_cast<const T *>(other->data());
    auto out_base = output->data();

    auto shape = input->shape();
    auto in_strides = input->strides();
    auto other_strides = other->strides();
    auto out_strides = output->strides();
    auto out_dtype = output->dtype();
    int ndim = input->ndim();
    size_t numel = input->numel();

#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        size_t temp_idx = i;
        size_t in_off = 0;
        size_t other_off = 0;
        size_t out_off = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp_idx % shape[d];
            temp_idx /= shape[d];
            in_off += coord * in_strides[d];
            other_off += coord * other_strides[d];
            out_off += coord * out_strides[d];
        }

        bool result = utils::cast<float>(in_ptr[in_off]) > utils::cast<float>(other_ptr[other_off]);

        if (out_dtype == DataType::BOOL) {
            *(reinterpret_cast<bool *>(out_base + out_off)) = result;
        } else if (out_dtype == DataType::F32) {
            *(reinterpret_cast<float *>(out_base + out_off * sizeof(float))) = result ? 1.0f : 0.0f;
        } else if (out_dtype == DataType::F16) {
            *(reinterpret_cast<fp16_t *>(out_base + out_off * sizeof(fp16_t))) = utils::cast<fp16_t>(result ? 1.0f : 0.0f);
        } else if (out_dtype == DataType::I32) {
            *(reinterpret_cast<int32_t *>(out_base + out_off * sizeof(int32_t))) = result ? 1 : 0;
        }
    }
}

void calculate(Tensor input, Tensor other, Tensor output) {
    auto dtype = input->dtype();
    if (dtype == DataType::F32) {
        calculate_gt_cpu<float>(input, other, output);
    } else if (dtype == DataType::F16) {
        calculate_gt_cpu<fp16_t>(input, other, output);
    } else if (dtype == DataType::BF16) {
        calculate_gt_cpu<bf16_t>(input, other, output);
    } else if (dtype == DataType::I32) {
        calculate_gt_cpu<int32_t>(input, other, output);
    } else {
        throw std::runtime_error("GT unsupported dtype");
    }
}

static bool registered = []() {
    Gt::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::gt_impl::cpu