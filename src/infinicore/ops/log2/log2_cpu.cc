#include "../../../utils.h"

#include "infinicore/device.hpp"
#include "infinicore/ops/log2.hpp"
#include "infinicore/tensor.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace infinicore::op::log2_impl::infiniop {

void calculate(Tensor output, Tensor input) {
    auto strides = input->strides(); // vector<long>
    auto shapes = input->shape();    // vector<size_t>
    auto ndim = input->ndim();
    auto dtype = input->dtype();
    auto dtype_size = input->element_size();
    auto numel = input->numel();

    auto input_base = input->data();
    auto output_base = output->data();

    std::vector<size_t> indices(ndim, 0);
    for (size_t idx = 0; idx < numel; ++idx) {
        // Calculate the offset for the current index
        size_t offset = 0;
        for (size_t dim = 0; dim < ndim; ++dim) {
            offset += indices[dim] * strides[dim];
        }

        // Compute log2 for the current element
        if (dtype == DataType::F32) {
            auto *input_ptr = reinterpret_cast<float *>(input_base + offset * dtype_size);
            auto *output_ptr = reinterpret_cast<float *>(output_base + offset * dtype_size);
            *output_ptr = std::log2(*input_ptr);
        } else if (dtype == DataType::F64) {
            auto *input_ptr = reinterpret_cast<double *>(input_base + offset * dtype_size);
            auto *output_ptr = reinterpret_cast<double *>(output_base + offset * dtype_size);
            *output_ptr = std::log2(*input_ptr);
        } else if (dtype == DataType::F16) {
            // F16: 转换为 F32 计算，再转回 F16
            auto *input_ptr = reinterpret_cast<fp16_t *>(input_base + offset * dtype_size);
            auto *output_ptr = reinterpret_cast<fp16_t *>(output_base + offset * dtype_size);

            float input_f32 = utils::cast<float>(*input_ptr);
            float output_f32 = std::log2(input_f32);
            *output_ptr = utils::cast<fp16_t>(output_f32);
        } else {
            throw std::runtime_error("Unsupported data type for log2 operation.");
        }

        // Update indices
        for (ssize_t dim = ndim - 1; dim >= 0; --dim) {
            indices[dim]++;
            if (indices[dim] < shapes[dim]) {
                break;
            } else {
                indices[dim] = 0;
            }
        }
    }
}

static bool registered = []() {
    Log2::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::log2_impl::infiniop