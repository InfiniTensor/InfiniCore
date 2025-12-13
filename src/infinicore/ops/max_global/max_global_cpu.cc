#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/max_global.hpp"
#include <float.h>
#include <limits>

namespace infinicore::op::max_global_impl::cpu {

void calculate(Tensor input, Tensor output) {
    auto strides = input->strides(); // vector<long>
    auto shapes = input->shape();    // vector<size_t>
    auto ndim = input->ndim();
    auto dtype = input->dtype();
    auto dtype_size = input->element_size();
    auto numel = input->numel();

    auto input_base = input->data();
    auto output_base = output->data();

    // ---- 正确的 max 变量定义（不再在 if block 内） ----
    float  max_f32  = -std::numeric_limits<float>::infinity();
    double max_f64  = -std::numeric_limits<double>::infinity();
    float  max_f16f = -std::numeric_limits<float>::infinity();  // F16 accumulate in F32

    // ---- 根据 dtype 分支遍历 ----

    // 初始化 indices
    std::vector<size_t> indices(ndim, 0);

    for (size_t idx = 0; idx < numel; ++idx) {
        size_t offset = 0;
        for (size_t dim = 0; dim < ndim; ++dim) {
            offset += indices[dim] * strides[dim];
        }

        if (dtype == DataType::F32) {
            auto* ptr = reinterpret_cast<float*>(input_base + offset * dtype_size);
            float v = *ptr;
            max_f32 = std::max(max_f32, v);

        } else if (dtype == DataType::F64) {
            auto* ptr = reinterpret_cast<double*>(input_base + offset * dtype_size);
            double v = *ptr;
            max_f64 = std::max(max_f64, v);

        } else if (dtype == DataType::F16) {
            auto* ptr = reinterpret_cast<fp16_t*>(input_base + offset * dtype_size);
            float v = utils::cast<float>(*ptr);
            max_f16f = std::max(max_f16f, v);

        } else {
            throw std::runtime_error("Unsupported dtype.");
        }

        // 更新 indices
        for (ssize_t dim = ndim - 1; dim >= 0; --dim) {
            indices[dim]++;
            if (indices[dim] < shapes[dim])
                break;
            indices[dim] = 0;
        }
    }

    // ---- 写输出（scalar）----
    if (dtype == DataType::F32) {
        auto* out = reinterpret_cast<float*>(output_base);
        *out = max_f32;

    } else if (dtype == DataType::F64) {
        auto* out = reinterpret_cast<double*>(output_base);
        *out = max_f64;

    } else if (dtype == DataType::F16) {
        auto* out = reinterpret_cast<fp16_t*>(output_base);
        *out = utils::cast<fp16_t>(max_f16f);

    } else {
        throw std::runtime_error("Unsupported dtype.");
    }
}


static bool registered = []() {
    MaxGlobal::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::max_global_impl::cpu
