#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/histc.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infinicore::op::histc_impl::cpu {

void calculate(Tensor input, Tensor output, size_t bins, double min, double max) {
    if (bins == 0) {
        throw std::runtime_error("histc CPU: bins must be > 0");
    }
    if (!(max > min)) {
        throw std::runtime_error("histc CPU: require max > min");
    }

    auto dtype = input->dtype();
    // 输出一维 [bins]，按创建逻辑默认连续
    auto out_base = output->data();
    const auto out_esize = output->element_size();

    // 初始化输出为 0
    for (size_t i = 0; i < bins; ++i) {
        const size_t off = i;
        if (dtype == DataType::F32) {
            auto *ptr = reinterpret_cast<float *>(out_base + off * out_esize);
            *ptr = 0.0f;
        } else if (dtype == DataType::F64) {
            auto *ptr = reinterpret_cast<double *>(out_base + off * out_esize);
            *ptr = 0.0;
        } else if (dtype == DataType::F16) {
            auto *ptr = reinterpret_cast<fp16_t *>(out_base + off * out_esize);
            *ptr = utils::cast<fp16_t>(0.0f);
        } else if (dtype == DataType::BF16) {
            auto *ptr = reinterpret_cast<bf16_t *>(out_base + off * out_esize);
            *ptr = utils::cast<bf16_t>(0.0f);
        } else {
            throw std::runtime_error("histc CPU: unsupported dtype for output");
        }
    }

    // 输入遍历（支持任意形状/步长），对每个元素进行分箱
    auto in_base = input->data();
    const auto in_esize = input->element_size();
    auto strides = input->strides();
    auto shapes = input->shape();
    const size_t ndim = input->ndim();
    const size_t numel = input->numel();

    const double width = (max - min) / static_cast<double>(bins);
    if (!(width > 0.0)) {
        // 极端情况：width==0（min==max），将等于 max 的值计入最后一箱
        // 其他值忽略
    }

    std::vector<size_t> indices(ndim, 0);
    for (size_t idx = 0; idx < numel; ++idx) {
        size_t off = 0;
        for (size_t d = 0; d < ndim; ++d) {
            off += indices[d] * static_cast<size_t>(strides[d]);
        }

        double vald = 0.0;
        if (dtype == DataType::F32) {
            auto *p = reinterpret_cast<float *>(in_base + off * in_esize);
            vald = static_cast<double>(*p);
        } else if (dtype == DataType::F64) {
            auto *p = reinterpret_cast<double *>(in_base + off * in_esize);
            vald = *p;
        } else if (dtype == DataType::F16) {
            auto *p = reinterpret_cast<fp16_t *>(in_base + off * in_esize);
            vald = static_cast<double>(utils::cast<float>(*p));
        } else if (dtype == DataType::BF16) {
            auto *p = reinterpret_cast<bf16_t *>(in_base + off * in_esize);
            vald = static_cast<double>(utils::cast<float>(*p));
        } else {
            throw std::runtime_error("histc CPU: unsupported dtype for input");
        }

        // 计算箱索引
        ssize_t bin = -1;
        if (vald < min || vald > max) {
            bin = -1; // 忽略越界
        } else if (vald == max) {
            bin = static_cast<ssize_t>(bins - 1);
        } else if (width > 0.0) {
            double pos = (vald - min) / width;
            ssize_t ib = static_cast<ssize_t>(std::floor(pos));
            if (ib < 0) {
                ib = 0;
            }
            if (ib >= static_cast<ssize_t>(bins)) {
                ib = static_cast<ssize_t>(bins - 1);
            }
            bin = ib;
        }

        if (bin >= 0) {
            const size_t out_off = static_cast<size_t>(bin);
            if (dtype == DataType::F32) {
                auto *op = reinterpret_cast<float *>(out_base + out_off * out_esize);
                *op = *op + 1.0f;
            } else if (dtype == DataType::F64) {
                auto *op = reinterpret_cast<double *>(out_base + out_off * out_esize);
                *op = *op + 1.0;
            } else if (dtype == DataType::F16) {
                auto *op = reinterpret_cast<fp16_t *>(out_base + out_off * out_esize);
                float cur = utils::cast<float>(*op);
                *op = utils::cast<fp16_t>(cur + 1.0f);
            } else if (dtype == DataType::BF16) {
                auto *op = reinterpret_cast<bf16_t *>(out_base + out_off * out_esize);
                float cur = utils::cast<float>(*op);
                *op = utils::cast<bf16_t>(cur + 1.0f);
            }
        }

        // 更新多维索引
        for (ssize_t d = static_cast<ssize_t>(ndim) - 1; d >= 0; --d) {
            indices[static_cast<size_t>(d)]++;
            if (indices[static_cast<size_t>(d)] < shapes[static_cast<size_t>(d)]) {
                break;
            } else {
                indices[static_cast<size_t>(d)] = 0;
            }
        }
    }
}

static bool registered = []() {
    Histc::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::histc_impl::cpu
