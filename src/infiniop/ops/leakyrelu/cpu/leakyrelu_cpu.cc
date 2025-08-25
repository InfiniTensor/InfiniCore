#include "leakyrelu_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../info.h"
#include "infinicore.h"
#include <algorithm>

namespace op::leakyrelu::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    float negative_slope) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info_r = LeakyReLUInfo::create(out_desc, in_desc, negative_slope);
    CHECK_RESULT(info_r);

    *desc_ptr = new Descriptor(
        info_r.take(),
        0,
        nullptr,
        handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

size_t Descriptor::workspaceSize() const { return _min_workspace_size; }

template <typename T>
static inline void cpu_leakyrelu_impl_incremental(
    void *output, const void *input, const op::leakyrelu::LeakyReLUInfo &info) {

    const size_t ndim = info.shape.size();
    const size_t n = info.n;

    if (n == 0) {
        return;
    }

    auto out_base = reinterpret_cast<T *>(output);
    auto in_base = reinterpret_cast<const T *>(input);

    const std::vector<size_t> &shape = info.shape;
    const std::vector<ptrdiff_t> &in_stride = info.in_stride;
    const std::vector<ptrdiff_t> &out_stride = info.out_stride;

    std::vector<size_t> idx(ndim, 0);
    ptrdiff_t in_off = 0;
    ptrdiff_t out_off = 0;

    for (size_t it = 0; it < n; ++it) {
        const T *in_elem = in_base + in_off;
        T *out_elem = out_base + out_off;

        float v = utils::cast<float, T>(*in_elem);
        float outv = v >= 0.0f ? v : v * info.negative_slope;
        *out_elem = utils::cast<T, float>(outv);
        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            idx[d] += 1;
            if (in_stride[d] != 0) {
                in_off += in_stride[d];
            }
            if (out_stride[d] != 0) {
                out_off += out_stride[d];
            }

            if (idx[d] < shape[d]) {
                break;
            } else {
                idx[d] = 0;
                if (in_stride[d] != 0) {
                    in_off -= static_cast<ptrdiff_t>(shape[d]) * in_stride[d];
                }
                if (out_stride[d] != 0) {
                    out_off -= static_cast<ptrdiff_t>(shape[d]) * out_stride[d];
                }
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    switch (_info.dt_in) {
    case INFINI_DTYPE_F16:
        cpu_leakyrelu_impl_incremental<fp16_t>(output, input, _info);
        break;
    case INFINI_DTYPE_BF16:
        cpu_leakyrelu_impl_incremental<bf16_t>(output, input, _info);
        break;
    case INFINI_DTYPE_F32:
        cpu_leakyrelu_impl_incremental<float>(output, input, _info);
        break;
    case INFINI_DTYPE_F64:
        cpu_leakyrelu_impl_incremental<double>(output, input, _info);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::leakyrelu::cpu
