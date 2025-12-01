#include "diagflat_cpu.h"

namespace op::diagflat::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_descs,
    int64_t offset) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    if (input_descs.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto in_desc = input_descs[0];
    auto dtype = out_desc->dtype();
    auto in_shape = in_desc->shape();
    auto in_strides = in_desc->strides();
    auto out_shape = out_desc->shape();
    auto ndim = in_desc->ndim();

    // 支持与 diagflat 测试一致的 dtype
    CHECK_DTYPE(
        dtype,
        INFINI_DTYPE_BF16,
        INFINI_DTYPE_F16,
        INFINI_DTYPE_F32,
        INFINI_DTYPE_F64);

    // 只要求输出连续，输入可以非连续
    if (!out_desc->isContiguous()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    // NOTE: 这里只做非常轻量的 shape 检查：
    // 输入展平后长度 n，输出必须是 2D 或更高维最后两维组成的矩阵，至少能容纳 diag。
    size_t in_numel = in_desc->numel();
    if (out_shape.size() < 2) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    size_t n = out_shape[out_shape.size() - 2];
    size_t m = out_shape[out_shape.size() - 1];
    // 最长对角线长度
    size_t max_diag = 0;
    if (offset >= 0) {
        if (offset >= static_cast<int64_t>(m)) {
            max_diag = 0;
        } else {
            max_diag = std::min<size_t>(n, m - static_cast<size_t>(offset));
        }
    } else { // offset < 0
        if (-offset >= static_cast<int64_t>(n)) {
            max_diag = 0;
        } else {
            max_diag = std::min<size_t>(m, n - static_cast<size_t>(-offset));
        }
    }
    if (in_numel > max_diag && max_diag > 0) {
        // 输入比对角线可容纳的更长，按 torch.diagflat 语义会被截断；
        // 这里允许这种情况（kernel 里会在越界前 break），不报错。
        (void)in_numel;
    }

    bool input_contiguous = in_desc->isContiguous();

    *desc_ptr = new Descriptor(
        dtype,
        std::move(in_shape),
        std::move(in_strides),
        std::move(out_shape),
        offset,
        0,
        input_contiguous,
        ndim,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
static void diagflat_kernel(
    const T *x,
    T *y,
    const std::vector<size_t> &in_shape,
    const std::vector<ptrdiff_t> &in_strides,
    bool input_contiguous,
    const std::vector<size_t> &out_shape,
    int64_t offset,
    size_t ndim) {

    // 计算输出元素个数并清零
    size_t out_numel = 1;
    for (auto d : out_shape) {
        out_numel *= d;
    }
    for (size_t i = 0; i < out_numel; ++i) {
        y[i] = T{};
    }

    // 输入展平
    size_t in_numel = 1;
    for (auto d : in_shape) {
        in_numel *= d;
    }

    // 视输出为二维矩阵 (n, m)
    size_t n = out_shape[out_shape.size() - 2];
    size_t m = out_shape[out_shape.size() - 1];

    size_t i0 = 0;
    size_t j0 = 0;
    if (offset >= 0) {
        j0 = static_cast<size_t>(offset);
    } else {
        i0 = static_cast<size_t>(-offset);
    }

    for (size_t k = 0; k < in_numel; ++k) {
        // 计算输入真实 index（支持非连续）
        size_t src_idx;
        if (input_contiguous) {
            src_idx = k;
        } else {
            src_idx = op::common_cpu::indexToOffset(
                k,
                ndim,
                in_shape.data(),
                in_strides.data());
        }
        size_t ii = i0 + k;
        size_t jj = j0 + k;
        if (ii >= n || jj >= m) {
            break;
        }
        size_t idx = ii * m + jj;
        y[idx] = x[src_idx];
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    if (inputs.size() != 1) {
        return INFINI_STATUS_BAD_PARAM;
    }

    const void *x = inputs[0];

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        diagflat_kernel(
            static_cast<const fp16_t *>(x),
            static_cast<fp16_t *>(output),
            _input_shape,
            _input_strides,
            _input_contiguous,
            _output_shape,
            _offset,
            _ndim);
        break;
    case INFINI_DTYPE_BF16:
        diagflat_kernel(
            static_cast<const bf16_t *>(x),
            static_cast<bf16_t *>(output),
            _input_shape,
            _input_strides,
            _input_contiguous,
            _output_shape,
            _offset,
            _ndim);
        break;
    case INFINI_DTYPE_F32:
        diagflat_kernel(
            static_cast<const float *>(x),
            static_cast<float *>(output),
            _input_shape,
            _input_strides,
            _input_contiguous,
            _output_shape,
            _offset,
            _ndim);
        break;
    case INFINI_DTYPE_F64:
        diagflat_kernel(
            static_cast<const double *>(x),
            static_cast<double *>(output),
            _input_shape,
            _input_strides,
            _input_contiguous,
            _output_shape,
            _offset,
            _ndim);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::diagflat::cpu