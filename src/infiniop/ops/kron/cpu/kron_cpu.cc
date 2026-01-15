#include "kron_cpu.h"
#include "../../../utils.h"

namespace op::kron::cpu {

utils::Result<KronInfo> KronInfo::create(
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto a_shape = a_desc->shape();
    auto b_shape = b_desc->shape();
    auto y_shape = y_desc->shape();

    // Kron requires same number of dimensions
    if (a_shape.size() != b_shape.size()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t ndim = a_shape.size();

    // Output shape: each dimension is a_shape[i] * b_shape[i]
    std::vector<size_t> expected_y_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        expected_y_shape[i] = a_shape[i] * b_shape[i];
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    KronInfo info;
    info.ndim = ndim;
    info.a_shape = a_shape;
    info.b_shape = b_shape;
    info.y_shape = y_shape;
    info.a_size = a_desc->numel();
    info.b_size = b_desc->numel();
    info.y_size = y_desc->numel();

    return utils::Result<KronInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto dtype = a_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto info_result = KronInfo::create(a_desc, b_desc, y_desc);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void kron_impl(
    const KronInfo &info,
    T *y,
    const T *a,
    const T *b) {

    // Kronecker product: for each element a[i] in A, multiply it with entire B
    // y[output_idx] = a[a_idx] * b[b_idx]
    // where output coordinates are computed from a and b coordinates

    // Helper to convert coordinates to linear index
    auto coordsToIndex = [](const std::vector<size_t> &coords, const std::vector<size_t> &shape) {
        size_t idx = 0;
        size_t stride = 1;
        for (size_t d = coords.size(); d-- > 0;) {
            idx += coords[d] * stride;
            stride *= shape[d];
        }
        return idx;
    };

    // Iterate over output tensor
    std::vector<size_t> y_coords(info.ndim, 0);
    for (size_t y_idx = 0; y_idx < info.y_size; ++y_idx) {
        // Convert linear index to coordinates
        size_t temp = y_idx;
        for (size_t d = info.ndim; d-- > 0;) {
            y_coords[d] = temp % info.y_shape[d];
            temp /= info.y_shape[d];
        }

        // Compute corresponding a and b coordinates
        std::vector<size_t> a_coords(info.ndim);
        std::vector<size_t> b_coords(info.ndim);
        for (size_t d = 0; d < info.ndim; ++d) {
            a_coords[d] = y_coords[d] / info.b_shape[d];
            b_coords[d] = y_coords[d] % info.b_shape[d];
        }

        size_t a_idx = coordsToIndex(a_coords, info.a_shape);
        size_t b_idx = coordsToIndex(b_coords, info.b_shape);

        y[y_idx] = a[a_idx] * b[b_idx];
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *a,
    const void *b,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        kron_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y),
                         reinterpret_cast<const fp16_t *>(a),
                         reinterpret_cast<const fp16_t *>(b));
        break;
    case INFINI_DTYPE_BF16:
        kron_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y),
                         reinterpret_cast<const bf16_t *>(a),
                         reinterpret_cast<const bf16_t *>(b));
        break;
    case INFINI_DTYPE_F32:
        kron_impl<float>(_info, reinterpret_cast<float *>(y),
                        reinterpret_cast<const float *>(a),
                        reinterpret_cast<const float *>(b));
        break;
    case INFINI_DTYPE_F64:
        kron_impl<double>(_info, reinterpret_cast<double *>(y),
                         reinterpret_cast<const double *>(a),
                         reinterpret_cast<const double *>(b));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::kron::cpu
