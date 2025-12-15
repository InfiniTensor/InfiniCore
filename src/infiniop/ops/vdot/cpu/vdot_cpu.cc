#include "vdot_cpu.h"
#include "../../../../utils.h"

namespace op::vdot::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto in_dtype = a_desc->dtype();
    auto b_dtype = b_desc->dtype();
    auto out_dtype = out_desc->dtype();

    // Inputs must be 1D vectors with same length
    if (a_desc->ndim() != 1 || b_desc->ndim() != 1) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (a_desc->numel() != b_desc->numel()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Input dtypes must match and be in supported set
    CHECK_OR_RETURN(in_dtype == b_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_DTYPE(
        in_dtype,
        INFINI_DTYPE_F16,
        INFINI_DTYPE_F32,
        INFINI_DTYPE_BF16);

    // Simplest: output dtype equals input dtype
    CHECK_OR_RETURN(out_dtype == in_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    size_t length = a_desc->numel();
    ptrdiff_t a_stride = a_desc->stride(0);
    ptrdiff_t b_stride = b_desc->stride(0);

    *desc_ptr = new Descriptor(
        in_dtype,
        out_dtype,
        length,
        a_stride,
        b_stride,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tin, typename Tout>
static void vdot_impl(
    Tout *out,
    const Tin *a_base,
    const Tin *b_base,
    size_t n,
    ptrdiff_t a_stride,
    ptrdiff_t b_stride) {

    if constexpr (std::is_same_v<Tin, fp16_t> || std::is_same_v<Tin, bf16_t>) {
        // Accumulate in float for half/bfloat16, then cast back
        float acc = 0.0f;

#pragma omp parallel for reduction(+ : acc)
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            const Tin &av = a_base[i * a_stride];
            const Tin &bv = b_base[i * b_stride];
            float av_f = utils::cast<float>(av);
            float bv_f = utils::cast<float>(bv);
            acc += av_f * bv_f;
        }

        *out = utils::cast<Tout>(acc);
    } else {
        Tout acc{};

#pragma omp parallel for reduction(+ : acc)
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(n); ++i) {
            const Tin &av = a_base[i * a_stride];
            const Tin &bv = b_base[i * b_stride];
            acc += static_cast<Tout>(av) * static_cast<Tout>(bv);
        }

        *out = acc;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    void *stream) const {

    switch (_in_dtype) {
    case INFINI_DTYPE_F16:
        vdot_impl<fp16_t, fp16_t>(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(a),
            reinterpret_cast<const fp16_t *>(b),
            _length,
            _a_stride,
            _b_stride);
        break;
    case INFINI_DTYPE_F32:
        vdot_impl<float, float>(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            _length,
            _a_stride,
            _b_stride);
        break;
    case INFINI_DTYPE_BF16:
        vdot_impl<bf16_t, bf16_t>(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(a),
            reinterpret_cast<const bf16_t *>(b),
            _length,
            _a_stride,
            _b_stride);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::vdot::cpu


