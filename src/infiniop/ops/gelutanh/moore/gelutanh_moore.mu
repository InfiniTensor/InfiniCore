#include "../../../elementwise/moore/elementwise_moore.h"
#include "gelutanh_moore.h"

#include <cmath>
#include <type_traits>

namespace op::gelutanh::moore {

struct GeluTanhOp {
    static constexpr size_t num_inputs = 1;
    static constexpr float alpha = 0.7978845608f;
    static constexpr float beta = 0.044715f;

    __device__ __forceinline__ float tanh_approx(float x) const {
        return __fsub_rn(__fmul_rn(2.0f, __frcp_rn(__fadd_rn(1.0f, __expf(-2.0f * x)))), 1.0f);
    }

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float inner = alpha * (xf + beta * xf * xf * xf);
            return __float2half(xf * 0.5f * (1.0f + tanh_approx(inner)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            float inner = alpha * (xf + beta * xf * xf * xf);
            return __float2bfloat16_rn(xf * 0.5f * (1.0f + tanh_approx(inner)));
        } else if constexpr (std::is_same_v<T, float>) {
            float inner = alpha * (x + beta * x * x * x);
            return __fmul_rn(x, __fmul_rn(0.5f, __fadd_rn(1.0f, tanh_approx(inner))));
        } else {
            double xd = static_cast<double>(x);
            double inner = static_cast<double>(alpha) * (xd + static_cast<double>(beta) * xd * xd * xd);
            double tanh_inner = 2.0 / (1.0 + exp(-2.0 * inner)) - 1.0;
            return static_cast<T>(xd * 0.5 * (1.0 + tanh_inner));
        }
    }
};

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();
    const auto &x_desc = input_desc_vec.at(0);

    CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    CHECK_SAME_SHAPE(out_desc->shape(), x_desc->shape());

    CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, GeluTanhOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, GeluTanhOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, GeluTanhOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, GeluTanhOp, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::gelutanh::moore
