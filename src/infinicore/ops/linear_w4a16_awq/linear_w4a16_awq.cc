#include "infinicore/ops/linear_w4a16_awq.hpp"
#include "infinicore/ops/dequantize_awq.hpp"
#include "infinicore/ops/gemm.hpp"

namespace infinicore::op {

Tensor linear_w4a16_awq(Tensor input,
                        Tensor weight_packed,
                        Tensor weight_scale,
                        Tensor weight_zeros,
                        std::optional<Tensor> bias) {

    // Input is of shape [M, K], Weight_packed is of shape [N, K],stirdes is [N, 1]
    Size ndim = input->ndim();
    Size out_features = weight_packed->shape()[0];

    // Assign memory to out variables
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    // Inplace Calculate
    linear_w4a16_awq_(out, input, weight_packed, weight_scale, weight_zeros, bias);
    return out;
}

void linear_w4a16_awq_(Tensor out,
                       Tensor input,
                       Tensor weight_packed,
                       Tensor weight_scale,
                       Tensor weight_zeros,
                       std::optional<Tensor> bias) {

    auto weight_packed_shape = weight_packed->shape();
    Size out_features = weight_packed_shape[0];
    Size in_features = weight_packed_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    auto weight = Tensor::empty(
        {out_features, in_features},
        out->dtype(),
        weight_packed->device());
    // op::per_channel_quant_i8_(input->view({N, in_features}), input_packed, input_scale);
    float alpha = 1.0f;
    float beta = 0.0f;
    op::dequantize_awq_(weight, weight_packed, weight_scale, weight_zeros);
    bias = std::make_optional(bias.value()->as_strided({N, out_features}, {0, 1}));
    // op::scaled_mm_i8_(
    //     out->view({N, out_features}),
    //     input_packed,
    //     input_scale,
    //     weight_packed->permute({1, 0}),
    //     weight_scale,
    //     bias);
    gemm_(out->view({N, out_features}),
          input->view({N, in_features}),
          weight->permute({1, 0}), alpha, beta);
}

} // namespace infinicore::op
