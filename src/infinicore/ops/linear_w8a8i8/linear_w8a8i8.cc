#include "infinicore/ops/linear_w8a8i8.hpp"
#include "infinicore/ops/per_channel_quant_i8.hpp"
#include "infinicore/ops/scaled_mm_i8.hpp"
#include <iostream>
namespace infinicore::op {

Tensor linear_w8a8i8(Tensor input,
                     Tensor weight_packed,
                     Tensor weight_scale,
                     std::optional<Tensor> bias) {
    Size ndim = input->ndim();
    Size out_features = weight_packed->shape()[1];
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());
    linear_w8a8i8_(out,
                   input,
                   weight_packed,
                   weight_scale,
                   bias);

    return out;
}

void linear_w8a8i8_(Tensor out,
                    Tensor input,
                    Tensor weight_packed,
                    Tensor weight_scale,
                    std::optional<Tensor> bias) {

    auto input_packed = Tensor::empty(
        input->shape(),
        DataType::I8,
        input->device());
    auto input_scale = Tensor::empty(
        {input->shape()[0], 1},
        DataType::F32,
        input->device());
    op::per_channel_quant_i8_(input, input_packed, input_scale);

    op::scaled_mm_i8_(
        out,
        input_packed,
        input_scale,
        weight_packed,
        weight_scale,
        bias);
}

} // namespace infinicore::op
