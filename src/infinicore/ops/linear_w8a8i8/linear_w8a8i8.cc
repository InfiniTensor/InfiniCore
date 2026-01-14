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
    Size out_features = weight_packed->shape()[0];
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    // tempe for batch=1
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());
    linear_w8a8i8_(out->view({output_shape[1], output_shape[2]}),
                   input->view({input->shape()[1], input->shape()[2]}),
                   weight_packed,
                   weight_scale,
                   bias);
    // auto out = Tensor::empty(output_shape, input->dtype(), input->device());
    // linear_w8a8i8_(out,
    //                input,
    //                weight_packed,
    //                weight_scale,
    //                bias);
    // out->debug();
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
        weight_packed->permute({1, 0}),
        weight_scale,
        bias);
}

} // namespace infinicore::op
