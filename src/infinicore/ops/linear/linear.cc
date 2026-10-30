#include "infinicore/ops/linear.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/gemm.hpp"
#include "infinicore/ops/rearrange.hpp"

#include <stdexcept>

namespace infinicore::op {

Tensor linear(Tensor input,
              Tensor weight,
              std::optional<Tensor> bias,
              float alpha) {

    Size ndim = input->ndim();
    Size out_features = weight->shape()[0];

    // Assign memory to out variables
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    // Inplace Calculate
    linear_(out, input, weight, bias, alpha);
    return out;
}

void linear_(Tensor out,
             Tensor input,
             Tensor weight,
             std::optional<Tensor> bias,
             float alpha) {

    auto weight_shape = weight->shape();
    Size out_features = weight_shape[0];
    Size in_features = weight_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    // Calculate the number of features
    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    // linear transformation
    Tensor out_view = out->view({N, out_features});
    // Add bias
    float beta = 0.0f;
    if (bias.has_value()) {
        rearrange_(out_view,
                   bias.value()->as_strided({N, out_features}, {0, 1}));
        beta = 1.0f;
    }

    gemm_(out_view,
          input->view({N, in_features}),
          weight->permute({1, 0}), alpha, beta);
}

// --- Pre-packed weight variants ---
// packed_weight layout: [in_features, out_features] (already transposed from [OC,IC])

Tensor linear_packed(Tensor input,
                     Tensor packed_weight,
                     std::optional<Tensor> bias,
                     float alpha) {

    Size ndim = input->ndim();
    // packed_weight shape is [IC, OC], so OC is dim 1
    Size out_features = packed_weight->shape()[1];

    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());

    linear_packed_(out, input, packed_weight, bias, alpha);
    return out;
}

void linear_packed_(Tensor out,
                    Tensor input,
                    Tensor packed_weight,
                    std::optional<Tensor> bias,
                    float alpha) {

    auto weight_shape = packed_weight->shape();
    // packed_weight is [IC, OC] — already the layout GEMM expects
    Size in_features = weight_shape[0];
    Size out_features = weight_shape[1];

    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    Size N = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        N *= input_shape[i];
    }

    Tensor out_view = out->view({N, out_features});

    // Add bias (same logic as linear_)
    float beta = 0.0f;
    if (bias.has_value()) {
        rearrange_(out_view,
                   bias.value()->as_strided({N, out_features}, {0, 1}));
        beta = 1.0f;
    }

    // KEY: no weight->permute({1, 0}) here!
    // packed_weight is already [IC, OC] contiguous.
    gemm_(out_view,
          input->view({N, in_features}),
          packed_weight, alpha, beta);
}

Tensor linear_packed_nz(Tensor input,
                        Tensor nz_weight,
                        std::optional<Tensor> bias,
                        float alpha) {
    Size ndim = input->ndim();
    Size out_features = nz_weight->shape()[1];
    auto output_shape = input->shape();
    output_shape[ndim - 1] = out_features;
    auto out = Tensor::empty(output_shape, input->dtype(), input->device());
    linear_packed_nz_(out, input, nz_weight, bias, alpha);
    return out;
}

void linear_packed_nz_(Tensor out,
                       Tensor input,
                       Tensor nz_weight,
                       std::optional<Tensor> bias,
                       float alpha) {
    Size in_features = nz_weight->shape()[0];
    Size out_features = nz_weight->shape()[1];
    Size ndim = input->ndim();
    assert(out->ndim() == ndim);

    Size rows = 1;
    auto input_shape = input->shape();
    for (size_t i = 0; i < ndim - 1; ++i) {
        rows *= input_shape[i];
    }

    Tensor out_view = out->view({rows, out_features});

    // aclnnMatmulWeightNz has pure MatMul semantics and does not accept the
    // beta=1 accumulation pattern used by the ND GEMM bias path. Keep the NZ
    // GEMM itself bias-free, then use the regular graph-aware add chain for
    // the broadcast bias. The Ascend add backend uses aclnnAdd and supports
    // output aliasing input, so this remains in-place without breaking graph
    // capture between the MatMul and the following attention operators.
    gemm_nz_(out_view,
             input->view({rows, in_features}),
             nz_weight, alpha, 0.0f);

    if (bias.has_value()) {
        if (out->device().getType() == Device::Type::ASCEND) {
            add_(out_view, out_view,
                 bias.value()->as_strided({rows, out_features}, {0, 1}));
        } else
        {
            throw std::runtime_error(
                "linear_packed_nz with bias is only implemented on Ascend");
        }
    }
}

} // namespace infinicore::op
