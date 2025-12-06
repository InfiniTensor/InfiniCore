#include "infinicore/ops/bilinear.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/rearrange.hpp"

namespace infinicore::op {

Tensor bilinear(Tensor x1, Tensor x2, Tensor weight, std::optional<Tensor> bias) {

    size_t batch_size = x1->shape()[0];
    size_t in1_features = x1->shape()[1];
    size_t in2_features = x2->shape()[1];
    size_t out_features = weight->shape()[0];

    auto dtype = x1->dtype();
    auto device = x1->device();
    Tensor x1_cont = x1->is_contiguous() ? x1 : x1->contiguous();
    Tensor x2_cont = x2->is_contiguous() ? x2 : x2->contiguous();
    Tensor weight_cont = weight->is_contiguous() ? weight : weight->contiguous();

    Tensor weight_permuted = weight_cont->permute({1, 0, 2});
    Tensor weight_permuted_cont = weight_permuted->contiguous();
    Tensor weight_matrix = weight_permuted_cont->view({in1_features, out_features * in2_features});

    Tensor intermediate = matmul(x1_cont, weight_matrix, 1.0f);

    Tensor intermediate_3d = intermediate->view({batch_size, out_features, in2_features});

    Tensor intermediate_3d_trans = intermediate_3d->permute({0, 2, 1})->contiguous();

    Tensor x2_row = x2_cont->view({batch_size, 1, in2_features});

    Tensor out_3d = matmul(x2_row, intermediate_3d_trans, 1.0f);
    Tensor out = out_3d->view({batch_size, out_features});

    if (bias) {
        Tensor bias_broadcast = (*bias)->as_strided({batch_size, out_features}, {0, (*bias)->strides()[0]});
        out = add(out, bias_broadcast);
    }
    return out;
}

void bilinear_(Tensor out, Tensor x1, Tensor x2, Tensor weight, std::optional<Tensor> bias) {
    Tensor result = bilinear(x1, x2, weight, bias);
    // Copy result to out
    rearrange_(out, result);
}

} // namespace infinicore::op