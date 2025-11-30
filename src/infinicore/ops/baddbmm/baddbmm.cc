#include "infinicore/ops/baddbmm.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/mul.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/rearrange.hpp"
#include <optional>

namespace infinicore::op {

Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2, 
                 std::optional<Tensor> beta, 
                 std::optional<Tensor> alpha) {

    size_t batch_size = batch1->shape()[0];
    size_t m = batch1->shape()[1];
    size_t k = batch1->shape()[2];
    size_t n = batch2->shape()[2];

    Tensor input_cont = input->is_contiguous() ? input : input->contiguous();
    Tensor batch1_cont = batch1->is_contiguous() ? batch1 : batch1->contiguous();
    Tensor batch2_cont = batch2->is_contiguous() ? batch2 : batch2->contiguous();

    Tensor result = matmul(batch1_cont, batch2_cont);

    if (alpha.has_value()) {
        Tensor alpha_broadcast = (*alpha)->as_strided({batch_size, m ,n}, {0, 0, 0});
        result = mul(alpha_broadcast, result);
    }

    Tensor input_part = input_cont;
    if (input_part->ndim() == 2) {
         input_part = input_part->as_strided({batch_size, m, n}, {0, input_part->strides()[0], input_part->strides()[1]});
    } else if (input_part->ndim() == 3 && input_part->shape()[0] == 1 && batch_size > 1) {
         input_part = input_part->as_strided({batch_size, m, n}, {0, input_part->strides()[1], input_part->strides()[2]});
    }

    if (beta.has_value()) {
        Tensor beta_broadcast = (*beta)->as_strided({batch_size, m ,n}, {0, 0, 0});
        input_part = mul(beta_broadcast, input_part);
    }

    return add(input_part, result);
}

void baddbmm_(Tensor out, Tensor input, Tensor batch1, Tensor batch2, 
                 std::optional<Tensor> beta, 
                 std::optional<Tensor> alpha) {
    Tensor result = baddbmm(input, batch1, batch2, beta, alpha);
    // Copy result to out
    rearrange_(out, result);
}

}   // namespace infinicore::op