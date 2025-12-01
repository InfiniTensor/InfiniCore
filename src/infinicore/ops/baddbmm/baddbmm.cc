#include "infinicore/ops/baddbmm.hpp"
#include "infinicore/ops/gemm.hpp"
#include "infinicore/ops/rearrange.hpp"
#include <optional>

namespace infinicore::op {

Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2, 
                 float beta, 
                 float alpha) {

    size_t batch_size = batch1->shape()[0];
    size_t m = batch1->shape()[1];
    size_t n = batch2->shape()[2];
    
    Tensor batch1_cont = batch1->is_contiguous() ? batch1 : batch1->contiguous();
    Tensor batch2_cont = batch2->is_contiguous() ? batch2 : batch2->contiguous();

    bool input_is_target_shape = (input->ndim() == 3 && 
                                   input->shape()[0] == batch_size &&
                                   input->shape()[1] == m &&
                                   input->shape()[2] == n);
    
    if (input_is_target_shape && input->is_contiguous()) {
        Tensor result = Tensor::empty({batch_size, m, n}, batch1->dtype(), batch1->device());
        if (beta != 0.0f) {
            rearrange_(result, input);
        }
        gemm_(result, batch1_cont, batch2_cont, alpha, beta);
        return result;
    }
    
    Tensor result = Tensor::empty({batch_size, m, n}, batch1->dtype(), batch1->device());
    
    if (beta != 0.0f) {
        if (input->ndim() == 2) {
            auto strides = input->strides();
            Tensor input_broadcast = input->as_strided(
                {batch_size, m, n}, 
                {0, strides[0], strides[1]});
            rearrange_(result, input_broadcast);
        } else if (input->ndim() == 3 && input->shape()[0] == 1 && batch_size > 1) {
            auto strides = input->strides();
            Tensor input_broadcast = input->as_strided(
                {batch_size, m, n}, 
                {0, strides[1], strides[2]});
            rearrange_(result, input_broadcast);
        } else {
            rearrange_(result, input);
        }
    }

    gemm_(result, batch1_cont, batch2_cont, alpha, beta);

    return result;
}

void baddbmm_(Tensor out, Tensor input, Tensor batch1, Tensor batch2, 
                 float beta, 
                 float alpha) {
    Tensor result = baddbmm(input, batch1, batch2, beta, alpha);
    rearrange_(out, result);
}

}   // namespace infinicore::op