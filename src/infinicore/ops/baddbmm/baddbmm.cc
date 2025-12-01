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

    Tensor result = Tensor::empty({batch_size, m, n}, batch1->dtype(), batch1->device());
    
    Tensor input_cont = input->is_contiguous() ? input : input->contiguous();
    if (input->ndim() == 2) {
        Tensor input_broadcast = input_cont->as_strided(
            {batch_size, m, n}, 
            {0, input_cont->strides()[0], input_cont->strides()[1]});
        result->copy_from(input_broadcast);
    } else if (input->ndim() == 3 && input->shape()[0] == 1 && batch_size > 1) {
        Tensor input_broadcast = input_cont->as_strided(
            {batch_size, m, n}, 
            {0, input_cont->strides()[1], input_cont->strides()[2]});
        result->copy_from(input_broadcast);
    } else {
        result->copy_from(input_cont);
    }

    // Fused operation: result = alpha * batch1 @ batch2 + beta * result
    gemm_(result, batch1_cont, batch2_cont, alpha, beta);

    return result;
}

void baddbmm_(Tensor out, Tensor input, Tensor batch1, Tensor batch2, 
                 float beta, 
                 float alpha) {
    Tensor result = baddbmm(input, batch1, batch2, beta, alpha);
    out->copy_from(result);
}

}   // namespace infinicore::op