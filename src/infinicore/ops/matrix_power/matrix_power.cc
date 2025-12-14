#include "infinicore/ops/matrix_power.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/tensor.hpp"
#include <stdexcept>

namespace infinicore::op {

common::OpDispatcher<MatrixPower::schema> &MatrixPower::dispatcher() {
    static common::OpDispatcher<MatrixPower::schema> dispatcher_;
    return dispatcher_;
};

void MatrixPower::execute(Tensor output, Tensor input, int n) {
    // For now, implement directly without dispatcher
    // This can be optimized later with device-specific kernels
    Shape shape = input->shape();
    
    if (n == 0) {
        // Identity matrix - should be handled in Python layer
        throw std::runtime_error("n=0 should be handled in Python layer");
    } else if (n == 1) {
        output->copy_from(input);
    } else if (n < 0) {
        throw std::runtime_error("Negative powers not supported");
    } else {
        // Binary exponentiation algorithm
        // Ensure all intermediate tensors are contiguous for correct computation
        Tensor base = input->contiguous();
        Tensor result = Tensor::empty(shape, input->dtype(), input->device());
        result->copy_from(base);  // Start with base (result is now contiguous)
        n--; // We already have one copy
        
        // Use two temporary tensors for alternating computation
        Tensor temp1 = Tensor::empty(shape, input->dtype(), input->device());
        Tensor temp2 = Tensor::empty(shape, input->dtype(), input->device());
        
        while (n > 0) {
            if (n & 1) {
                // If n is odd, multiply result by base: result = result * base
                // Ensure both inputs are contiguous
                Tensor result_cont = result->contiguous();
                Tensor base_cont = base->contiguous();
                matmul_(temp1, result_cont, base_cont);
                // Swap: result now points to temp1's data (which is contiguous)
                std::swap(result, temp1);
            }
            // Square base: base = base * base
            // Ensure base is contiguous
            Tensor base_cont = base->contiguous();
            matmul_(temp2, base_cont, base_cont);
            // Swap: base now points to temp2's data (which is contiguous)
            std::swap(base, temp2);
            n >>= 1; // Divide n by 2
        }
        
        output->copy_from(result);
    }
}

Tensor matrix_power(Tensor input, int n) {
    Shape shape = input->shape();
    if (shape.size() < 2 || shape[shape.size() - 2] != shape[shape.size() - 1]) {
        throw std::runtime_error("matrix_power: input must be a square matrix");
    }
    
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    matrix_power_(output, input, n);
    return output;
}

void matrix_power_(Tensor output, Tensor input, int n) {
    MatrixPower::execute(output, input, n);
}

} // namespace infinicore::op

