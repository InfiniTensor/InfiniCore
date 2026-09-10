#include "infinicore/ops/linear_gguf.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearGguf);

LinearGguf::LinearGguf(Tensor output,
                       const Tensor &input,
                       const Tensor &weight,
                       int64_t ggml_type) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, weight, ggml_type);
}

void LinearGguf::execute(Tensor output,
                         const Tensor &input,
                         const Tensor &weight,
                         int64_t ggml_type) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        LinearGguf, output, input, weight, ggml_type);
}

Tensor linear_gguf(const Tensor &input,
                   const Tensor &weight,
                   int64_t ggml_type) {
    // The decode kernel takes a single [M, K] batch; folding extra leading dims
    // in here is left to the prefill path, so refuse them instead of pretending.
    INFINICORE_ASSERT(input->ndim() == 2);
    INFINICORE_ASSERT(weight->ndim() == 2);
    auto output_shape = input->shape();
    output_shape.back() = weight->size(0);
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    linear_gguf_(output, input, weight, ggml_type);
    return output;
}

void linear_gguf_(Tensor output,
                  const Tensor &input,
                  const Tensor &weight,
                  int64_t ggml_type) {
    LinearGguf::execute(output, input, weight, ggml_type);
}

} // namespace infinicore::op
