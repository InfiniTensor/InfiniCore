#include "infinicore/ops/block_fp8_linear.hpp"

#include "../../utils.hpp"

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(BlockFP8Linear);

BlockFP8Linear::BlockFP8Linear(Tensor output,
                                const Tensor &input,
                                const Tensor &weight,
                                const Tensor &weight_scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(
        output, input, weight, weight_scale);
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, weight, weight_scale);
}

void BlockFP8Linear::execute(Tensor output,
                              const Tensor &input,
                              const Tensor &weight,
                              const Tensor &weight_scale) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        BlockFP8Linear, output, input, weight, weight_scale);
}

Tensor block_fp8_linear(const Tensor &input,
                        const Tensor &weight,
                        const Tensor &weight_scale) {
    INFINICORE_ASSERT(input->ndim() >= 2);
    INFINICORE_ASSERT(weight->ndim() == 2);
    auto output_shape = input->shape();
    output_shape.back() = weight->size(0);
    auto output = Tensor::empty(output_shape, input->dtype(), input->device());
    block_fp8_linear_(output, input, weight, weight_scale);
    return output;
}

void block_fp8_linear_(Tensor output,
                       const Tensor &input,
                       const Tensor &weight,
                       const Tensor &weight_scale) {
    BlockFP8Linear::execute(output, input, weight, weight_scale);
}

} // namespace infinicore::op
