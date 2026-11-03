#include "infinicore/ops/linear_allreduce.hpp"
#include "infinicore/ops/add.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"
#include "infinicore/ops/gemm.hpp"
#include "infinicore/ops/linear.hpp"

#include "../../utils.hpp"

#include <stdexcept>

namespace infinicore::op {
#ifdef ENABLE_ASCEND_API
namespace linear_allreduce_impl::ascend {
bool try_record_pipelined_nz(Tensor output, const Tensor &input,
                             const Tensor &weight, infinicclComm_t communicator);
}
#endif

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(LinearAllReduce);

LinearAllReduce::LinearAllReduce(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input, weight);
    if (bias) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, *bias);
    }
    INFINICORE_GRAPH_OP_DISPATCH(
        output->device().getType(), output, input, weight, bias, communicator);
}

void LinearAllReduce::execute(
    Tensor output,
    const Tensor &input,
    const Tensor &weight,
    const std::optional<Tensor> &bias,
    infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        LinearAllReduce, output, input, weight, bias, communicator);
}

enum class PackedLayout {
    NONE,
    ND,
    ASCEND_NZ,
};

static Tensor linear_allreduce_dispatch(
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator,
    PackedLayout packed_layout) {
    const bool weight_is_packed = packed_layout != PackedLayout::NONE;
    Size in_features = weight->shape()[weight_is_packed ? 0 : 1];
    Size out_features = weight->shape()[weight_is_packed ? 1 : 0];
    auto output_shape = input->shape();
    output_shape.back() = out_features;

    if (packed_layout == PackedLayout::ASCEND_NZ) {
        // Plain (non-MC2) path for FRACTAL_NZ weights: dedicated
        // aclnnMatmulWeightNz gemm followed by a plain HCCL all-reduce. This
        // keeps the NZ weight layout (the layout that gave the ND->NZ win on
        // the column-parallel layers) without paying the fused
        // aclnnMatmulAllReduce AI_CPU coordination cost measured on device.
        if (input->device().getType() != Device::Type::ASCEND) {
            throw std::runtime_error(
                "FRACTAL_NZ linear/all-reduce is only supported on Ascend");
        }
        auto output = Tensor::empty(output_shape, input->dtype(),
                                    input->device());
        Size rows = 1;
        for (Size i = 0; i + 1 < input->ndim(); ++i) {
            rows *= input->size(i);
        }
        bool pipelined = false;
#ifdef ENABLE_ASCEND_API
        pipelined = linear_allreduce_impl::ascend::try_record_pipelined_nz(
            output->view({rows, out_features}),
            input->view({rows, in_features}), weight, communicator);
#endif
        if (!pipelined) {
            op::gemm_nz_(output->view({rows, out_features}),
                         input->view({rows, in_features}), weight, 1.0f, 0.0f);
            distributed::allreduce_(output, output, INFINICCL_SUM, communicator);
        }
        if (bias) {
            return add(output, *bias);
        }
        return output;
    }

    auto output = weight_is_packed
                    ? linear_packed(input, weight, std::nullopt)
                    : linear(input, weight, std::nullopt);
    distributed::allreduce_(
        output, output, INFINICCL_SUM, communicator);
    if (bias) {
        return add(output, *bias);
    }
    return output;
}

Tensor linear_allreduce(
    Tensor input,
    Tensor weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    return linear_allreduce_dispatch(
        input, weight, bias, communicator, PackedLayout::NONE);
}

Tensor linear_allreduce_packed(
    Tensor input,
    Tensor packed_weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    return linear_allreduce_dispatch(
        input, packed_weight, bias, communicator, PackedLayout::ND);
}

Tensor linear_allreduce_packed_nz(
    Tensor input,
    Tensor nz_weight,
    std::optional<Tensor> bias,
    infinicclComm_t communicator) {
    return linear_allreduce_dispatch(
        input, nz_weight, bias, communicator, PackedLayout::ASCEND_NZ);
}

} // namespace infinicore::op
