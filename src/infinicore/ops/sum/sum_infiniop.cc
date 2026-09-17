#include "../infiniop_impl.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/sum.hpp"

namespace infinicore::op::sum_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Sum, 100);

class RecordedSum final : public graph::GraphOperator {
public:
    RecordedSum(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim)
        : output_(output), input_(input), dim_(std::move(dim)), keepdim_(keepdim) {
        size_t seed = hash_combine(output, input, dim_.size(), keepdim_);
        for (auto axis : dim_) {
            hash_combine(seed, axis);
        }
        INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
            Descriptor, descriptor, Sum, seed,
            output->desc(), input->desc(), dim_.data(), dim_.size(), keepdim_);
        INFINIOP_WORKSPACE_TENSOR(workspace, Sum, descriptor);
        descriptor_ = std::move(descriptor);
        workspace_ = std::make_unique<graph::GraphTensor>(workspace);
    }

    void run() const override {
        auto output = output_;
        INFINICORE_CHECK_ERROR(infiniopSum(
            descriptor_->desc, (*workspace_)->data(), (*workspace_)->numel(),
            output->data(), input_->data(), dim_.data(), dim_.size(), keepdim_, context::getStream()));
    }

private:
    graph::GraphTensor output_, input_;
    mutable std::vector<size_t> dim_;
    bool keepdim_;
    std::shared_ptr<Descriptor> descriptor_;
    std::unique_ptr<graph::GraphTensor> workspace_;
};

void calculate(Tensor output, Tensor input, std::vector<size_t> dim, bool keepdim) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(RecordedSum, output, input, std::move(dim), keepdim);
}

static bool registered = []() {
    Sum::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::sum_impl::infiniop
