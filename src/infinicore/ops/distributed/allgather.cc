#include "infinicore/ops/distributed/allgather.hpp"
#include "../../utils.hpp"

#include "infinicore/context/context.hpp"

#include <algorithm>
#include <utility>

namespace infinicore::op::distributed {
namespace {

bool all_equal_to(const std::vector<size_t> &values, size_t expected) {
    return !values.empty() && std::all_of(values.begin(), values.end(), [&](size_t value) {
        return value == expected;
    });
}

bool can_use_grouped_allgather(const std::vector<Tensor> &inputs,
                               const std::vector<size_t> &split_sizes) {
    return !inputs.empty() && all_equal_to(split_sizes, inputs.front()->shape()[0]) && std::all_of(inputs.begin(), inputs.end(), [&](const Tensor &input) {
        return input && input->ndim() > 0 && input->shape()[0] == split_sizes.front();
    });
}

} // namespace

struct AllGatherPlannedMeta {
    graph::GraphTensor output, input;
    infinicclComm_t communicator;
};

struct AllGatherVPlannedMeta {
    graph::GraphTensor output, input;
    std::vector<size_t> split_counts;
    infinicclComm_t communicator;
};

AllGather::AllGather(Tensor output, const Tensor &input, infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(input->numel() > 0);
    INFINICORE_ASSERT(output->numel() % input->numel() == 0);
    planned_meta_ = new AllGatherPlannedMeta{graph::GraphTensor(output), graph::GraphTensor(input), communicator};
}

AllGather::~AllGather() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<AllGatherPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void AllGather::run() const {
    auto *meta = reinterpret_cast<AllGatherPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclAllGather(meta->input->data(),
                                              meta->output->data(),
                                              meta->input->numel(),
                                              static_cast<infiniDtype_t>(static_cast<int>(meta->input->dtype())),
                                              meta->communicator,
                                              infinicore::context::getStream()));
}

void AllGather::execute(Tensor output, const Tensor &input, infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(AllGather, output, input, communicator);
}

AllGatherV::AllGatherV(Tensor output,
                       const Tensor &input,
                       std::vector<size_t> split_counts,
                       infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(input->numel() > 0);
    size_t total_count = 0;
    for (auto count : split_counts) {
        total_count += count;
    }
    INFINICORE_ASSERT(output->numel() == total_count);
    planned_meta_ = new AllGatherVPlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        std::move(split_counts),
        communicator,
    };
}

AllGatherV::~AllGatherV() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<AllGatherVPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void AllGatherV::run() const {
    auto *meta = reinterpret_cast<AllGatherVPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclAllGatherV(meta->input->data(),
                                               meta->output->data(),
                                               meta->split_counts.data(),
                                               static_cast<int>(meta->split_counts.size()),
                                               static_cast<infiniDtype_t>(static_cast<int>(meta->input->dtype())),
                                               meta->communicator,
                                               infinicore::context::getStream()));
}

void AllGatherV::execute(Tensor output,
                         const Tensor &input,
                         std::vector<size_t> split_counts,
                         infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(AllGatherV, output, input, std::move(split_counts), communicator);
}

Tensor allgather(const Tensor &input, size_t world_size, infinicclComm_t communicator) {
    INFINICORE_ASSERT(input->ndim() > 0);
    auto shape = input->shape();
    shape[0] *= world_size;
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    allgather_(output, input, communicator);
    return output;
}

void allgather_(Tensor output, const Tensor &input, infinicclComm_t communicator) {
    AllGather::execute(output, input, communicator);
}

Tensor allgatherv(const Tensor &input, const std::vector<size_t> &split_sizes, infinicclComm_t communicator) {
    INFINICORE_ASSERT(input->ndim() > 0);
    size_t total_dim0 = 0;
    for (auto size : split_sizes) {
        total_dim0 += size;
    }
    auto shape = input->shape();
    shape[0] = total_dim0;
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    allgatherv_(output, input, split_sizes, communicator);
    return output;
}

void allgatherv_(Tensor output, const Tensor &input, const std::vector<size_t> &split_sizes, infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(input->shape()[0] > 0);
    if (all_equal_to(split_sizes, input->shape()[0])) {
        INFINICORE_ASSERT(output->numel() == input->numel() * split_sizes.size());
        AllGather::execute(output, input, communicator);
        return;
    }

    const size_t inner = input->numel() / input->shape()[0];
    std::vector<size_t> split_counts;
    split_counts.reserve(split_sizes.size());
    size_t total_dim0 = 0;
    for (auto size : split_sizes) {
        split_counts.push_back(size * inner);
        total_dim0 += size;
    }
    INFINICORE_ASSERT(output->numel() == total_dim0 * inner);
    AllGatherV::execute(output, input, std::move(split_counts), communicator);
}

std::vector<Tensor> allgatherv_many(const std::vector<Tensor> &inputs,
                                    const std::vector<size_t> &split_sizes,
                                    infinicclComm_t communicator) {
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    size_t total_dim0 = 0;
    for (auto size : split_sizes) {
        total_dim0 += size;
    }
    for (const auto &input : inputs) {
        INFINICORE_ASSERT(input->ndim() > 0);
        auto shape = input->shape();
        shape[0] = total_dim0;
        outputs.push_back(Tensor::empty(shape, input->dtype(), input->device()));
    }
    allgatherv_many_(outputs, inputs, split_sizes, communicator);
    return outputs;
}

void allgatherv_many_(const std::vector<Tensor> &outputs,
                      const std::vector<Tensor> &inputs,
                      const std::vector<size_t> &split_sizes,
                      infinicclComm_t communicator) {
    INFINICORE_ASSERT(outputs.size() == inputs.size());
    if (inputs.empty()) {
        return;
    }

    const bool use_group = can_use_grouped_allgather(inputs, split_sizes) && !infinicore::context::isGraphRecording();
    if (use_group) {
        INFINICORE_CHECK_ERROR(infinicclGroupStart(communicator));
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        allgatherv_(outputs[i], inputs[i], split_sizes, communicator);
    }
    if (use_group) {
        INFINICORE_CHECK_ERROR(infinicclGroupEnd(communicator));
    }
}

} // namespace infinicore::op::distributed
