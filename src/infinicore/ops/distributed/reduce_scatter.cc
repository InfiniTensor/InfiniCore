#include "infinicore/ops/distributed/reduce_scatter.hpp"
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

bool can_use_grouped_reduce_scatter(const std::vector<Tensor> &outputs,
                                    const std::vector<size_t> &split_sizes) {
    return !outputs.empty() && all_equal_to(split_sizes, outputs.front()->shape()[0]) && std::all_of(outputs.begin(), outputs.end(), [&](const Tensor &output) {
        return output && output->ndim() > 0 && output->shape()[0] == split_sizes.front();
    });
}

} // namespace

struct ReduceScatterPlannedMeta {
    graph::GraphTensor output, input;
    infinicclReduceOp_t op;
    infinicclComm_t communicator;
};

struct ReduceScatterVPlannedMeta {
    graph::GraphTensor output, input;
    std::vector<size_t> split_counts;
    infinicclReduceOp_t op;
    infinicclComm_t communicator;
};

ReduceScatter::ReduceScatter(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(output->numel() > 0);
    INFINICORE_ASSERT(input->numel() % output->numel() == 0);
    planned_meta_ = new ReduceScatterPlannedMeta{graph::GraphTensor(output), graph::GraphTensor(input), op, communicator};
}

ReduceScatter::~ReduceScatter() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<ReduceScatterPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void ReduceScatter::run() const {
    auto *meta = reinterpret_cast<ReduceScatterPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclReduceScatter(meta->input->data(),
                                                  meta->output->data(),
                                                  meta->output->numel(),
                                                  static_cast<infiniDtype_t>(static_cast<int>(meta->input->dtype())),
                                                  meta->op,
                                                  meta->communicator,
                                                  infinicore::context::getStream()));
}

void ReduceScatter::execute(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(ReduceScatter, output, input, op, communicator);
}

ReduceScatterV::ReduceScatterV(Tensor output,
                               const Tensor &input,
                               std::vector<size_t> split_counts,
                               infinicclReduceOp_t op,
                               infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(output->numel() > 0);
    size_t total_count = 0;
    for (auto count : split_counts) {
        total_count += count;
    }
    INFINICORE_ASSERT(input->numel() == total_count);
    planned_meta_ = new ReduceScatterVPlannedMeta{
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        std::move(split_counts),
        op,
        communicator,
    };
}

ReduceScatterV::~ReduceScatterV() {
    if (planned_meta_) {
        auto *meta = reinterpret_cast<ReduceScatterVPlannedMeta *>(planned_meta_);
        delete meta;
    }
}

void ReduceScatterV::run() const {
    auto *meta = reinterpret_cast<ReduceScatterVPlannedMeta *>(planned_meta_);
    INFINICORE_CHECK_ERROR(infinicclReduceScatterV(meta->input->data(),
                                                   meta->output->data(),
                                                   meta->split_counts.data(),
                                                   static_cast<int>(meta->split_counts.size()),
                                                   static_cast<infiniDtype_t>(static_cast<int>(meta->input->dtype())),
                                                   meta->op,
                                                   meta->communicator,
                                                   infinicore::context::getStream()));
}

void ReduceScatterV::execute(Tensor output,
                             const Tensor &input,
                             std::vector<size_t> split_counts,
                             infinicclReduceOp_t op,
                             infinicclComm_t communicator) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(ReduceScatterV, output, input, std::move(split_counts), op, communicator);
}

Tensor reduce_scatter(const Tensor &input, size_t world_size, infinicclReduceOp_t op, infinicclComm_t communicator) {
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(input->shape()[0] % world_size == 0);
    auto shape = input->shape();
    shape[0] /= world_size;
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    reduce_scatter_(output, input, op, communicator);
    return output;
}

void reduce_scatter_(Tensor output, const Tensor &input, infinicclReduceOp_t op, infinicclComm_t communicator) {
    ReduceScatter::execute(output, input, op, communicator);
}

Tensor reduce_scatterv(const Tensor &input,
                       const std::vector<size_t> &split_sizes,
                       size_t rank,
                       infinicclReduceOp_t op,
                       infinicclComm_t communicator) {
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(rank < split_sizes.size());
    auto shape = input->shape();
    shape[0] = split_sizes[rank];
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    reduce_scatterv_(output, input, split_sizes, op, communicator);
    return output;
}

void reduce_scatterv_(Tensor output,
                      const Tensor &input,
                      const std::vector<size_t> &split_sizes,
                      infinicclReduceOp_t op,
                      infinicclComm_t communicator) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);
    INFINICORE_ASSERT(output->is_contiguous() && input->is_contiguous());
    INFINICORE_ASSERT(input->ndim() > 0);
    INFINICORE_ASSERT(input->shape()[0] > 0);
    if (all_equal_to(split_sizes, output->shape()[0])) {
        INFINICORE_ASSERT(input->numel() == output->numel() * split_sizes.size());
        ReduceScatter::execute(output, input, op, communicator);
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
    INFINICORE_ASSERT(input->numel() == total_dim0 * inner);
    ReduceScatterV::execute(output, input, std::move(split_counts), op, communicator);
}

std::vector<Tensor> reduce_scatterv_many(const std::vector<Tensor> &inputs,
                                         const std::vector<size_t> &split_sizes,
                                         size_t rank,
                                         infinicclReduceOp_t op,
                                         infinicclComm_t communicator) {
    INFINICORE_ASSERT(rank < split_sizes.size());
    std::vector<Tensor> outputs;
    outputs.reserve(inputs.size());
    for (const auto &input : inputs) {
        INFINICORE_ASSERT(input->ndim() > 0);
        auto shape = input->shape();
        shape[0] = split_sizes[rank];
        outputs.push_back(Tensor::empty(shape, input->dtype(), input->device()));
    }
    reduce_scatterv_many_(outputs, inputs, split_sizes, op, communicator);
    return outputs;
}

void reduce_scatterv_many_(const std::vector<Tensor> &outputs,
                           const std::vector<Tensor> &inputs,
                           const std::vector<size_t> &split_sizes,
                           infinicclReduceOp_t op,
                           infinicclComm_t communicator) {
    INFINICORE_ASSERT(outputs.size() == inputs.size());
    if (inputs.empty()) {
        return;
    }

    const bool use_group = can_use_grouped_reduce_scatter(outputs, split_sizes) && !infinicore::context::isGraphRecording();
    if (use_group) {
        INFINICORE_CHECK_ERROR(infinicclGroupStart(communicator));
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        reduce_scatterv_(outputs[i], inputs[i], split_sizes, op, communicator);
    }
    if (use_group) {
        INFINICORE_CHECK_ERROR(infinicclGroupEnd(communicator));
    }
}

} // namespace infinicore::op::distributed
