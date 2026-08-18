#include "infinicore/ops/select_last_token_hidden.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/add.h"
#include "base/index_select.h"

namespace infinicore::op::select_last_token_hidden_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

struct PlannedMeta {
    TensorMeta output, hidden_states, input_offsets, one, indices;
    graph::GraphTensor output_tensor, hidden_states_tensor, input_offsets_tensor, one_tensor, indices_tensor;
};
} // namespace

void *plan(Tensor output, const Tensor &hidden_states, const Tensor &input_offsets) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, hidden_states, input_offsets);

    const auto hidden_size = hidden_states->size(2);
    auto hidden_states_view = hidden_states->view({hidden_states->numel() / hidden_size, hidden_size});
    auto output_view = output->view({output->numel() / hidden_size, hidden_size});
    const auto num_requests = input_offsets->numel() - 1;
    auto input_offsets_view = input_offsets->narrow({{0, 1, num_requests}});
    auto indices = Tensor::empty({num_requests}, DataType::I32, input_offsets->device());
    auto one = Tensor::empty({1}, DataType::I32, input_offsets->device());
    const int32_t one_value = 1;
    context::memcpyH2D(one->data(), &one_value, sizeof(one_value), false);

    return new PlannedMeta{
        TensorMeta(output_view),
        TensorMeta(hidden_states_view),
        TensorMeta(input_offsets_view),
        TensorMeta(one),
        TensorMeta(indices),
        graph::GraphTensor(output_view),
        graph::GraphTensor(hidden_states_view),
        graph::GraphTensor(input_offsets_view),
        graph::GraphTensor(one),
        graph::GraphTensor(indices)};
}

void run(void *planned_meta) {
    auto *planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config add_config;

    infini::ops::Add::Call(
        handle,
        add_config,
        planned->input_offsets.tensor(planned->input_offsets_tensor),
        planned->one.tensor(planned->one_tensor),
        -1.0,
        planned->indices.tensor(planned->indices_tensor));
    infini::ops::Config index_select_config;
    index_select_config.set_implementation_index(8);
    infini::ops::IndexSelect::Call(
        handle,
        index_select_config,
        planned->hidden_states.tensor(planned->hidden_states_tensor),
        planned->indices.tensor(planned->indices_tensor),
        int64_t{0},
        planned->output.tensor(planned->output_tensor));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    SelectLastTokenHidden::plan_dispatcher().registerDevice(Device::Type::NVIDIA, &plan);
    SelectLastTokenHidden::run_dispatcher().registerDevice(Device::Type::NVIDIA, &run);
    SelectLastTokenHidden::cleanup_dispatcher().registerDevice(Device::Type::NVIDIA, &cleanup);
    SelectLastTokenHidden::plan_dispatcher().registerDevice(Device::Type::METAX, &plan);
    SelectLastTokenHidden::run_dispatcher().registerDevice(Device::Type::METAX, &run);
    SelectLastTokenHidden::cleanup_dispatcher().registerDevice(Device::Type::METAX, &cleanup);
    return true;
}();

} // namespace infinicore::op::select_last_token_hidden_impl::infiniops
#endif
