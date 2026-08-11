#include "infinicore/ops/select_last_token_hidden_states.hpp"

#include <cstdint>
#include <stdexcept>

namespace infinicore::op {

Tensor select_last_token_hidden_states(
    const Tensor &hidden_states,
    const Tensor &input_offsets) {
    if (input_offsets->dtype() != DataType::I32) {
        throw std::runtime_error(
            "select_last_token_hidden_states: input_offsets must have I32 dtype");
    }
    if (input_offsets->ndim() != 1
        || input_offsets->size(0) < 2) {
        throw std::runtime_error(
            "select_last_token_hidden_states: input_offsets must be a 1D tensor with at least two elements");
    }
    if (hidden_states->ndim() != 3) {
        throw std::runtime_error(
            "select_last_token_hidden_states: expected rank-3 hidden_states");
    }

    const auto num_requests = input_offsets->size(0) - 1;
    const auto hidden_size = hidden_states->size(2);
    const auto total_tokens =
        hidden_states->size(0) * hidden_states->size(1);
    if (total_tokens < num_requests) {
        throw std::runtime_error(
            "select_last_token_hidden_states: more requests than input tokens");
    }
    if (total_tokens == num_requests) {
        return hidden_states;
    }

    auto input_offsets_cpu = input_offsets->to(Device::cpu());
    const auto *offsets = reinterpret_cast<const int32_t *>(
        input_offsets_cpu->data());
    if (offsets[0] != 0
        || offsets[num_requests] < 0
        || static_cast<size_t>(offsets[num_requests])
            != total_tokens) {
        throw std::runtime_error(
            "select_last_token_hidden_states: input_offsets must cover all input tokens");
    }
    for (size_t i = 0; i < num_requests; ++i) {
        const auto begin = offsets[i];
        const auto end = offsets[i + 1];
        if (begin < 0
            || end <= begin
            || static_cast<size_t>(end) > total_tokens) {
            throw std::runtime_error(
                "select_last_token_hidden_states: input_offsets must be strictly increasing and in range");
        }
    }

    auto flat_hidden_states =
        hidden_states->view({total_tokens, hidden_size});
    auto selected_hidden_states = Tensor::empty(
        {1, num_requests, hidden_size},
        hidden_states->dtype(),
        hidden_states->device());
    for (size_t i = 0; i < num_requests; ++i) {
        const auto token_index =
            static_cast<size_t>(offsets[i + 1] - 1);
        selected_hidden_states
            ->narrow({{1, i, 1}})
            ->view({1, hidden_size})
            ->copy_from(
                flat_hidden_states->narrow(
                    {{0, token_index, 1}}));
    }
    return selected_hidden_states;
}

} // namespace infinicore::op
