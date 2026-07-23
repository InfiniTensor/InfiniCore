#include "infinicore/ops/select_last_token_hidden.hpp"

#include "../../utils.hpp"
#include <stdexcept>

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SelectLastTokenHidden);

SelectLastTokenHidden::SelectLastTokenHidden(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &input_offsets) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, hidden_states, input_offsets);
    INFINICORE_GRAPH_OP_DISPATCH(output->device().getType(), output, hidden_states, input_offsets);
}

void SelectLastTokenHidden::execute(
    Tensor output,
    const Tensor &hidden_states,
    const Tensor &input_offsets) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SelectLastTokenHidden, output, hidden_states, input_offsets);
}

void select_last_token_hidden_(Tensor output,
                               const Tensor &hidden_states,
                               const Tensor &input_offsets) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, hidden_states, input_offsets);
    if (hidden_states->ndim() != 3 || output->ndim() != 3 || input_offsets->ndim() != 1) {
        throw std::runtime_error(
            "select_last_token_hidden expects 3D hidden/output and 1D offsets");
    }
    if (input_offsets->dtype() != DataType::I32) {
        throw std::runtime_error("select_last_token_hidden expects int32 offsets");
    }
    if (output->dtype() != hidden_states->dtype()) {
        throw std::runtime_error("select_last_token_hidden output dtype mismatch");
    }
    if (input_offsets->numel() < 2) {
        throw std::runtime_error("select_last_token_hidden requires at least one request");
    }
    const size_t num_requests = input_offsets->numel() - 1;
    if (output->size(0) != 1
        || output->size(1) != num_requests
        || output->size(2) != hidden_states->size(2)) {
        throw std::runtime_error("select_last_token_hidden shape mismatch");
    }
    if (!output->is_contiguous() || !hidden_states->is_contiguous() || !input_offsets->is_contiguous()) {
        throw std::runtime_error("select_last_token_hidden expects contiguous tensors");
    }
    SelectLastTokenHidden::execute(output, hidden_states, input_offsets);
}

} // namespace infinicore::op
