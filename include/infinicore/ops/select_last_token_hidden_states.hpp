#pragma once

#include "../tensor.hpp"

namespace infinicore::op {

// Select the final token of every packed request.
//
// hidden_states: [batch, tokens, hidden_size]
// input_offsets: [num_requests + 1], I32
// result: [1, num_requests, hidden_size]
Tensor select_last_token_hidden_states(
    const Tensor &hidden_states,
    const Tensor &input_offsets);

} // namespace infinicore::op
