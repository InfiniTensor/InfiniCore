#pragma once
#include "infinicore/tensor.hpp"
#include <optional>

namespace infinicore::adaptor::iluvatar_gdn {
// Prefill-only operator bridge. CPU offsets are supplied by the scheduler;
// the bridge never reads device metadata back to the host.
std::optional<Tensor> chunk(
    const Tensor &q, const Tensor &k, const Tensor &v,
    const Tensor &g, const Tensor &beta, Tensor state,
    const Tensor &offsets, const Tensor &cpu_offsets,
    const Tensor &initial_indices, const Tensor &final_indices);
} // namespace infinicore::adaptor::iluvatar_gdn
