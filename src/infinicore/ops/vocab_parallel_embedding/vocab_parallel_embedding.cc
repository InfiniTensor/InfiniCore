#include "infinicore/ops/vocab_parallel_embedding.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include <ATen/ATen.h>
#endif
namespace infinicore::op {
void vocab_parallel_embedding_(Tensor out, const Tensor &ids, const Tensor &w, int64_t start, int64_t end) {
    if (!out || !ids || !w || w->ndim() != 2) {
        throw std::runtime_error("vocab_parallel_embedding_: invalid tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, ids, w);
#if defined(ENABLE_ATEN)
    auto o = adaptor::to_aten_tensor(out), i = adaptor::to_aten_tensor(ids), wt = adaptor::to_aten_tensor(w);
    auto mask = i.lt(start).logical_or(i.ge(end));
    auto local = (i - start).clamp(0, end - start - 1);
    auto y = at::embedding(wt, local, -1, false, false);
    y.masked_fill_(mask.unsqueeze(-1), 0);
    o.copy_(y);
    return;
#else
    throw std::runtime_error("vocab_parallel_embedding_ requires ATen");
#endif
}
} // namespace infinicore::op
