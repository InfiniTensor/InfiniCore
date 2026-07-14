#include "infinicore/ops/cast.hpp"
#include "../../utils.hpp"
#include <stdexcept>
#if defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#endif
namespace infinicore::op {
void cast_(Tensor out, const Tensor &in) {
    if (!out || !in || out->shape() != in->shape()) {
        throw std::runtime_error("cast_ expects equal non-empty shapes");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, in);
#if defined(ENABLE_ATEN)
    auto o = adaptor::to_aten_tensor(out), x = adaptor::to_aten_tensor(in);
    o.copy_(x);
    return;
#else
    throw std::runtime_error("cast_ requires ATen");
#endif
}
} // namespace infinicore::op
