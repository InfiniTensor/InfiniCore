#include "infinicore/ops/rope.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/nn/rope.hpp"
#include <infiniop.h>
#include <stdexcept>

namespace infinicore::op {

namespace {
    // Convert RoPEAlgo to infiniopRoPEAlgo_t
    infiniopRoPEAlgo_t toInfiniopRoPEAlgo(RoPEAlgo algo) {
        switch (algo) {
            case RoPEAlgo::GPT_J:
                return INFINIOP_ROPE_ALGO_GPT_J;
            case RoPEAlgo::GPT_NEOX:
                return INFINIOP_ROPE_ALGO_GPT_NEOX;
            default:
                throw std::runtime_error("Unsupported RoPE algorithm");
        }
    }
} // namespace

// Internal schema that uses infiniop types (for dispatcher storage)
// This matches the actual infiniop function signatures
using infiniop_schema = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, infiniopRoPEAlgo_t);

// Internal dispatcher that stores infiniop functions
// Made accessible to rope_infiniop.cc for registration
namespace infinicore::op {
    common::OpDispatcher<infiniop_schema> &infiniop_dispatcher() {
        static common::OpDispatcher<infiniop_schema> dispatcher_;
        return dispatcher_;
    }
}

common::OpDispatcher<RoPE::schema> &RoPE::dispatcher() {
    // Internal dispatcher uses infiniop types, but we need to provide a public dispatcher
    // with RoPEAlgo. We'll create wrapper functions at registration time.
    // For now, return a static dispatcher - registration should use infiniop_dispatcher directly
    static common::OpDispatcher<RoPE::schema> dispatcher_;
    return dispatcher_;
}


void RoPE::execute(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, RoPEAlgo algo) {
    auto device_type = context::getDevice().getType();
    auto infiniop_func = infiniop_dispatcher().lookup(device_type);

    if (infiniop_func == nullptr) {
        throw std::runtime_error("No RoPE implementation found for device type: " + std::to_string(static_cast<int>(device_type)));
    }

    // Convert RoPEAlgo to infiniopRoPEAlgo_t and call the infiniop function
    infiniop_func(x_out, x, pos, sin_cache, cos_cache, toInfiniopRoPEAlgo(algo));
}

void rope_(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, RoPEAlgo algo) {
    RoPE::execute(x_out, x, pos, sin_cache, cos_cache, algo);
}

Tensor rope(const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, RoPEAlgo algo) {
    Shape shape = x->shape();
    auto x_out = Tensor::empty(shape, x->dtype(), x->device());
    rope_(x_out, x, pos, sin_cache, cos_cache, algo);
    return x_out;
}

} // namespace infinicore::op
