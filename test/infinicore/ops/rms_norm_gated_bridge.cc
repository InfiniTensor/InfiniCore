#include "infinicore/ops/rms_norm_gated.hpp"
#include <cstdio>
extern "C" int test_rms_norm_gated(void *out, void *x, void *gate, void *weight, int tokens, int heads, int xs, int zs, int ys) {
    using namespace infinicore;
    try {
        Device device(Device::Type::ILUVATAR, 0);
        context::setDevice(device);
        auto tensor = [&](void *p, int stride) { return Tensor::strided_from_blob(p, {size_t(tokens), size_t(heads * 128)}, {stride, 1}, DataType::BF16, device); };
        op::rms_norm_gated_(tensor(out, ys), tensor(x, xs), tensor(gate, zs), Tensor::from_blob(weight, {128}, DataType::BF16, device), 1e-6f);
        context::syncStream();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
