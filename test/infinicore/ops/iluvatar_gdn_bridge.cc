// Test-only C ABI for the public C++ operator adaptor.
#include "infinicore/adaptor/iluvatar_gdn_adaptor.hpp"
#include "infinicore/context/context.hpp"
#include <cstdio>
#include <exception>

extern "C" int test_iluvatar_chunk(
    void *q, void *k, void *v, void *g, void *beta, void *state, void *output,
    void *cpu_offsets, void *offsets, void *initial, void *final,
    int tokens, int batch, int pool) {
    using namespace infinicore;
    try {
        const Device device(Device::Type::ILUVATAR, 0);
        context::setDevice(device);
        auto tensor = [&](void *p, const Shape &shape, DataType dtype) {
            return Tensor::from_blob(p, shape, dtype, device);
        };
        const Shape qshape{1, size_t(tokens), 16, 128};
        const Shape vshape{1, size_t(tokens), 32, 128};
        const Shape gate_shape{1, size_t(tokens), 32};
        auto host = Tensor::from_blob(cpu_offsets, {size_t(batch + 1)}, DataType::I32, Device::cpu());
        auto out = adaptor::iluvatar_gdn::chunk(
            tensor(q, qshape, DataType::BF16), tensor(k, qshape, DataType::BF16),
            tensor(v, vshape, DataType::BF16), tensor(g, gate_shape, DataType::BF16),
            tensor(beta, gate_shape, DataType::BF16), tensor(state, {size_t(pool), 32, 128, 128}, DataType::BF16),
            tensor(offsets, {size_t(batch + 1)}, DataType::I32), host,
            tensor(initial, {size_t(batch)}, DataType::I32), tensor(final, {size_t(batch)}, DataType::I32));
        if (!out.has_value()) {
            return 2;
        }
        tensor(output, vshape, DataType::BF16)->copy_from(out.value());
        context::syncStream();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
