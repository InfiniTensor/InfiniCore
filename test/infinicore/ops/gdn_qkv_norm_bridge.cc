#include "infinicore/ops/gdn_qkv_norm.hpp"
#include <chrono>
#include <cstdio>
extern "C" double test_gdn_pack(void *qo, void *ko, void *vo, void *qp, void *kp, void *vp, const int64_t *c, int mode, int loops) {
    using namespace infinicore;
    try {
        Device dev(Device::Type::ILUVATAR, 0);
        context::setDevice(dev);
        const size_t n = c[0];
        auto q = Tensor::strided_from_blob(qp, {1, n, 16, 128}, {c[1] * static_cast<int64_t>(n), c[1], c[2], 1}, DataType::BF16, dev);
        auto k = Tensor::strided_from_blob(kp, {1, n, 16, 128}, {c[3] * static_cast<int64_t>(n), c[3], c[4], 1}, DataType::BF16, dev);
        auto v = Tensor::strided_from_blob(vp, {1, n, 32, 128}, {c[5] * static_cast<int64_t>(n), c[5], c[6], 1}, DataType::BF16, dev);
        auto q_out = Tensor::from_blob(qo, {1, n, 16, 128}, DataType::BF16, dev);
        auto k_out = Tensor::from_blob(ko, {1, n, 16, 128}, DataType::BF16, dev);
        auto v_out = Tensor::from_blob(vo, {1, n, 32, 128}, DataType::BF16, dev);
        auto run = [&] { op::gdn_qkv_norm_(q_out, k_out, v_out, q, k, v); };
        run();
        context::syncStream();
        std::shared_ptr<graph::Graph> graph;
        if (mode == 2) {
            context::startGraphRecording();
            run();
            graph = context::stopGraphRecording();
        }
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < loops; ++i) {
            if (graph) {
                graph->run();
            } else {
                run();
            }
        }
        context::syncStream();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / loops;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return -1;
    }
}
