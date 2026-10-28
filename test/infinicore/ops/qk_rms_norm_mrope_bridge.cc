#include "infinicore/ops/mrope.hpp"
#include "infinicore/ops/qk_rms_norm_mrope.hpp"
#include "infinicore/ops/rms_norm.hpp"
#include <chrono>
#include <cstdio>
extern "C" double test_qk_norm_mrope(void *q_out, void *k_out, void *q_ptr, void *k_ptr, void *qw_ptr, void *kw_ptr,
                                     void *cos_ptr, void *sin_ptr, void *pos_ptr, const int64_t *c, int mode, int loops) {
    using namespace infinicore;
    try {
        Device dev(Device::Type::ILUVATAR, 0);
        context::setDevice(dev);
        const size_t n = c[0], qh = c[1], kh = c[2];
        auto q = Tensor::strided_from_blob(q_ptr, {n, qh, 256}, {c[3], c[4], 1}, DataType::BF16, dev);
        auto k = Tensor::strided_from_blob(k_ptr, {n, kh, 256}, {c[5], c[6], 1}, DataType::BF16, dev);
        auto qo = Tensor::strided_from_blob(q_out, {n, qh, 256}, {c[7], c[8], 1}, DataType::BF16, dev);
        auto ko = Tensor::strided_from_blob(k_out, {n, kh, 256}, {c[9], c[10], 1}, DataType::BF16, dev);
        auto qw = Tensor::from_blob(qw_ptr, {256}, DataType::BF16, dev), kw = Tensor::from_blob(kw_ptr, {256}, DataType::BF16, dev);
        auto cs = Tensor::strided_from_blob(cos_ptr, {size_t(c[11]), 32}, {c[12], 1}, DataType::BF16, dev);
        auto sn = Tensor::strided_from_blob(sin_ptr, {size_t(c[11]), 32}, {c[13], 1}, DataType::BF16, dev);
        auto pos = c[14] ? Tensor::strided_from_blob(pos_ptr, {3, n}, {c[14], c[15]}, DataType::I64, dev)
                         : Tensor::strided_from_blob(pos_ptr, {n}, {c[15]}, DataType::I64, dev);
        auto run = [&] {
            if (mode == 1) {
                auto rq = op::rms_norm(q, qw, 1e-6f), rk = op::rms_norm(k, kw, 1e-6f);
                op::mrope_(qo->view({n, qh * 256}), ko->view({n, kh * 256}), rq->view({n, qh * 256}), rk->view({n, kh * 256}), cs, sn, pos, 256, 64, c[16], c[17], c[18], c[19]);
            } else {
                op::qk_rms_norm_mrope_(qo, ko, q, k, qw, kw, cs, sn, pos, 1e-6f, c[16], c[17], c[18], c[19]);
            }
        };
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
