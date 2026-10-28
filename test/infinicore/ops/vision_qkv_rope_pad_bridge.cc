#include "infinicore/ops/cat.hpp"
#include "infinicore/ops/mrope.hpp"
#include "infinicore/ops/vision_qkv_rope_pad.hpp"
#include <chrono>
#include <cstdio>
extern "C" double test_vision_pack(void *qo_ptr, void *ko_ptr, void *vo_ptr, void *qkv_ptr,
                                   void *cos_ptr, void *sin_ptr, void *rows_ptr, void *cols_ptr,
                                   const int64_t *c, int mode, int loops) {
    using namespace infinicore;
    try {
        Device dev(Device::Type::ILUVATAR, 0);
        context::setDevice(dev);
        const size_t n = c[0];
        auto qkv = Tensor::strided_from_blob(qkv_ptr, {n, 3, 16, 72}, {c[1], c[2], c[3], 1}, DataType::BF16, dev);
        auto qo = Tensor::strided_from_blob(qo_ptr, {n, 16, 128}, {c[8], 128, 1}, DataType::BF16, dev);
        auto ko = Tensor::strided_from_blob(ko_ptr, {n, 16, 128}, {c[9], 128, 1}, DataType::BF16, dev);
        auto vo = Tensor::strided_from_blob(vo_ptr, {n, 16, 128}, {c[10], 128, 1}, DataType::BF16, dev);
        auto cs = Tensor::strided_from_blob(cos_ptr, {256, 72}, {c[4], 1}, DataType::BF16, dev);
        auto sn = Tensor::strided_from_blob(sin_ptr, {256, 72}, {c[5], 1}, DataType::BF16, dev);
        auto rows = Tensor::strided_from_blob(rows_ptr, {n}, {c[6]}, DataType::I64, dev);
        auto cols = Tensor::strided_from_blob(cols_ptr, {n}, {c[7]}, DataType::I64, dev);
        auto run = [&] {
            op::vision_qkv_rope_pad_(qo, ko, vo, qkv, cs, sn, rows, cols);
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
