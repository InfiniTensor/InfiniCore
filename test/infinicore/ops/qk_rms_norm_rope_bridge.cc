#include "infinicore/ops/qk_rms_norm_rope.hpp"
#include <cstdio>
extern "C" int test_qk_norm_rope(void *q_out, void *k_out, void *packed, void *qw, void *kw, void *cos, void *sin, void *positions, int tokens, int q_heads, int k_heads, int stride, int table_rows, int output_padding, int cos_stride, int sin_stride, int pos_stride) {
    using namespace infinicore;
    try {
        Device device(Device::Type::ILUVATAR, 0);
        context::setDevice(device);
        const size_t q_width = q_heads * 128, k_width = k_heads * 128;
        auto q = Tensor::strided_from_blob(packed, {size_t(tokens), q_width}, {stride, 1}, DataType::BF16, device);
        auto k = Tensor::strided_from_blob(static_cast<char *>(packed) + q_width * 2, {size_t(tokens), k_width}, {stride, 1}, DataType::BF16, device);
        auto qo = Tensor::strided_from_blob(q_out, {size_t(tokens), q_width}, {ptrdiff_t(q_width + output_padding), 1}, DataType::BF16, device);
        auto ko = Tensor::strided_from_blob(k_out, {size_t(tokens), k_width}, {ptrdiff_t(k_width + output_padding), 1}, DataType::BF16, device);
        auto tensor = [&](void *p, Shape shape, DataType dtype) { return Tensor::from_blob(p, shape, dtype, device); };
        op::qk_rms_norm_rope_(qo, ko, q, k, tensor(qw, {128}, DataType::BF16), tensor(kw, {128}, DataType::BF16),
                              Tensor::strided_from_blob(cos, {size_t(table_rows), 64}, {cos_stride, 1}, DataType::BF16, device), Tensor::strided_from_blob(sin, {size_t(table_rows), 64}, {sin_stride, 1}, DataType::BF16, device), Tensor::strided_from_blob(positions, {size_t(tokens)}, {pos_stride}, DataType::I64, device), 1e-6f);
        context::syncStream();
        return 0;
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
