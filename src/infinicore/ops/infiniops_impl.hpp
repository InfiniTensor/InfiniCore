#pragma once

#include "../utils.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "config.h"
#include "data_type.h"
#include "handle.h"
#include "tensor.h"
#include "infini/ops.h"

namespace infini::ops::generated_dispatch {

void CallSigmoidInfinilm(const Handle &handle, const Config &config, Tensor input, Tensor out);
void CallPagedAttentionInfinilm(const Handle &handle, const Config &config, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor seq_lens, std::optional<Tensor> alibi_slopes, float scale, Tensor out);
void CallSwiglu(const Handle &handle, const Config &config, Tensor input, Tensor gate, Tensor out);
void CallAdd(const Handle &handle, const Config &config, Tensor input, Tensor other, Tensor out);
void CallRotaryEmbeddingInfinilm(const Handle &handle, const Config &config, Tensor input, Tensor pos_ids, Tensor sin_table, Tensor cos_table, bool is_neox, Tensor out);
void CallRandomSampleInfinilm(const Handle &handle, const Config &config, Tensor logits, float random_val, float topp, int64_t topk, float temperature, Tensor out);
void CallPagedCachingInfinilm(const Handle &handle, const Config &config, Tensor k, Tensor v, Tensor slot_mapping, Tensor k_cache, Tensor v_cache);
void CallSoftmaxInfinilm(const Handle &handle, const Config &config, Tensor input, int64_t dim, std::optional<DataType> dtype, Tensor out);
void CallPagedAttentionPrefillInfinilm(const Handle &handle, const Config &config, Tensor q, Tensor k_cache, Tensor v_cache, Tensor block_tables, Tensor seq_lens, Tensor cum_seq_lens_q, std::optional<Tensor> alibi_slopes, float scale, Tensor out);
void CallTopksoftmaxInfinilm(const Handle &handle, const Config &config, Tensor input, int64_t topk, bool norm, Tensor values, Tensor indices);
void CallGemm(const Handle &handle, const Config &config, Tensor a, Tensor b, std::optional<float> alpha, std::optional<float> beta, std::optional<int> trans_a, std::optional<int> trans_b, Tensor c);
void CallGemm(const Handle &handle, const Config &config, Tensor a, Tensor b, Tensor c);
void CallGemm(const Handle &handle, const Config &config, Tensor a, Tensor b, std::optional<float> alpha, std::optional<float> beta, Tensor c);
void CallConvInfinilm(const Handle &handle, const Config &config, Tensor input, Tensor weight, std::optional<Tensor> bias, std::vector<int64_t> padding, std::vector<int64_t> stride, std::vector<int64_t> dilation, int64_t groups, Tensor out);
void CallGeluInfinilm(const Handle &handle, const Config &config, Tensor input, std::string approximate, Tensor out);
void CallCausalSoftmax(const Handle &handle, const Config &config, Tensor input, Tensor out);
void CallKvCachingInfinilm(const Handle &handle, const Config &config, Tensor k, Tensor v, Tensor past_kv_lengths, Tensor k_cache, Tensor v_cache);
void CallRearrangeInfinilm(const Handle &handle, const Config &config, Tensor input, Tensor out);
void CallSiluAndMulInfinilm(const Handle &handle, const Config &config, Tensor input, Tensor out);
void CallRmsNorm(const Handle &handle, const Config &config, Tensor input, Tensor weight, float eps, Tensor out);
void CallRmsNorm(const Handle &handle, const Config &config, Tensor input, Tensor weight, Tensor out);
void CallEmbedding(const Handle &handle, const Config &config, Tensor input, Tensor weight, Tensor out);
void CallZerosInfinilm(const Handle &handle, const Config &config, Tensor input, Tensor out);
void CallGelutanhInfinilm(const Handle &handle, const Config &config, Tensor input, Tensor out);
void CallAddRmsNorm(const Handle &handle, const Config &config, Tensor input, Tensor residual, Tensor weight, std::optional<float> eps, Tensor out, Tensor residual_out);
void CallAddRmsNorm(const Handle &handle, const Config &config, Tensor input, Tensor residual, Tensor weight, Tensor out, Tensor residual_out);
void CallSilu(const Handle &handle, const Config &config, Tensor input, Tensor out);
void CallReluInfinilm(const Handle &handle, const Config &config, Tensor input, Tensor out);

} // namespace infini::ops::generated_dispatch

namespace infinicore::op::infiniops {

inline infini::ops::DataType toInfiniOpsDtype(DataType dtype) {
    switch (dtype) {
    case DataType::I8:
        return infini::ops::DataType::kInt8;
    case DataType::I16:
        return infini::ops::DataType::kInt16;
    case DataType::I32:
        return infini::ops::DataType::kInt32;
    case DataType::I64:
        return infini::ops::DataType::kInt64;
    case DataType::U8:
    case DataType::BYTE:
        return infini::ops::DataType::kUInt8;
    case DataType::U16:
        return infini::ops::DataType::kUInt16;
    case DataType::U32:
        return infini::ops::DataType::kUInt32;
    case DataType::U64:
        return infini::ops::DataType::kUInt64;
    case DataType::F16:
        return infini::ops::DataType::kFloat16;
    case DataType::BF16:
        return infini::ops::DataType::kBFloat16;
    case DataType::F32:
        return infini::ops::DataType::kFloat32;
    case DataType::F64:
        return infini::ops::DataType::kFloat64;
    default:
        throw std::runtime_error("InfiniOps backend does not support this tensor dtype.");
    }
}

inline infini::ops::Device toInfiniOpsDevice(const Device &device) {
    INFINICORE_ASSERT(device.getType() == Device::Type::NVIDIA);
    return infini::ops::Device{infini::ops::Device::Type::kNvidia, static_cast<int>(device.getIndex())};
}

struct TensorMeta {
    Shape shape;
    Strides strides;
    infini::ops::DataType dtype;
    infini::ops::Device device;

    explicit TensorMeta(const Tensor &tensor)
        : shape(tensor->shape()),
          strides(tensor->strides()),
          dtype(toInfiniOpsDtype(tensor->dtype())),
          device(toInfiniOpsDevice(tensor->device())) {}

    infini::ops::Tensor tensor(const void *data) const {
        return infini::ops::Tensor(
            const_cast<void *>(data), shape, dtype, device, strides);
    }

    infini::ops::Tensor tensor(const Tensor &tensor) const {
        return this->tensor(tensor->data());
    }
};

} // namespace infinicore::op::infiniops
