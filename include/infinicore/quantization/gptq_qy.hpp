#pragma once

#include "../tensor.hpp"
#include "base_quantization.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <vector>

namespace {
#ifndef INFINICORE_FLOAT16_DEFINED
#define INFINICORE_FLOAT16_DEFINED
struct float16_raw {
    uint16_t data;
    float16_raw() : data(0) {}
    explicit float16_raw(float f) : data(fp32_to_fp16_bits(f)) {}

    static uint16_t fp32_to_fp16_bits(float value) {
        union {
            float f;
            uint32_t u;
        } f2u;
        f2u.f = value;
        uint32_t x = f2u.u;

        uint32_t sign = (x >> 16) & 0x8000;
        int32_t exp = ((x >> 23) & 0xFF) - 127;
        uint32_t mantissa = x & 0x007FFFFF;

        if (exp == 128) {
            if (mantissa == 0) {
                return static_cast<uint16_t>(sign | 0x7C00);
            } else {
                return static_cast<uint16_t>(sign | 0x7C00 | (mantissa >> 13));
            }
        }
        if (exp > 15) {
            return static_cast<uint16_t>(sign | 0x7C00);
        }
        if (exp < -14) {
            if (exp < -24) {
                return static_cast<uint16_t>(sign);
            }
            mantissa |= 0x00800000;
            uint32_t shift = -exp - 14;
            mantissa >>= shift;
            if ((mantissa & 0x1000) && ((mantissa & 0x2FFF) != 0)) {
                mantissa += 0x2000;
            }
            return static_cast<uint16_t>(sign | (mantissa >> 13));
        }

        uint32_t exp16 = static_cast<uint32_t>(exp + 15) << 10;
        uint32_t mantissa16 = mantissa >> 13;
        if ((mantissa & 0x1000) && ((mantissa & 0x2FFF) || (mantissa16 & 1))) {
            mantissa16++;
            if (mantissa16 == 0x400) {
                exp16 += 0x400;
                mantissa16 = 0;
            }
        }
        return static_cast<uint16_t>(sign | exp16 | mantissa16);
    }
};
using float16_t = float16_raw;
#endif

inline std::vector<uint16_t> float_to_fp16_bits(const std::vector<float> &values) {
    std::vector<uint16_t> result;
    result.reserve(values.size());
    for (float f : values) {
#ifdef INFINICORE_HAS_FLOAT16
        infinicore::float16_t h(f);
        result.push_back(*reinterpret_cast<uint16_t *>(&h));
#else
        result.push_back(float16_raw::fp32_to_fp16_bits(f));
#endif
    }
    return result;
}
} // anonymous namespace

namespace infinicore::quantization {

class GPTQ_QY : public BaseQuantization {
public:
    explicit GPTQ_QY(const nlohmann::json &quant_config)
        : BaseQuantization(quant_config) {
        int bits = weight_bits();
        if (bits != 4) {
            spdlog::warn("GPTQ_QY: bits={} not fully tested, expected 4", bits);
        }
    }

    QuantScheme get_quant_scheme() const override {
        return QuantScheme::GPTQ_W4A16_QY;
    }

    int get_packing_num() const {
        return 32 / weight_bits();
    }

    int get_group_size() const {
        return get_or<int>("group_size", 128);
    }

    void convert_from_gptq_w4a16(const Tensor &original_qweight,
                                 const Tensor &original_qzeros,
                                 const Tensor &original_scales,
                                 const Tensor &g_idx,
                                 const Device &target_device) {
        if (converted_) {
            spdlog::debug("GPTQ_QY: weights already converted, skipping");
            return;
        }

        const int bits = weight_bits();
        const int values_per_int32 = 32 / bits;

        {
            const auto &shape = original_qweight->shape();
            assert(shape.size() == 2);
            size_t M = shape[0], N = shape[1];

            auto weight_unpacked = unpack_int32_to_nibbles_3d_(original_qweight, bits);
            auto weight_packed = combine_nibbles_last_dim_(weight_unpacked, M, values_per_int32, N);

            size_t dimY = N;
            size_t total_bytes = M * values_per_int32 * (N / 2);
            size_t dimX = total_bytes / dimY;

            assert(dimX * dimY == total_bytes && "Weight shape calculation mismatch");

            converted_weight_ = make_tensor_from_host_(
                weight_packed.data(),
                total_bytes * sizeof(uint8_t),
                {dimX, dimY},
                DataType::U8,
                target_device);
        }

        {
            const auto &shape = original_qzeros->shape();
            assert(shape.size() == 2);
            size_t P = shape[0], Q = shape[1];

            auto zeros_fp32 = unpack_zeros_to_fp32_2d_(original_qzeros, bits);
            auto zeros_fp16 = ::float_to_fp16_bits(zeros_fp32);

            converted_zeros_ = make_tensor_from_host_(
                zeros_fp16.data(),
                zeros_fp16.size() * sizeof(uint16_t),
                {P, Q * static_cast<size_t>(values_per_int32)},
                DataType::F16,
                target_device);
        }

        {
            auto scales_cpu = original_scales->to(Device::Type::CPU);
            size_t num_elements = scales_cpu->numel();
            const void *raw_data = scales_cpu->data();

            std::vector<uint16_t> scales_fp16(num_elements);
            if (scales_cpu->dtype() == DataType::F16) {
                std::memcpy(scales_fp16.data(), raw_data, num_elements * sizeof(uint16_t));
            } else if (scales_cpu->dtype() == DataType::F32) {
                std::vector<float> scales_fp32(num_elements);
                std::memcpy(scales_fp32.data(), raw_data, num_elements * sizeof(float));
                scales_fp16 = ::float_to_fp16_bits(scales_fp32);
            } else {
                spdlog::error("Unsupported scales dtype, expected F16 or F32");
                assert(false && "Unsupported scales dtype");
            }

            converted_scales_ = make_tensor_from_host_(
                scales_fp16.data(),
                scales_fp16.size() * sizeof(uint16_t),
                original_scales->shape(),
                DataType::F16,
                target_device);
        }

        if (g_idx->numel() > 0) {
            g_idx_ = g_idx->to(target_device);
        }

        converted_ = true;
    }

    void release_buffers() {
        converted_weight_ = Tensor();
        converted_zeros_ = Tensor();
        converted_scales_ = Tensor();
        g_idx_ = Tensor();
    }

    void convert_and_take_ownership(Tensor &weight, Tensor &zeros, Tensor &scales,
                                    const Tensor &g_idx, const Device &target_device) {
        if (converted_) {
            spdlog::warn("GPTQ_QY: Already converted, skipping");
            return;
        }

        convert_from_gptq_w4a16(weight, zeros, scales, g_idx, target_device);

        weight = std::move(converted_weight_);
        zeros = std::move(converted_zeros_);
        scales = std::move(converted_scales_);

        converted_ = false;
        spdlog::debug("GPTQ_QY: Ownership transferred, internal buffers cleared.");
    }

    const Tensor &get_converted_weight() const { return std::move(converted_weight_); }
    const Tensor &get_converted_zeros() const { return std::move(converted_zeros_); }
    const Tensor &get_converted_scales() const { return std::move(converted_scales_); }
    const Tensor &get_g_idx() const { return g_idx_; }
    bool is_converted() const { return converted_; }

    int weight_bits() const { return get_or<int>("bits", 4); }
    bool desc_act() const { return get_or<bool>("desc_act", false); }

private:
    static inline std::vector<uint8_t> unpack_int32_to_nibbles_3d_(const Tensor &packed, int bits) {
        assert(bits == 4 || bits == 8);
        const int values_per_int32 = 32 / bits;

        auto packed_cpu = packed->to(Device::Type::CPU);
        const int32_t *packed_host = reinterpret_cast<const int32_t *>(packed_cpu->data());

        const auto &shape = packed->shape();
        assert(shape.size() == 2);
        size_t M = shape[0], N = shape[1];

        std::vector<uint8_t> unpacked(M * values_per_int32 * N);

        for (size_t i = 0; i < M; ++i) {
            for (int k = 0; k < values_per_int32; ++k) {
                for (size_t j = 0; j < N; ++j) {
                    int32_t val = packed_host[i * N + j];
                    uint8_t extracted = static_cast<uint8_t>((val >> (k * bits)) & ((1 << bits) - 1));
                    size_t idx = i * (values_per_int32 * N) + k * N + j;
                    unpacked[idx] = extracted;
                }
            }
        }
        return unpacked;
    }

    static inline std::vector<uint8_t> combine_nibbles_last_dim_(
        const std::vector<uint8_t> &nibbles, size_t M, size_t K, size_t N) {
        assert(N % 2 == 0 && "Last dimension must be even for nibble pairing");

        std::vector<uint8_t> combined(M * K * (N / 2));
        size_t out_idx = 0;

        for (size_t i = 0; i < M; ++i) {
            for (size_t k = 0; k < K; ++k) {
                size_t row_base = i * (K * N) + k * N;
                for (size_t j = 0; j < N; j += 2) {
                    uint8_t low = nibbles[row_base + j] & 0x0F;
                    uint8_t high = nibbles[row_base + j + 1] & 0x0F;
                    combined[out_idx++] = static_cast<uint8_t>((high << 4) | low);
                }
            }
        }
        return combined;
    }

    static inline std::vector<float> unpack_zeros_to_fp32_2d_(const Tensor &packed_zeros, int bits) {
        assert(bits == 4 || bits == 8);
        const int values_per_int32 = 32 / bits;
        const int mask = (1 << bits) - 1;

        auto packed_cpu = packed_zeros->to(Device::Type::CPU);
        const int32_t *packed_host = reinterpret_cast<const int32_t *>(packed_cpu->data());

        const auto &shape = packed_zeros->shape();
        assert(shape.size() == 2);
        size_t P = shape[0], Q = shape[1];

        std::vector<float> result(P * Q * values_per_int32);
        size_t out_idx = 0;

        for (size_t p = 0; p < P; ++p) {
            for (size_t q = 0; q < Q; ++q) {
                int32_t val = packed_host[p * Q + q];
                for (int k = 0; k < values_per_int32; ++k) {
                    uint8_t extracted = static_cast<uint8_t>((val >> (k * bits)) & mask);
                    int dequant_val = (static_cast<int>(extracted) + 1) & mask;
                    result[out_idx++] = static_cast<float>(dequant_val);
                }
            }
        }
        return result;
    }

    static inline Tensor make_tensor_from_host_(const void *data, size_t bytes,
                                                const std::vector<size_t> &shape,
                                                DataType dtype, const Device &device) {
        auto tensor = Tensor::empty(shape, dtype, Device::Type::CPU);
        std::memcpy(reinterpret_cast<void *>(tensor->data()), data, bytes);

        if (device != Device::Type::CPU) {
            return tensor->to(device);
        }
        return tensor;
    }

    Tensor converted_weight_;
    Tensor converted_zeros_;
    Tensor converted_scales_;
    Tensor g_idx_;
    bool converted_ = false;
};

} // namespace infinicore::quantization
