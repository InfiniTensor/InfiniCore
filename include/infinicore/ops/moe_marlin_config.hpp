#pragma once

#include "../device.hpp"
#include "../dtype.hpp"

#include <cstddef>

namespace infinicore::op {

struct HygonMarlinGemmConfig {
    int mode = 103;
    int delta = 1;
    size_t block_size_m = 16;
    bool found = false;
};

struct HygonW16A16MarlinRuntimeConfig {
    HygonMarlinGemmConfig gemm1;
    HygonMarlinGemmConfig gemm2;
    bool supported = false;
};

struct HygonW8A8MarlinRuntimeConfig {
    HygonMarlinGemmConfig gemm1;
    HygonMarlinGemmConfig gemm2;
    bool supported = false;
};

HygonW16A16MarlinRuntimeConfig select_hygon_w16a16_marlin_config(
    size_t num_tokens,
    size_t hidden_size,
    size_t intermediate_size,
    DataType hidden_dtype,
    size_t device_index);

HygonW8A8MarlinRuntimeConfig select_hygon_w8a8_marlin_config(
    size_t num_tokens,
    size_t hidden_size,
    size_t intermediate_size,
    size_t device_index);

} // namespace infinicore::op
