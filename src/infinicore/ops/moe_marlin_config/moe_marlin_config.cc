#include "infinicore/ops/moe_marlin_config.hpp"

#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

namespace infinicore::op {
namespace {

enum class HygonMarlinModePolicy {
    LegacyOnly,
    LegacyAndBf16Mode1000,
    All,
};

std::string lightop_config_dir() {
    const char *value = std::getenv("INFINICORE_LIGHTOP_CONFIG_DIR");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }

    // Keep the old override working while ownership moves from InfiniLM.
    value = std::getenv("INFINILM_LIGHTOP_CONFIG_DIR");
    if (value != nullptr && value[0] != '\0') {
        return value;
    }
    return "/usr/local/lib/python3.10/dist-packages/lightop/configs";
}

std::string normalize_hygon_gpu_target(std::string target, bool uppercase) {
    const auto feature_pos = target.find(':');
    if (feature_pos != std::string::npos) {
        target.resize(feature_pos);
    }
    std::transform(target.begin(), target.end(), target.begin(), [uppercase](unsigned char ch) {
        return static_cast<char>(uppercase ? std::toupper(ch) : std::tolower(ch));
    });

    std::string lowercase = target;
    std::transform(lowercase.begin(), lowercase.end(), lowercase.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (lowercase.size() <= 3
        || lowercase.compare(0, 3, "gfx") != 0
        || !std::all_of(lowercase.begin() + 3, lowercase.end(), [](unsigned char ch) {
               return std::isalnum(ch) != 0;
           })) {
        throw std::runtime_error("Invalid Hygon GPU target for LightOP config: " + target);
    }
    return target;
}

HygonMarlinGemmConfig load_lightop_marlin_config(
    size_t n,
    size_t k,
    size_t m,
    const std::string &file_prefix,
    const adaptor::lightop::DeviceInfo &device_info,
    HygonMarlinModePolicy mode_policy,
    bool uppercase_device_name,
    bool num_cus_with_cu_prefix) {
    HygonMarlinGemmConfig result;
    if (device_info.gpu_target.empty() || device_info.compute_units <= 0) {
        throw std::runtime_error("Unable to query Hygon device properties for LightOP config");
    }

    const std::string device_name = normalize_hygon_gpu_target(
        device_info.gpu_target,
        uppercase_device_name);
    const std::string num_cus = std::to_string(device_info.compute_units);
    const std::string num_cus_suffix =
        num_cus_with_cu_prefix ? ("_CU" + num_cus) : ("_" + num_cus);
    const std::string file_name =
        lightop_config_dir() + "/" + file_prefix + "_"
        + std::to_string(n) + "_" + std::to_string(k) + "_"
        + device_name + num_cus_suffix + ".json";

    std::ifstream file(file_name);
    if (!file.is_open()) {
        return result;
    }

    nlohmann::json config_json;
    file >> config_json;
    const std::string shape_key = std::to_string(n) + "_" + std::to_string(k);
    if (!config_json.contains(shape_key) || !config_json.at(shape_key).is_object()) {
        return result;
    }
    const auto &configs = config_json.at(shape_key);

    auto usable = [&](size_t token) -> bool {
        const auto key = std::to_string(token);
        if (!configs.contains(key) || !configs.at(key).is_object()) {
            return false;
        }
        const int mode = configs.at(key).value("MODE", result.mode);
        return mode < 1000
            || mode_policy == HygonMarlinModePolicy::All
            || (mode_policy == HygonMarlinModePolicy::LegacyAndBf16Mode1000
                && mode == 1000);
    };

    size_t chosen = 0;
    bool has_choice = false;
    size_t chosen_ge = std::numeric_limits<size_t>::max();
    size_t closest_diff = std::numeric_limits<size_t>::max();
    for (auto it = configs.begin(); it != configs.end(); ++it) {
        size_t token = 0;
        try {
            token = static_cast<size_t>(std::stoull(it.key()));
        } catch (const std::exception &) {
            continue;
        }
        if (!usable(token)) {
            continue;
        }
        if (token >= m && token < chosen_ge) {
            chosen_ge = token;
            chosen = token;
            has_choice = true;
        }
        const size_t diff = token > m ? token - m : m - token;
        if (diff < closest_diff) {
            closest_diff = diff;
            if (chosen_ge == std::numeric_limits<size_t>::max()) {
                chosen = token;
                has_choice = true;
            }
        }
    }
    if (!has_choice) {
        return result;
    }

    const auto &config = configs.at(std::to_string(chosen));
    result.mode = config.value("MODE", result.mode);
    result.delta = config.value("DELTA", result.delta);
    result.block_size_m = config.value("BLOCK_SIZE_M", result.block_size_m);
    result.found = config.contains("MODE");
    return result;
}

} // namespace

HygonW16A16MarlinRuntimeConfig select_hygon_w16a16_marlin_config(
    size_t num_tokens,
    size_t hidden_size,
    size_t intermediate_size,
    DataType hidden_dtype,
    size_t device_index) {
    HygonW16A16MarlinRuntimeConfig config;
    const auto device_info = adaptor::lightop::device_info(device_index);
    const auto mode_policy = hidden_dtype == DataType::BF16
        ? HygonMarlinModePolicy::LegacyAndBf16Mode1000
        : HygonMarlinModePolicy::LegacyOnly;
    config.gemm1 = load_lightop_marlin_config(
        intermediate_size * 2,
        hidden_size,
        num_tokens,
        "MOE_W16A16_CUDA_MARLIN",
        device_info,
        mode_policy,
        false,
        false);
    config.gemm2 = load_lightop_marlin_config(
        hidden_size,
        intermediate_size,
        num_tokens,
        "MOE_W16A16_CUDA_MARLIN",
        device_info,
        mode_policy,
        false,
        false);
    config.supported = config.gemm1.found && config.gemm2.found;
    return config;
}

HygonW8A8MarlinRuntimeConfig select_hygon_w8a8_marlin_config(
    size_t num_tokens,
    size_t hidden_size,
    size_t intermediate_size,
    size_t device_index) {
    HygonW8A8MarlinRuntimeConfig config;
    const auto device_info = adaptor::lightop::device_info(device_index);
    config.gemm1 = load_lightop_marlin_config(
        intermediate_size * 2,
        hidden_size,
        num_tokens,
        "MOE_BLOCKINT8_CUDA_MARLIN",
        device_info,
        HygonMarlinModePolicy::All,
        true,
        true);
    config.gemm2 = load_lightop_marlin_config(
        hidden_size,
        intermediate_size,
        num_tokens,
        "MOE_BLOCKINT8_CUDA_MARLIN",
        device_info,
        HygonMarlinModePolicy::All,
        true,
        true);
    config.supported = config.gemm1.found && config.gemm2.found;
    return config;
}

} // namespace infinicore::op
