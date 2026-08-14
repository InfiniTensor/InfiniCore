#include "infinicore/ops/moe_marlin_config.hpp"

#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iterator>
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

// LightOP config files have a fixed object -> object -> integer object schema.
// Parsing only that schema keeps this path dependency-free on every platform.
class LightopConfigReader {
public:
    explicit LightopConfigReader(const std::string &text) : _text(text) {}

    bool consume(char expected) {
        skip_whitespace();
        if (_pos < _text.size() && _text[_pos] == expected) {
            ++_pos;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    std::string key() {
        expect('"');
        const size_t begin = _pos;
        while (_pos < _text.size() && _text[_pos] != '"') {
            const unsigned char ch = static_cast<unsigned char>(_text[_pos]);
            if (ch == '\\' || ch < 0x20) {
                fail("LightOP config keys must be unescaped strings");
            }
            ++_pos;
        }
        if (_pos == _text.size()) {
            fail("unterminated key");
        }
        const std::string result = _text.substr(begin, _pos - begin);
        ++_pos;
        return result;
    }

    long long integer() {
        skip_whitespace();
        const size_t begin = _pos;
        if (_pos < _text.size() && _text[_pos] == '-') {
            ++_pos;
        }
        const size_t digits_begin = _pos;
        while (_pos < _text.size() && _text[_pos] >= '0' && _text[_pos] <= '9') {
            ++_pos;
        }
        if (_pos == digits_begin) {
            fail("expected integer");
        }
        if (_pos - digits_begin > 1 && _text[digits_begin] == '0') {
            fail("integer has a leading zero");
        }
        if (_pos < _text.size()
            && (_text[_pos] == '.' || _text[_pos] == 'e' || _text[_pos] == 'E')) {
            fail("expected integer, found non-integral number");
        }
        try {
            return std::stoll(_text.substr(begin, _pos - begin));
        } catch (const std::exception &) {
            fail("integer is out of range");
        }
    }

    void finish() {
        skip_whitespace();
        if (_pos != _text.size()) {
            fail("unexpected trailing content");
        }
    }

private:
    void skip_whitespace() {
        while (_pos < _text.size()
               && std::isspace(static_cast<unsigned char>(_text[_pos])) != 0) {
            ++_pos;
        }
    }

    [[noreturn]] void fail(const std::string &message) const {
        throw std::runtime_error(
            "invalid LightOP JSON at byte " + std::to_string(_pos) + ": " + message);
    }

    const std::string &_text;
    size_t _pos = 0;
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

bool parse_size(const std::string &text, size_t &value) {
    if (text.empty()) {
        return false;
    }
    size_t result = 0;
    for (const char ch : text) {
        if (ch < '0' || ch > '9') {
            return false;
        }
        const size_t digit = static_cast<size_t>(ch - '0');
        if (result > (std::numeric_limits<size_t>::max() - digit) / 10) {
            return false;
        }
        result = result * 10 + digit;
    }
    value = result;
    return true;
}

int checked_int(long long value, const std::string &field) {
    if (value < std::numeric_limits<int>::min()
        || value > std::numeric_limits<int>::max()) {
        throw std::runtime_error("LightOP field " + field + " is out of int range");
    }
    return static_cast<int>(value);
}

HygonMarlinGemmConfig parse_gemm_config(LightopConfigReader &json) {
    HygonMarlinGemmConfig config;
    bool has_mode = false;
    json.expect('{');
    if (json.consume('}')) {
        return config;
    }
    while (true) {
        const std::string field = json.key();
        json.expect(':');
        const long long value = json.integer();
        if (field == "MODE") {
            config.mode = checked_int(value, field);
            has_mode = true;
        } else if (field == "DELTA") {
            config.delta = checked_int(value, field);
        } else if (field == "BLOCK_SIZE_M") {
            if (value < 0
                || static_cast<unsigned long long>(value)
                       > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("LightOP field BLOCK_SIZE_M is out of size_t range");
            }
            config.block_size_m = static_cast<size_t>(value);
        }
        if (json.consume('}')) {
            break;
        }
        json.expect(',');
    }
    config.found = has_mode;
    return config;
}

bool mode_is_usable(int mode, HygonMarlinModePolicy mode_policy) {
    return mode < 1000
        || mode_policy == HygonMarlinModePolicy::All
        || (mode_policy == HygonMarlinModePolicy::LegacyAndBf16Mode1000 && mode == 1000);
}

struct ConfigChoice {
    HygonMarlinGemmConfig config;
    size_t chosen_ge = std::numeric_limits<size_t>::max();
    size_t closest_diff = std::numeric_limits<size_t>::max();

    void consider(
        size_t token,
        const HygonMarlinGemmConfig &candidate,
        size_t requested_tokens,
        HygonMarlinModePolicy mode_policy) {
        if (!candidate.found || !mode_is_usable(candidate.mode, mode_policy)) {
            return;
        }
        if (token >= requested_tokens && token < chosen_ge) {
            chosen_ge = token;
            config = candidate;
        }
        const size_t diff = token > requested_tokens
                              ? token - requested_tokens
                              : requested_tokens - token;
        if (chosen_ge == std::numeric_limits<size_t>::max() && diff < closest_diff) {
            closest_diff = diff;
            config = candidate;
        }
    }
};

void parse_token_configs(
    LightopConfigReader &json,
    bool select,
    size_t requested_tokens,
    HygonMarlinModePolicy mode_policy,
    ConfigChoice &choice) {
    json.expect('{');
    if (json.consume('}')) {
        return;
    }
    while (true) {
        const std::string token_key = json.key();
        json.expect(':');
        const HygonMarlinGemmConfig candidate = parse_gemm_config(json);
        size_t token = 0;
        if (select && parse_size(token_key, token)) {
            choice.consider(token, candidate, requested_tokens, mode_policy);
        }
        if (json.consume('}')) {
            break;
        }
        json.expect(',');
    }
}

HygonMarlinGemmConfig parse_lightop_marlin_config(
    const std::string &contents,
    const std::string &shape_key,
    size_t requested_tokens,
    HygonMarlinModePolicy mode_policy) {
    LightopConfigReader json(contents);
    ConfigChoice choice;
    json.expect('{');
    if (!json.consume('}')) {
        while (true) {
            const std::string current_shape = json.key();
            json.expect(':');
            parse_token_configs(
                json,
                current_shape == shape_key,
                requested_tokens,
                mode_policy,
                choice);
            if (json.consume('}')) {
                break;
            }
            json.expect(',');
        }
    }
    json.finish();
    return choice.config;
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
    const std::string num_cus_suffix = num_cus_with_cu_prefix ? ("_CU" + num_cus) : ("_" + num_cus);
    const std::string file_name = lightop_config_dir() + "/" + file_prefix + "_"
                                + std::to_string(n) + "_" + std::to_string(k) + "_"
                                + device_name + num_cus_suffix + ".json";

    std::ifstream file(file_name);
    if (!file.is_open()) {
        return result;
    }

    const std::string shape_key = std::to_string(n) + "_" + std::to_string(k);
    const std::string contents{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()};
    try {
        return parse_lightop_marlin_config(contents, shape_key, m, mode_policy);
    } catch (const std::exception &error) {
        throw std::runtime_error(
            "Failed to parse LightOP config " + file_name + ": " + error.what());
    }
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
