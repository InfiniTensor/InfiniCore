#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "infinicore/adaptor/hygon_aiter_adaptor.hpp"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::adaptor::hygon_aiter {
namespace {

using dynamic_per_token_scaled_quant_ref_fn = void (*)(
    at::Tensor &,
    const at::Tensor &,
    at::Tensor &,
    const std::optional<at::Tensor> &,
    bool,
    const std::optional<at::Tensor> &,
    int);
using dynamic_per_token_scaled_quant_value_fn = void (*)(
    at::Tensor &,
    const at::Tensor &,
    at::Tensor &,
    const std::optional<at::Tensor> &,
    bool,
    std::optional<at::Tensor>,
    int);

struct Symbols {
    void *handle = nullptr;
    void *pos_encoding_handle = nullptr;
    void *moe_handle = nullptr;
    dynamic_per_token_scaled_quant_ref_fn dynamic_per_token_scaled_quant_ref = nullptr;
    dynamic_per_token_scaled_quant_value_fn dynamic_per_token_scaled_quant_value = nullptr;
    std::string error;
};

Symbols &symbols() {
    static Symbols value;
    static std::once_flag once;
    std::call_once(once, []() {
        const char *env_path = std::getenv("INFINICORE_HYGON_AITER_CPP_API_SO");
        const char *paths[] = {
            env_path,
            "/usr/local/lib/python3.10/dist-packages/aiter/jit/module_cpp_api.so",
        };
        for (const char *path : paths) {
            if (path == nullptr || path[0] == '\0') {
                continue;
            }
            void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (handle == nullptr) {
                value.error = dlerror();
                continue;
            }
            auto ref_fn = reinterpret_cast<dynamic_per_token_scaled_quant_ref_fn>(
                dlsym(handle,
                      "_ZN5aiter6native30dynamic_per_token_scaled_quantERN2at6TensorERKS2_S3_RKSt8optionalIS2_EbS9_i"));
            auto value_fn = reinterpret_cast<dynamic_per_token_scaled_quant_value_fn>(
                dlsym(handle,
                      "_ZN5aiter6native30dynamic_per_token_scaled_quantERN2at6TensorERKS2_S3_RKSt8optionalIS2_EbS7_i"));
            if (ref_fn == nullptr && value_fn == nullptr) {
                value.error = dlerror();
                dlclose(handle);
                continue;
            }
            value.handle = handle;
            value.dynamic_per_token_scaled_quant_ref = ref_fn;
            value.dynamic_per_token_scaled_quant_value = value_fn;
            value.error.clear();
            break;
        }

        const char *pos_encoding_env_path =
            std::getenv("INFINICORE_HYGON_AITER_POS_ENCODING_SO");
        const char *pos_encoding_paths[] = {
            pos_encoding_env_path,
            "/usr/local/lib/python3.10/dist-packages/aiter/jit/module_pos_encoding.so",
        };
        for (const char *path : pos_encoding_paths) {
            if (path == nullptr || path[0] == '\0') {
                continue;
            }
            void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (handle == nullptr) {
                value.error = dlerror();
                continue;
            }
            value.pos_encoding_handle = handle;
            break;
        }

        const char *moe_env_path = std::getenv("INFINICORE_HYGON_AITER_MOE_SO");
        const char *moe_paths[] = {
            moe_env_path,
            "/usr/local/lib/python3.10/dist-packages/aiter/jit/module_moe.so",
        };
        for (const char *path : moe_paths) {
            if (path == nullptr || path[0] == '\0') {
                continue;
            }
            void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (handle == nullptr) {
                value.error = dlerror();
                continue;
            }
            value.moe_handle = handle;
            break;
        }
    });
    return value;
}

} // namespace

bool dynamic_per_token_scaled_quant_available() {
    const auto &value = symbols();
    return value.dynamic_per_token_scaled_quant_ref != nullptr
        || value.dynamic_per_token_scaled_quant_value != nullptr;
}

bool rotary_embedding_available() {
    return symbols().pos_encoding_handle != nullptr
        && c10::Dispatcher::singleton()
               .findSchema({"aiter::rotary_embedding_fwd", ""})
               .has_value();
}

bool grouped_topk_available() {
    if (symbols().moe_handle == nullptr) {
        return false;
    }
    auto &dispatcher = c10::Dispatcher::singleton();
    return dispatcher.findSchema({"aiter::grouped_topk", ""}).has_value()
        && dispatcher.findSchema({"aiter::biased_grouped_topk_hip", ""}).has_value();
}

bool fused_moe_w8a8_available() {
    return c10::Dispatcher::singleton()
        .findSchema({"aiter::moe_c_fused_experts", ""})
        .has_value();
}

void dynamic_per_token_scaled_quant(at::Tensor &output,
                                    const at::Tensor &input,
                                    at::Tensor &scales) {
    auto &value = symbols();
    if (value.dynamic_per_token_scaled_quant_ref == nullptr
        && value.dynamic_per_token_scaled_quant_value == nullptr) {
        throw std::runtime_error(
            "AITER dynamic_per_token_scaled_quant is unavailable: " + value.error);
    }
    const std::optional<at::Tensor> scale_ub = std::nullopt;
    const std::optional<at::Tensor> num_rows = std::nullopt;
    if (value.dynamic_per_token_scaled_quant_ref != nullptr) {
        value.dynamic_per_token_scaled_quant_ref(
            output, input, scales, scale_ub, false, num_rows, 1);
    } else {
        value.dynamic_per_token_scaled_quant_value(
            output, input, scales, scale_ub, false, num_rows, 1);
    }
}

void rotary_embedding(const at::Tensor &positions,
                      at::Tensor &query,
                      at::Tensor &key,
                      int64_t head_size,
                      const at::Tensor &cos_cache,
                      const at::Tensor &sin_cache,
                      bool is_neox,
                      bool is_nope_first) {
    if (!rotary_embedding_available()) {
        throw std::runtime_error(
            "AITER rotary_embedding_fwd is unavailable: " + symbols().error);
    }
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "aiter::rotary_embedding_fwd", "");
    c10::Stack stack;
    stack.emplace_back(positions);
    stack.emplace_back(query);
    stack.emplace_back(key);
    stack.emplace_back(head_size);
    stack.emplace_back(cos_cache);
    stack.emplace_back(sin_cache);
    stack.emplace_back(is_neox);
    stack.emplace_back(is_nope_first);
    op.callBoxed(&stack);
}

void grouped_topk(at::Tensor &topk_weights,
                  at::Tensor &topk_ids,
                  const at::Tensor &scores,
                  int64_t num_expert_group,
                  int64_t topk_group,
                  bool renormalize,
                  const std::optional<at::Tensor> &bias,
                  bool is_softmax) {
    if (!grouped_topk_available()) {
        throw std::runtime_error("AITER grouped top-k operators are unavailable: "
                                 + symbols().error);
    }

    c10::Stack stack;
    stack.emplace_back(scores);
    if (bias.has_value()) {
        if (is_softmax) {
            throw std::runtime_error(
                "AITER biased grouped top-k only supports sigmoid scoring");
        }
        stack.emplace_back(*bias);
        stack.emplace_back(topk_weights);
        stack.emplace_back(topk_ids);
        stack.emplace_back(num_expert_group);
        stack.emplace_back(topk_group);
        stack.emplace_back(renormalize);
        stack.emplace_back(1.0);
        auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
            "aiter::biased_grouped_topk_hip", "");
        op.callBoxed(&stack);
        return;
    }

    stack.emplace_back(topk_weights);
    stack.emplace_back(topk_ids);
    stack.emplace_back(num_expert_group);
    stack.emplace_back(topk_group);
    stack.emplace_back(renormalize);
    stack.emplace_back(is_softmax);
    stack.emplace_back(1.0);
    auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "aiter::grouped_topk", "");
    op.callBoxed(&stack);
}

void fused_moe_w8a8(at::Tensor &output,
                     const at::Tensor &input,
                     const at::Tensor &w1,
                     const at::Tensor &w2,
                     const at::Tensor &topk_weights,
                     const at::Tensor &topk_ids,
                     const at::Tensor &w1_scale,
                     const at::Tensor &w2_scale) {
    if (!fused_moe_w8a8_available()) {
        throw std::runtime_error(
            "AITER moe_c_fused_experts is unavailable; import aiter.fused_moe_c before use");
    }

    auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "aiter::moe_c_fused_experts", "");
    c10::Stack stack;
    stack.emplace_back(input);
    stack.emplace_back(w1);
    stack.emplace_back(w2);
    stack.emplace_back(topk_weights);
    stack.emplace_back(topk_ids);
    for (size_t i = 0; i < 11; ++i) {
        stack.emplace_back(int64_t{1});
    }
    stack.emplace_back(false);                 // inplace
    stack.emplace_back(std::string("silu"));   // activation
    stack.emplace_back(true);                  // is_gated
    stack.emplace_back(true);                  // use_fp8_w8a8
    for (size_t i = 0; i < 8; ++i) {
        stack.emplace_back(false);
    }
    stack.emplace_back(int64_t{-1});           // global_num_experts
    stack.emplace_back();                      // expert_map
    stack.emplace_back(w1_scale);
    stack.emplace_back(w2_scale);
    for (size_t i = 0; i < 4; ++i) {
        stack.emplace_back();
    }
    stack.emplace_back();                      // block_shape
    stack.emplace_back(1.0);                   // routed_scaling_factor
    stack.emplace_back();                      // gemm1_alpha
    stack.emplace_back();                      // gemm1_limit
    stack.emplace_back();                      // compute_dtype
    stack.emplace_back();                      // w16a16_config
    stack.emplace_back();                      // fp8_w8a8_config
    op.callBoxed(&stack);

    output.copy_(std::move(stack.back()).toTensor());
}

} // namespace infinicore::adaptor::hygon_aiter

#endif
