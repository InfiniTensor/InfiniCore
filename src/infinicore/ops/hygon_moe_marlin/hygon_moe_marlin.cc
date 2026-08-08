#include "infinicore/ops/hygon_moe_marlin.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/moe_align.hpp"
#include "infinicore/ops/moe_marlin_config.hpp"
#include "infinicore/ops/moe_w16a16_marlin.hpp"
#include "infinicore/ops/moe_w8a8_marlin.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <string>

namespace infinicore::op {
namespace {

constexpr size_t kHygonMoeSliceTokens = 16384;

struct RoutingMetadata {
    Tensor sorted_token_ids;
    Tensor expert_ids;
    Tensor num_tokens_post_padded;
};

bool same_device(const Tensor &tensor, const Device &device) {
    return tensor
        && tensor->device().getType() == device.getType()
        && tensor->device().getIndex() == device.getIndex();
}

void ensure_tensor(
    Tensor &tensor,
    const Shape &shape,
    DataType dtype,
    const Device &device,
    const char *name) {
    if (!same_device(tensor, device)
        || tensor->dtype() != dtype
        || tensor->shape() != shape) {
        if (context::isGraphRecording()) {
            throw std::runtime_error(
                std::string("Hygon MoE Marlin ") + name
                + " workspace was not initialized before graph capture");
        }
        tensor = Tensor::empty(shape, dtype, device);
    }
}

std::string shape_to_string(const Shape &shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) {
            oss << ", ";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

void check_packed_weight_tensor(
    const Tensor &tensor,
    const std::string &name,
    const Device &device,
    DataType dtype,
    const Shape &shape) {
    if (!tensor) {
        throw std::runtime_error(
            "Hygon MoE Marlin requires " + name);
    }
    if (tensor->device().getType() != device.getType()
        || tensor->device().getIndex() != device.getIndex()) {
        throw std::runtime_error(
            "Hygon MoE Marlin requires packed weights on the hidden_states device");
    }
    if (tensor->dtype() != dtype) {
        throw std::runtime_error(
            "Hygon MoE Marlin packed tensor dtype mismatch for " + name);
    }
    if (tensor->shape() != shape) {
        throw std::runtime_error(
            "Hygon MoE Marlin packed weight shape mismatch for " + name
            + ": expected " + shape_to_string(shape)
            + ", got " + shape_to_string(tensor->shape()));
    }
}

RoutingMetadata prepare_routing(
    const Tensor &topk_ids,
    const Tensor &expert_map,
    HygonMoeMarlinWorkspace &workspace,
    size_t num_local_experts,
    size_t block_size) {
    const auto &topk_shape = topk_ids->shape();
    if (topk_shape.size() != 2) {
        throw std::runtime_error(
            "Hygon MoE Marlin requires topk_ids [M, top_k]");
    }
    const size_t num_pairs = topk_shape[0] * topk_shape[1];
    const size_t align_num_experts = num_local_experts + 1;
    const size_t max_num_tokens_padded =
        num_pairs < align_num_experts
        ? num_pairs * block_size
        : num_pairs + align_num_experts * (block_size - 1);
    const size_t sorted_token_ids_capacity =
        ((max_num_tokens_padded + 3) / 4) * 4;
    const size_t max_num_blocks =
        (max_num_tokens_padded + block_size - 1) / block_size;
    const auto device = topk_ids->device();

    if (!same_device(workspace.sorted_token_ids, device)
        || workspace.sorted_token_ids_capacity
            < sorted_token_ids_capacity) {
        if (context::isGraphRecording()) {
            throw std::runtime_error(
                "Hygon MoE Marlin sorted_token_ids workspace was not initialized before graph capture");
        }
        workspace.sorted_token_ids = Tensor::empty(
            {sorted_token_ids_capacity},
            DataType::I32,
            device);
        workspace.sorted_token_ids_capacity =
            sorted_token_ids_capacity;
    }
    if (!same_device(workspace.expert_ids, device)
        || workspace.expert_ids_capacity < max_num_blocks) {
        if (context::isGraphRecording()) {
            throw std::runtime_error(
                "Hygon MoE Marlin expert_ids workspace was not initialized before graph capture");
        }
        workspace.expert_ids = Tensor::empty(
            {max_num_blocks},
            DataType::I32,
            device);
        workspace.expert_ids_capacity = max_num_blocks;
    }
    if (!same_device(workspace.num_tokens_post_padded, device)) {
        if (context::isGraphRecording()) {
            throw std::runtime_error(
                "Hygon MoE Marlin num_tokens_post_padded workspace was not initialized before graph capture");
        }
        workspace.num_tokens_post_padded = Tensor::empty(
            {1},
            DataType::I32,
            device);
    }

    auto sorted_token_ids = workspace.sorted_token_ids->narrow(
        {{0, 0, sorted_token_ids_capacity}});
    auto expert_ids = workspace.expert_ids->narrow(
        {{0, 0, max_num_blocks}});

    if (expert_map) {
        moe_align_with_expert_map_(
            sorted_token_ids,
            expert_ids,
            workspace.num_tokens_post_padded,
            topk_ids,
            expert_map,
            num_local_experts,
            block_size,
            true);
    } else {
        moe_align_(
            sorted_token_ids,
            expert_ids,
            workspace.num_tokens_post_padded,
            topk_ids,
            num_local_experts,
            block_size,
            true);
    }

    return RoutingMetadata{
        sorted_token_ids,
        expert_ids,
        workspace.num_tokens_post_padded,
    };
}

Tensor run_w16a16(
    const Tensor &hidden_states,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const RoutingMetadata &routing,
    const HygonMoeMarlinWeights &weights,
    HygonMoeMarlinWorkspace &workspace,
    size_t num_local_experts,
    size_t hidden_size,
    size_t intermediate_size,
    const HygonW16A16MarlinRuntimeConfig &config) {
    const auto activation_dtype = hidden_states->dtype();
    if (activation_dtype != DataType::BF16
        && activation_dtype != DataType::F16) {
        throw std::runtime_error(
            "Hygon W16A16 Marlin MoE requires BF16 or FP16 activations");
    }
    check_packed_weight_tensor(
        weights.packed_w13,
        "w13",
        hidden_states->device(),
        activation_dtype,
        {num_local_experts,
         hidden_size / 16,
         intermediate_size * 2 * 16});
    check_packed_weight_tensor(
        weights.packed_w2,
        "w2",
        hidden_states->device(),
        activation_dtype,
        {num_local_experts,
         intermediate_size / 16,
         hidden_size * 16});

    const size_t top_k = topk_ids->shape()[1];
    const size_t num_tokens = hidden_states->shape()[0];
    if (num_tokens > kHygonMoeSliceTokens) {
        throw std::runtime_error(
            "Hygon W16A16 Marlin MoE inputs above 16384 tokens must be sliced");
    }
    const size_t cache13_required =
        num_tokens * top_k
        * std::max(intermediate_size * 2, hidden_size);
    const size_t cache2_required =
        num_tokens * top_k * intermediate_size;

    ensure_tensor(
        workspace.output,
        hidden_states->shape(),
        hidden_states->dtype(),
        hidden_states->device(),
        "output");
    if (!same_device(workspace.cache13, hidden_states->device())
        || workspace.cache13->dtype() != hidden_states->dtype()
        || workspace.cache13_capacity < cache13_required) {
        if (context::isGraphRecording()) {
            throw std::runtime_error(
                "Hygon W16A16 Marlin cache13 workspace was not initialized before graph capture");
        }
        workspace.cache13 = Tensor::empty(
            {cache13_required},
            hidden_states->dtype(),
            hidden_states->device());
        workspace.cache13_capacity = cache13_required;
    }
    if (!same_device(workspace.cache2, hidden_states->device())
        || workspace.cache2->dtype() != hidden_states->dtype()
        || workspace.cache2_capacity < cache2_required) {
        if (context::isGraphRecording()) {
            throw std::runtime_error(
                "Hygon W16A16 Marlin cache2 workspace was not initialized before graph capture");
        }
        workspace.cache2 = Tensor::empty(
            {cache2_required},
            hidden_states->dtype(),
            hidden_states->device());
        workspace.cache2_capacity = cache2_required;
    }

    moe_w16a16_marlin_fused_dense_(
        workspace.output,
        workspace.cache13,
        workspace.cache2,
        hidden_states,
        weights.packed_w13,
        weights.packed_w2,
        topk_weights,
        routing.sorted_token_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        top_k,
        config.gemm1.mode,
        config.gemm1.delta,
        config.gemm2.mode,
        config.gemm2.delta);
    return workspace.output;
}

Tensor run_w8a8(
    const Tensor &hidden_states,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const RoutingMetadata &routing,
    const HygonMoeMarlinWeights &weights,
    HygonMoeMarlinWorkspace &workspace,
    size_t num_local_experts,
    size_t hidden_size,
    size_t intermediate_size,
    const HygonW8A8MarlinRuntimeConfig &config) {
    check_packed_weight_tensor(
        weights.packed_w13,
        "w13",
        hidden_states->device(),
        DataType::I8,
        {num_local_experts,
         hidden_size / 64,
         intermediate_size * 2 * 64});
    check_packed_weight_tensor(
        weights.packed_w2,
        "w2",
        hidden_states->device(),
        DataType::I8,
        {num_local_experts,
         intermediate_size / 64,
         hidden_size * 64});
    check_packed_weight_tensor(
        weights.packed_w13_scale,
        "w13_scale",
        hidden_states->device(),
        DataType::F32,
        {num_local_experts, intermediate_size * 2, 1});
    check_packed_weight_tensor(
        weights.packed_w2_scale,
        "w2_scale",
        hidden_states->device(),
        DataType::F32,
        {num_local_experts, hidden_size, 1});

    const size_t top_k = topk_ids->shape()[1];
    const size_t num_tokens = hidden_states->shape()[0];
    const size_t cache13_required =
        num_tokens * top_k
        * std::max(intermediate_size * 2, hidden_size);

    ensure_tensor(
        workspace.output,
        hidden_states->shape(),
        hidden_states->dtype(),
        hidden_states->device(),
        "output");
    if (!same_device(workspace.cache13, hidden_states->device())
        || workspace.cache13->dtype() != hidden_states->dtype()
        || workspace.cache13_capacity < cache13_required) {
        if (context::isGraphRecording()) {
            throw std::runtime_error(
                "Hygon W8A8 Marlin cache13 workspace was not initialized before graph capture");
        }
        workspace.cache13 = Tensor::empty(
            {cache13_required},
            hidden_states->dtype(),
            hidden_states->device());
        workspace.cache13_capacity = cache13_required;
    }
    ensure_tensor(
        workspace.input_i8,
        {num_tokens, hidden_size},
        DataType::I8,
        hidden_states->device(),
        "input_i8");
    ensure_tensor(
        workspace.input_scale,
        {num_tokens, 1},
        DataType::F32,
        hidden_states->device(),
        "input_scale");
    ensure_tensor(
        workspace.cache2_i8,
        {num_tokens * top_k, intermediate_size},
        DataType::I8,
        hidden_states->device(),
        "cache2_i8");
    ensure_tensor(
        workspace.cache2_scale,
        {num_tokens * top_k, 1},
        DataType::F32,
        hidden_states->device(),
        "cache2_scale");

    moe_w8a8_marlin_fused_dense_(
        workspace.output,
        workspace.cache13,
        workspace.cache2_i8,
        workspace.input_i8,
        workspace.input_scale,
        workspace.cache2_scale,
        hidden_states,
        weights.packed_w13,
        weights.packed_w2,
        weights.packed_w13_scale,
        weights.packed_w2_scale,
        topk_weights,
        routing.sorted_token_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        top_k,
        config.gemm1.mode,
        config.gemm1.block_size_m,
        config.gemm1.delta,
        config.gemm2.mode,
        config.gemm2.delta);
    return workspace.output;
}

HygonMoeMarlinOutput run_sliced(
    const Tensor &hidden_states,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const HygonMoeMarlinWeights &weights,
    HygonMoeMarlinWorkspace &workspace,
    size_t num_local_experts,
    size_t hidden_size,
    size_t intermediate_size) {
    if (context::isGraphRecording()) {
        throw std::runtime_error(
            "Hygon sliced MoE Marlin cannot allocate or copy outputs during graph capture");
    }

    ensure_tensor(
        workspace.output,
        hidden_states->shape(),
        hidden_states->dtype(),
        hidden_states->device(),
        "output");
    HygonMoeMarlinWorkspace slice_workspace;
    const auto activation_dtype = hidden_states->dtype();
    const auto device_index = hidden_states->device().getIndex();

    HygonW16A16MarlinRuntimeConfig full_w16_config;
    HygonW8A8MarlinRuntimeConfig full_w8_config;
    if (weights.format == HygonMoeMarlinWeightFormat::W16A16) {
        full_w16_config = select_hygon_w16a16_marlin_config(
            kHygonMoeSliceTokens,
            hidden_size,
            intermediate_size,
            activation_dtype,
            device_index);
        if (!full_w16_config.supported) {
            throw std::runtime_error(
                "No LightOP W16A16 Marlin config found for full Hygon slice");
        }
    } else {
        full_w8_config = select_hygon_w8a8_marlin_config(
            kHygonMoeSliceTokens,
            hidden_size,
            intermediate_size,
            device_index);
        if (!full_w8_config.supported) {
            throw std::runtime_error(
                "No LightOP W8A8 Marlin config found for full Hygon slice");
        }
    }

    const size_t num_tokens = hidden_states->shape()[0];
    size_t offset = 0;
    while (offset < num_tokens) {
        const size_t slice_tokens = std::min(
            kHygonMoeSliceTokens,
            num_tokens - offset);
        auto hidden_slice =
            hidden_states->narrow({{0, offset, slice_tokens}});
        auto topk_weights_slice =
            topk_weights->narrow({{0, offset, slice_tokens}});
        auto topk_ids_slice =
            topk_ids->narrow({{0, offset, slice_tokens}});

        if (weights.format == HygonMoeMarlinWeightFormat::W16A16) {
            const auto config =
                slice_tokens == kHygonMoeSliceTokens
                ? full_w16_config
                : select_hygon_w16a16_marlin_config(
                      slice_tokens,
                      hidden_size,
                      intermediate_size,
                      activation_dtype,
                      device_index);
            if (!config.supported) {
                throw std::runtime_error(
                    "No LightOP W16A16 Marlin config found for sliced Hygon shape");
            }
            const auto routing = prepare_routing(
                topk_ids_slice,
                Tensor(),
                slice_workspace,
                num_local_experts,
                config.gemm1.block_size_m);
            const auto slice_output = run_w16a16(
                hidden_slice,
                topk_weights_slice,
                topk_ids_slice,
                routing,
                weights,
                slice_workspace,
                num_local_experts,
                hidden_size,
                intermediate_size,
                config);
            workspace.output
                ->narrow({{0, offset, slice_tokens}})
                ->copy_from(slice_output);
        } else {
            const auto config =
                slice_tokens == kHygonMoeSliceTokens
                ? full_w8_config
                : select_hygon_w8a8_marlin_config(
                      slice_tokens,
                      hidden_size,
                      intermediate_size,
                      device_index);
            if (!config.supported) {
                throw std::runtime_error(
                    "No LightOP W8A8 Marlin config found for sliced Hygon shape");
            }
            const auto routing = prepare_routing(
                topk_ids_slice,
                Tensor(),
                slice_workspace,
                num_local_experts,
                config.gemm1.block_size_m);
            const auto slice_output = run_w8a8(
                hidden_slice,
                topk_weights_slice,
                topk_ids_slice,
                routing,
                weights,
                slice_workspace,
                num_local_experts,
                hidden_size,
                intermediate_size,
                config);
            workspace.output
                ->narrow({{0, offset, slice_tokens}})
                ->copy_from(slice_output);
        }
        offset += slice_tokens;
    }

    return HygonMoeMarlinOutput{
        workspace.output,
        Tensor(),
        Tensor(),
        Tensor(),
        false,
    };
}

} // namespace

HygonMoeMarlinOutput hygon_moe_marlin_fused(
    const Tensor &hidden_states,
    const Tensor &topk_weights,
    const Tensor &topk_ids,
    const Tensor &expert_map,
    const HygonMoeMarlinWeights &weights,
    HygonMoeMarlinWorkspace &workspace,
    size_t num_local_experts,
    size_t hidden_size,
    size_t intermediate_size,
    size_t fallback_align_block_size) {
    const auto &hidden_shape = hidden_states->shape();
    if (hidden_shape.size() != 2) {
        throw std::runtime_error(
            "Hygon MoE Marlin requires hidden_states [M, K]");
    }
    if (hidden_shape[1] != hidden_size) {
        throw std::runtime_error(
            "Hygon MoE Marlin hidden size mismatch");
    }
    if (topk_weights->shape() != topk_ids->shape()
        || topk_ids->shape().size() != 2
        || topk_ids->shape()[0] != hidden_shape[0]) {
        throw std::runtime_error(
            "Hygon MoE Marlin topk tensors must have shape [M, top_k]");
    }
    if (hidden_shape[0] > kHygonMoeSliceTokens) {
        return run_sliced(
            hidden_states,
            topk_weights,
            topk_ids,
            weights,
            workspace,
            num_local_experts,
            hidden_size,
            intermediate_size);
    }

    size_t block_size = fallback_align_block_size;
    Tensor output;
    RoutingMetadata routing;
    if (weights.format == HygonMoeMarlinWeightFormat::W16A16) {
        const auto config = select_hygon_w16a16_marlin_config(
            hidden_shape[0],
            hidden_size,
            intermediate_size,
            hidden_states->dtype(),
            hidden_states->device().getIndex());
        if (!config.supported) {
            throw std::runtime_error(
                "No LightOP W16A16 Marlin config found for this Hygon shape");
        }
        block_size = config.gemm1.block_size_m;
        routing = prepare_routing(
            topk_ids,
            expert_map,
            workspace,
            num_local_experts,
            block_size);
        output = run_w16a16(
            hidden_states,
            topk_weights,
            topk_ids,
            routing,
            weights,
            workspace,
            num_local_experts,
            hidden_size,
            intermediate_size,
            config);
    } else {
        const auto config = select_hygon_w8a8_marlin_config(
            hidden_shape[0],
            hidden_size,
            intermediate_size,
            hidden_states->device().getIndex());
        if (!config.supported) {
            throw std::runtime_error(
                "No LightOP W8A8 Marlin config found for this Hygon shape");
        }
        block_size = config.gemm1.block_size_m;
        routing = prepare_routing(
            topk_ids,
            expert_map,
            workspace,
            num_local_experts,
            block_size);
        output = run_w8a8(
            hidden_states,
            topk_weights,
            topk_ids,
            routing,
            weights,
            workspace,
            num_local_experts,
            hidden_size,
            intermediate_size,
            config);
    }

    return HygonMoeMarlinOutput{
        output,
        routing.sorted_token_ids,
        routing.expert_ids,
        routing.num_tokens_post_padded,
        true,
    };
}

} // namespace infinicore::op
