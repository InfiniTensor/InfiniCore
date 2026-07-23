#if defined(ENABLE_ASCEND_FLASH_ATTN)

#include "infinicore/context/context.hpp"
#include "infinicore/ops/mha.hpp"

#include <acl/acl.h>
#include <aclnnop/aclnn_prompt_flash_attention_v3.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace infinicore::op::mha_impl::flashattn_ascend {

static aclDataType to_acl_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::F16:
        return ACL_FLOAT16;
    case DataType::BF16:
        return ACL_BF16;
    case DataType::F32:
        return ACL_FLOAT;
    default:
        throw std::runtime_error("[mha/ascend] unsupported dtype");
    }
}

static std::vector<int64_t> to_i64_shape(const std::vector<size_t> &shape) {
    std::vector<int64_t> dims;
    dims.reserve(shape.size());
    for (size_t dim : shape) {
        dims.push_back(static_cast<int64_t>(dim));
    }
    return dims;
}

static std::vector<int64_t>
contiguous_strides(const std::vector<int64_t> &dims) {
    std::vector<int64_t> strides(dims.size(), 1);
    for (size_t i = dims.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

static aclTensor *to_acl_tensor(const Tensor &tensor) {
    auto dims = to_i64_shape(tensor->shape());
    auto strides = contiguous_strides(dims);
    return aclCreateTensor(
        dims.data(), dims.size(), to_acl_dtype(tensor->dtype()), strides.data(),
        0, ACL_FORMAT_ND, dims.data(), dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(tensor->data())));
}

struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    bool is_causal;
};

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi_slopes, float scale, bool is_causal) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        alibi_slopes
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes))
            : std::nullopt,
        scale,
        is_causal};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    infinicore::context::setDevice(p->q->device());

    if (p->alibi_slopes.has_value()) {
        throw std::runtime_error(
            "[mha/ascend] ALiBi is not supported by PromptFlashAttentionV3");
    }

    const auto q_shape = p->q->shape();
    const auto k_shape = p->k->shape();
    const auto v_shape = p->v->shape();
    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::runtime_error(
            "[mha/ascend] expected q, k and v to use BSND layout");
    }
    if (q_shape[0] != k_shape[0] || k_shape[0] != v_shape[0]
        || k_shape[1] != v_shape[1] || k_shape[2] != v_shape[2]
        || q_shape[3] != k_shape[3] || k_shape[3] != v_shape[3]) {
        throw std::runtime_error("[mha/ascend] incompatible q, k and v shapes");
    }
    if (k_shape[2] == 0 || q_shape[2] % k_shape[2] != 0) {
        throw std::runtime_error(
            "[mha/ascend] query heads must be divisible by key/value heads");
    }

    Tensor q_work = p->q->is_contiguous() ? Tensor(p->q) : p->q->contiguous();
    Tensor k_work = p->k->is_contiguous() ? Tensor(p->k) : p->k->contiguous();
    Tensor v_work = p->v->is_contiguous() ? Tensor(p->v) : p->v->contiguous();
    Tensor out_work
        = p->out->is_contiguous() ? Tensor(p->out) : p->out->contiguous();

    aclTensor *query_acl = to_acl_tensor(q_work);
    aclTensor *key_acl = to_acl_tensor(k_work);
    aclTensor *value_acl = to_acl_tensor(v_work);
    aclTensor *out_acl = to_acl_tensor(out_work);
    if (query_acl == nullptr || key_acl == nullptr || value_acl == nullptr
        || out_acl == nullptr) {
        if (query_acl != nullptr) {
            aclDestroyTensor(query_acl);
        }
        if (key_acl != nullptr) {
            aclDestroyTensor(key_acl);
        }
        if (value_acl != nullptr) {
            aclDestroyTensor(value_acl);
        }
        if (out_acl != nullptr) {
            aclDestroyTensor(out_acl);
        }
        throw std::runtime_error("[mha/ascend] failed to create aclTensor");
    }

    void *mask_device = nullptr;
    aclTensor *mask_acl = nullptr;
    if (p->is_causal) {
        const int64_t query_len = static_cast<int64_t>(q_shape[1]);
        const int64_t kv_len = static_cast<int64_t>(k_shape[1]);
        const int64_t mask_kv_len = (kv_len + 31) / 32 * 32;
        const int64_t causal_offset = kv_len - query_len;
        std::vector<uint8_t> mask_host(query_len * mask_kv_len, 1);
        for (int64_t i = 0; i < query_len; ++i) {
            for (int64_t j = 0; j < kv_len; ++j) {
                if (j <= i + causal_offset) {
                    mask_host[i * mask_kv_len + j] = 0;
                }
            }
        }

        const size_t mask_bytes = mask_host.size() * sizeof(uint8_t);
        aclError ret
            = aclrtMalloc(&mask_device, mask_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret == ACL_SUCCESS) {
            ret = aclrtMemcpy(mask_device, mask_bytes, mask_host.data(),
                              mask_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (ret != ACL_SUCCESS) {
            if (mask_device != nullptr) {
                aclrtFree(mask_device);
            }
            aclDestroyTensor(query_acl);
            aclDestroyTensor(key_acl);
            aclDestroyTensor(value_acl);
            aclDestroyTensor(out_acl);
            throw std::runtime_error(
                "[mha/ascend] failed to allocate causal mask: "
                + std::to_string(ret));
        }

        std::vector<int64_t> mask_dims = {query_len, mask_kv_len};
        std::vector<int64_t> mask_strides = {mask_kv_len, 1};
        mask_acl = aclCreateTensor(
            mask_dims.data(), mask_dims.size(), ACL_UINT8, mask_strides.data(),
            0, ACL_FORMAT_ND, mask_dims.data(), mask_dims.size(), mask_device);
        if (mask_acl == nullptr) {
            aclrtFree(mask_device);
            aclDestroyTensor(query_acl);
            aclDestroyTensor(key_acl);
            aclDestroyTensor(value_acl);
            aclDestroyTensor(out_acl);
            throw std::runtime_error(
                "[mha/ascend] failed to create causal mask tensor");
        }
    }

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;
    const int64_t num_heads = static_cast<int64_t>(q_shape[2]);
    const int64_t num_kv_heads = static_cast<int64_t>(k_shape[2]);
    aclnnStatus ret = aclnnPromptFlashAttentionV3GetWorkspaceSize(
        query_acl, key_acl, value_acl,
        nullptr, // pseShift
        mask_acl,
        nullptr, // actualSeqLengths
        nullptr, // actualSeqLengthsKv
        nullptr, // deqScale1
        nullptr, // quantScale1
        nullptr, // deqScale2
        nullptr, // quantScale2
        nullptr, // quantOffset2
        num_heads, static_cast<double>(p->scale), 2147483647, 2147483647,
        const_cast<char *>("BSND"), num_kv_heads,
        0, // sparseMode: use the full mask when causal
        1, out_acl, &workspace_size, &executor);

    if (ret != ACL_SUCCESS) {
        if (mask_acl != nullptr) {
            aclDestroyTensor(mask_acl);
            aclrtFree(mask_device);
        }
        aclDestroyTensor(query_acl);
        aclDestroyTensor(key_acl);
        aclDestroyTensor(value_acl);
        aclDestroyTensor(out_acl);
        const char *error_message = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string(
                "[mha/ascend] PromptFlashAttentionV3GetWorkspaceSize failed: ")
            + std::to_string(ret) + ", msg: "
            + (error_message != nullptr ? error_message : "(null)"));
    }

    void *workspace = nullptr;
    if (workspace_size > 0) {
        aclError alloc_ret = aclrtMalloc(
            &workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if (alloc_ret != ACL_SUCCESS) {
            if (mask_acl != nullptr) {
                aclDestroyTensor(mask_acl);
                aclrtFree(mask_device);
            }
            aclDestroyTensor(query_acl);
            aclDestroyTensor(key_acl);
            aclDestroyTensor(value_acl);
            aclDestroyTensor(out_acl);
            throw std::runtime_error(
                "[mha/ascend] failed to allocate workspace: "
                + std::to_string(alloc_ret));
        }
    }

    aclrtStream stream
        = static_cast<aclrtStream>(infinicore::context::getStream());
    ret = aclnnPromptFlashAttentionV3(workspace, workspace_size, executor,
                                      stream);
    aclrtSynchronizeStream(stream);

    if (workspace != nullptr) {
        aclrtFree(workspace);
    }
    if (mask_acl != nullptr) {
        aclDestroyTensor(mask_acl);
        aclrtFree(mask_device);
    }
    aclDestroyTensor(query_acl);
    aclDestroyTensor(key_acl);
    aclDestroyTensor(value_acl);
    aclDestroyTensor(out_acl);

    if (ret != ACL_SUCCESS) {
        const char *error_message = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string("[mha/ascend] PromptFlashAttentionV3 failed: ")
            + std::to_string(ret) + ", msg: "
            + (error_message != nullptr ? error_message : "(null)"));
    }

    if (!p->out->is_contiguous()) {
        p->out->copy_from(out_work);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::ASCEND,
                                                         &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::ASCEND,
                                                        &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(
        Device::Type::ASCEND, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::flashattn_ascend
#endif
