#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/acos.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <unordered_map>

namespace infinicore::op::acos_impl::infiniop {

// -------------------------------------------
// descriptor 缓存
// -------------------------------------------
thread_local common::OpCache<size_t, infiniopAcosDescriptor_t> caches(
    100, // capacity
    [](infiniopAcosDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAcosDescriptor(desc));
            desc = nullptr;
        }
    }
);

// 每个 descriptor 对应的 workspace 缓存条目
struct WorkspaceEntry {
    size_t size = 0;
    std::shared_ptr<Memory> buf = nullptr;
};

// -------------------------------------------
// 核心计算函数
// -------------------------------------------
void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    // 获取或创建 descriptor
    auto desc_opt = cache.get(seed);
    infiniopAcosDescriptor_t desc = nullptr;
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAcosDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // thread-local map: 每个线程维护 descriptor -> workspace 映射
    static thread_local std::unordered_map<infiniopAcosDescriptor_t, WorkspaceEntry> s_workspace_map;
    auto it = s_workspace_map.find(desc);

    if (it == s_workspace_map.end()) {
        // 第一次使用 descriptor：查询 workspace size 并分配
        size_t workspace_size = 0;
        INFINICORE_CHECK_ERROR(infiniopGetAcosWorkspaceSize(desc, &workspace_size));

        WorkspaceEntry entry;
        if (workspace_size > 0) {
            entry.buf = context::allocateMemory(workspace_size);
            entry.size = workspace_size;
        } else {
            entry.buf = nullptr;
            entry.size = 0;
        }
        it = s_workspace_map.emplace(desc, std::move(entry)).first;
    } else {
        // 已缓存：检查 workspace 是否足够，如果不够则重新分配
        size_t required_size = 0;
        INFINICORE_CHECK_ERROR(infiniopGetAcosWorkspaceSize(desc, &required_size));
        if (required_size > it->second.size) {
            it->second.buf = context::allocateMemory(required_size);
            it->second.size = required_size;
        }
    }

    // 使用缓存 workspace，注意迭代器检查
    void* workspace_ptr = (it != s_workspace_map.end() && it->second.buf) ? it->second.buf->data() : nullptr;
    size_t workspace_size = (it != s_workspace_map.end()) ? it->second.size : 0;

    // 调用 infiniopAcos 内核
    INFINICORE_CHECK_ERROR(infiniopAcos(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()
    ));
}

// -------------------------------------------
// 注册 dispatcher
// -------------------------------------------
static bool registered = []() {
    Acos::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::acos_impl::infiniop
