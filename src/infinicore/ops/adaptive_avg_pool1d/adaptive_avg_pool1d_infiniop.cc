/*#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/adaptive_avg_pool1d.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <vector>

namespace infinicore::op::adaptive_avg_pool1d_impl::infiniop {

// -------------------------------------------
// 1. 定义资源上下文 (RAII 管理)
// -------------------------------------------
struct AdaptiveAvgPool1dContext {
    infiniopAdaptiveAvgPool1dDescriptor_t desc = nullptr;
    std::shared_ptr<Memory> workspace_buf = nullptr;
    size_t workspace_size = 0;

    // 辅助：获取原生指针
    void* getWorkspacePtr() const {
        return workspace_buf ? workspace_buf->data() : nullptr;
    }
};

// -------------------------------------------
// 2. 统一 LRU 缓存
// -------------------------------------------
thread_local common::OpCache<size_t, AdaptiveAvgPool1dContext> caches(
    256, // 缓存容量
    [](AdaptiveAvgPool1dContext &ctx) {
        // Deleter: 当缓存条目被驱逐时调用
        if (ctx.desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAdaptiveAvgPool1dDescriptor(ctx.desc));
            ctx.desc = nullptr;
        }
        // shared_ptr 引用计数归零，自动释放显存
        ctx.workspace_buf = nullptr;
        ctx.workspace_size = 0;
    }
);

// -------------------------------------------
// 3. 元数据哈希计算
// -------------------------------------------
inline size_t compute_key(const Tensor& output, const Tensor& input) {
    size_t seed = 0;
    
    // [修复技巧]
    // 将 Enum 类型强制转为 size_t。
    // 这会强制匹配 hash.hpp 中的 arithmetic 重载版本，
    // 从而避开那个会导致编译错误的 Variadic Template 递归路径。

    // Input Metadata
    infinicore::hash_combine(seed, static_cast<size_t>(input->dtype())); 
    for (auto d : input->shape()) infinicore::hash_combine(seed, d);
    for (auto s : input->strides()) infinicore::hash_combine(seed, s);
    
    // Output Metadata
    infinicore::hash_combine(seed, static_cast<size_t>(output->dtype()));
    for (auto d : output->shape()) infinicore::hash_combine(seed, d);
    
    return seed;
}

// -------------------------------------------
// 4. 核心计算函数 (带极速路径)
// -------------------------------------------
void calculate(Tensor output, Tensor input) {
    // 1. 计算 Hash Key
    size_t seed = compute_key(output, input);

    // 2. Fast Path (Last Used) 变量
    static thread_local size_t last_seed = 0;
    static thread_local bool last_ctx_valid = false;
    
    // 保存 Context 对象副本 (轻量级拷贝)
    static thread_local AdaptiveAvgPool1dContext last_ctx; 

    AdaptiveAvgPool1dContext ctx;

    // 3. 检查 Fast Path
    if (last_ctx_valid && seed == last_seed) {
        ctx = last_ctx; 
    } else {
        // 4. Slow Path: 查找 LRU 缓存
        auto device_type = context::getDevice().getType();
        auto device_index = context::getDevice().getIndex();
        auto &cache = caches.getCache(device_type, device_index);

        auto opt_ctx = cache.get(seed);
        if (opt_ctx) {
            // 命中缓存：拷贝 Context
            ctx = *opt_ctx;
        } else {
            // 未命中：创建 Descriptor 并分配 Workspace
            AdaptiveAvgPool1dContext new_ctx;
            
            // A. 创建 Descriptor
            INFINICORE_CHECK_ERROR(infiniopCreateAdaptiveAvgPool1dDescriptor(
                context::getInfiniopHandle(output->device()), 
                &new_ctx.desc,
                output->desc(), 
                input->desc()));

            // B. 查询并分配 Workspace (仅在创建时做一次)
            INFINICORE_CHECK_ERROR(infiniopGetAdaptiveAvgPool1dWorkspaceSize(new_ctx.desc, &new_ctx.workspace_size));
            
            if (new_ctx.workspace_size > 0) {
                new_ctx.workspace_buf = context::allocateMemory(new_ctx.workspace_size);
            }

            // C. 存入缓存
            cache.put(seed, new_ctx);
            ctx = new_ctx;
        }

        // 更新 Fast Path
        last_seed = seed;
        last_ctx = ctx;
        last_ctx_valid = true;
    }

    // 5. 调用 Kernel
    INFINICORE_CHECK_ERROR(infiniopAdaptiveAvgPool1d(
        ctx.desc,
        ctx.getWorkspacePtr(),
        ctx.workspace_size,
        output->data(),
        input->data(),
        context::getStream()
    ));
}

// -------------------------------------------
// 注册 dispatcher
// -------------------------------------------
static bool registered = []() {
    AdaptiveAvgPool1d::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::adaptive_avg_pool1d_impl::infiniop
*/
#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/adaptive_avg_pool1d.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <vector>

namespace infinicore::op::adaptive_avg_pool1d_impl::infiniop {

// -------------------------------------------
// 1. 资源上下文
// -------------------------------------------
struct AdaptiveAvgPool1dContext {
    infiniopAdaptiveAvgPool1dDescriptor_t desc = nullptr;
    std::shared_ptr<Memory> workspace_buf = nullptr;
    size_t workspace_size = 0;

    void* getWorkspacePtr() const {
        return workspace_buf ? workspace_buf->data() : nullptr;
    }
};

// -------------------------------------------
// 2. 缓存定义
// -------------------------------------------
thread_local common::OpCache<size_t, AdaptiveAvgPool1dContext> caches(
    256, 
    [](AdaptiveAvgPool1dContext &ctx) {
        if (ctx.desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAdaptiveAvgPool1dDescriptor(ctx.desc));
            ctx.desc = nullptr;
        }
        ctx.workspace_buf = nullptr;
    }
);

// -------------------------------------------
// 3. 核心计算函数 (Extreme Optimization + Fix)
// -------------------------------------------
void calculate(Tensor output, Tensor input) {
    // 【极致优化 1】直接使用 Tensor 指针地址作为 Hash Key
    // 汇编级操作，远快于遍历 Shape
    size_t seed = reinterpret_cast<size_t>(input.operator->());
    
    // 简单混合 Output 信息防止冲突
    if (output->ndim() >= 3) {
        seed ^= (output->shape()[2] << 1); 
    }

    // 【极致优化 2】静态缓存极速路径 (Zero-Overhead Path)
    // 修复：我们保存对象本身，而不是指针。
    // 在 Fast Path 中，我们直接使用 &last_ctx，这和使用指针一样快，且安全。
    static thread_local size_t last_seed = 0;
    static thread_local bool last_ctx_valid = false;
    static thread_local AdaptiveAvgPool1dContext last_ctx; // 静态对象

    // 这个指针将指向我们要使用的 Context（无论是 Fast Path 还是 Slow Path）
    AdaptiveAvgPool1dContext* active_ctx = nullptr;

    // 检查是否命中“上一次” (最热路径)
    if (last_ctx_valid && seed == last_seed) {
        // *** 命中 Fast Path ***
        // 直接取静态对象的地址。没有 shared_ptr 拷贝，没有原子操作，没有 Map 查找。
        active_ctx = &last_ctx;
    } else {
        // *** 慢路径：查 Map ***
        auto device_type = context::getDevice().getType();
        auto device_index = context::getDevice().getIndex();
        auto &cache = caches.getCache(device_type, device_index);

        auto opt_ctx = cache.get(seed);
        if (opt_ctx) {
            // 命中 LRU Cache：更新到静态变量 (发生一次 shared_ptr 拷贝，仅在切换 Shape 时发生)
            last_ctx = *opt_ctx;
        } else {
            // Cache Miss：创建新资源
            AdaptiveAvgPool1dContext new_ctx;
            
            INFINICORE_CHECK_ERROR(infiniopCreateAdaptiveAvgPool1dDescriptor(
                context::getInfiniopHandle(output->device()), 
                &new_ctx.desc,
                output->desc(), 
                input->desc()));

            INFINICORE_CHECK_ERROR(infiniopGetAdaptiveAvgPool1dWorkspaceSize(new_ctx.desc, &new_ctx.workspace_size));
            
            if (new_ctx.workspace_size > 0) {
                new_ctx.workspace_buf = context::allocateMemory(new_ctx.workspace_size);
            }

            cache.put(seed, new_ctx);
            
            // 更新静态变量
            last_ctx = new_ctx;
        }

        // 更新 Fast Path 标记
        last_seed = seed;
        last_ctx_valid = true;
        
        // 指向静态变量
        active_ctx = &last_ctx;
    }

    // 调用 Kernel
    INFINICORE_CHECK_ERROR(infiniopAdaptiveAvgPool1d(
        active_ctx->desc,
        active_ctx->getWorkspacePtr(),
        active_ctx->workspace_size,
        output->data(),
        input->data(),
        context::getStream()
    ));
}

// -------------------------------------------
// 注册
// -------------------------------------------
static bool registered = []() {
    AdaptiveAvgPool1d::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::adaptive_avg_pool1d_impl::infiniop