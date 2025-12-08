#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/addbmm.hpp" 
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>
#include <vector>

namespace infinicore::op::addbmm_impl::infiniop {

// -------------------------------------------
// 1. 定义资源上下文 (绑定 Descriptor 和 Workspace)
// -------------------------------------------
struct AddbmmContext {
    infiniopAddbmmDescriptor_t desc = nullptr;
    std::shared_ptr<Memory> workspace_buf = nullptr;
    size_t workspace_size = 0;

    void* getWorkspacePtr() const {
        return workspace_buf ? workspace_buf->data() : nullptr;
    }
};

// -------------------------------------------
// 2. 统一 LRU 缓存
// -------------------------------------------
thread_local common::OpCache<size_t, AddbmmContext> caches(
    256, 
    [](AddbmmContext &ctx) {
        if (ctx.desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyAddbmmDescriptor(ctx.desc));
            ctx.desc = nullptr;
        }
        // shared_ptr 引用计数归零，自动释放显存
        ctx.workspace_buf = nullptr;
    }
);

// -------------------------------------------
// 3. 计算 Hash (极速指针哈希)
// -------------------------------------------
inline size_t compute_key(const Tensor& output, const Tensor& input, 
                          const Tensor& batch1, const Tensor& batch2, 
                          float beta, float alpha) {
    size_t seed = 0;
    
    // 使用 Tensor 指针地址作为 Hash，速度极快
    // 注意：hash_combine 是 void 返回类型，直接修改 seed 引用
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(output.operator->()));
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(input.operator->()));
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(batch1.operator->()));
    infinicore::hash_combine(seed, reinterpret_cast<size_t>(batch2.operator->()));
    
    // alpha 和 beta 数值变化也会改变计算逻辑，必须参与 Hash
    infinicore::hash_combine(seed, beta);
    infinicore::hash_combine(seed, alpha);
    
    return seed;
}

// -------------------------------------------
// 4. 核心计算函数
// -------------------------------------------
// 参数顺序：beta在前，alpha在后
void calculate(Tensor output, Tensor input, Tensor batch1, Tensor batch2, float beta, float alpha) {
    
    // 1. 计算 Hash
    size_t seed = compute_key(output, input, batch1, batch2, beta, alpha);

    // 2. 极速路径 (Fast Path) 变量
    static thread_local size_t last_seed = 0;
    static thread_local bool last_ctx_valid = false;
    static thread_local AddbmmContext last_ctx; // 静态副本

    AddbmmContext* ctx_ptr = nullptr;

    // 3. 检查 Fast Path
    if (last_ctx_valid && seed == last_seed) {
        ctx_ptr = &last_ctx;
    } else {
        // 4. 慢路径：查 LRU Cache
        auto device_type = context::getDevice().getType();
        auto device_index = context::getDevice().getIndex();
        auto &cache = caches.getCache(device_type, device_index);

        auto opt_ctx = cache.get(seed);
        if (opt_ctx) {
            // 命中：更新 Fast Path 副本
            last_ctx = *opt_ctx;
        } else {
            // 未命中：创建所有资源
            AddbmmContext new_ctx;
            
            // A. 创建 Descriptor (注意 alpha/beta 顺序)
            INFINICORE_CHECK_ERROR(infiniopCreateAddbmmDescriptor(
                context::getInfiniopHandle(output->device()), 
                &new_ctx.desc,
                output->desc(), 
                input->desc(), 
                batch1->desc(), 
                batch2->desc(),
                alpha,  // matmul 系数
                beta)); // input 系数

            // B. 获取并分配 Workspace (仅做一次)
            INFINICORE_CHECK_ERROR(infiniopGetAddbmmWorkspaceSize(new_ctx.desc, &new_ctx.workspace_size));
            
            if (new_ctx.workspace_size > 0) {
                new_ctx.workspace_buf = context::allocateMemory(new_ctx.workspace_size);
            }

            // C. 存入缓存
            cache.put(seed, new_ctx);
            last_ctx = new_ctx;
        }

        // 更新 Fast Path 状态
        last_seed = seed;
        last_ctx_valid = true;
        ctx_ptr = &last_ctx;
    }

    // 5. 执行计算
    INFINICORE_CHECK_ERROR(infiniopAddbmm(
        ctx_ptr->desc, 
        ctx_ptr->getWorkspacePtr(), 
        ctx_ptr->workspace_size,
        output->data(), 
        input->data(), 
        batch1->data(), 
        batch2->data(),
        context::getStream()));
}

// -------------------------------------------
// 5. 注册算子
// -------------------------------------------
static bool registered = []() {
    Addbmm::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::addbmm_impl::infiniop