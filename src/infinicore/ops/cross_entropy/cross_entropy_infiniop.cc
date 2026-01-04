#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
// 引入你之前写的 cross_entropy.hpp 头文件
#include "infinicore/ops/cross_entropy.hpp"
#include "infinicore/ops/common/cache.hpp"
// 引入底层 C-API 头文件 (包含 infiniopCrossEntropy 等声明)
#include <infiniop.h>

namespace infinicore::op::cross_entropy_impl::infiniop {

// 1. 定义描述符缓存
// Key 是 size_t (Hash), Value 是底层 CrossEntropy 描述符
thread_local common::OpCache<size_t, infiniopCrossEntropyDescriptor_t> caches(
    100, // capacity (缓存容量)
    [](infiniopCrossEntropyDescriptor_t &desc) {
        if (desc != nullptr) {
            // 缓存被清理时，调用底层的 Destroy 函数
            INFINICORE_CHECK_ERROR(infiniopDestroyCrossEntropyDescriptor(desc));
            desc = nullptr;
        }
    });

// 2. 实现计算逻辑
// 注意：CrossEntropy 需要 3 个 Tensor 参数 (Output, Logits, Target)
void calculate(Tensor output, Tensor input, Tensor target) {
    // [关键修改] 哈希计算必须包含 target，因为 target 的形状变化也需要新描述符
    size_t seed = hash_combine(output, input, target);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopCrossEntropyDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // [关键修改] 缓存未命中，调用 Create 函数
        // 这里需要传入 output, input(logits), target 三个 Tensor 的描述符
        INFINICORE_CHECK_ERROR(infiniopCreateCrossEntropyDescriptor(
            context::getInfiniopHandle(device), 
            &desc,
            output->desc(), 
            input->desc(), 
            target->desc()  // 新增 target_desc
        ));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 准备 Workspace (临时显存)
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetCrossEntropyWorkspaceSize(desc, &workspace_size));
    // 即使 workspace_size 为 0 (我们之前的 CPU 实现是 0)，这里的 allocateMemory 也能正确处理
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 4. 发射内核
    // [关键修改] 调用底层 Execute 函数，传入 target 数据指针
    INFINICORE_CHECK_ERROR(infiniopCrossEntropy(
        desc, 
        workspace->data(), 
        workspace_size,
        output->data(), 
        input->data(), 
        target->data(), // 新增 target 指针
        context::getStream()
    ));
}

// 5. 自动注册到 Dispatcher
static bool registered = []() {
    // 将 calculate 函数注册到 CrossEntropy 类的分发器中
    // 这里的 calculate 函数签名必须与 CrossEntropy::schema 匹配 (即接受3个 Tensor)
    CrossEntropy::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::cross_entropy_impl::infiniop