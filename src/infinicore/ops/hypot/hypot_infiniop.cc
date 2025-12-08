#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/hypot.hpp" // 引入 Hypot 头文件
#include <infiniop.h>

namespace infinicore::op::hypot_impl::infiniop {

// 定义描述符缓存，类型为 infiniopHypotDescriptor_t
thread_local common::OpCache<size_t, infiniopHypotDescriptor_t> caches(
    100, // capacity
    [](infiniopHypotDescriptor_t &desc) {
        if (desc != nullptr) {
            // 销毁 Hypot 描述符
            INFINICORE_CHECK_ERROR(infiniopDestroyHypotDescriptor(desc));
            desc = nullptr;
        }
    });

// 计算函数实现：注意增加了 input_a 和 input_b
void calculate(Tensor output, Tensor input_a, Tensor input_b) {
    // 1. 计算哈希：包含输出和两个输入
    size_t seed = hash_combine(output, input_a, input_b);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopHypotDescriptor_t desc = nullptr;

    if (!desc_opt) {
        // 2. 创建描述符：传入两个输入的 descriptor
        INFINICORE_CHECK_ERROR(infiniopCreateHypotDescriptor(
            context::getInfiniopHandle(output->device()), &desc,
            output->desc(), input_a->desc(), input_b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetHypotWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 3. 执行计算：传入两个输入的数据指针
    INFINICORE_CHECK_ERROR(infiniopHypot(
        desc, 
        workspace->data(), workspace_size,
        output->data(), input_a->data(), input_b->data(), 
        context::getStream()));
}

static bool registered = []() {
    // 注册到 Hypot 的 dispatcher
    Hypot::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::hypot_impl::infiniop