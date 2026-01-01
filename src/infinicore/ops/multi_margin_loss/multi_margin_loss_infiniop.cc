#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/multi_margin_loss.hpp"
#include <infiniop.h>

namespace infinicore::op::multi_margin_loss_impl::infiniop {

// 定义描述符缓存
thread_local common::OpCache<size_t, infiniopMultiMarginLossDescriptor_t> caches(
    100, // capacity
    [](infiniopMultiMarginLossDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyMultiMarginLossDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, Tensor target, Tensor weight, int64_t p, float margin, int64_t reduction) {
    // Tensor 类通常重载了 operator bool()，直接使用 !!weight 或 static_cast<bool>(weight) 检查是否有效
    bool has_weight = static_cast<bool>(weight);
    // hash_combine 不接受 void*。当 weight 为空时，我们传入 size_t(0) 作为替代占位符。
    size_t seed;
    if (has_weight) {
        seed = hash_combine(output, input, target, weight, p, margin, reduction);
    } else {
        // 使用 0 代替 weight 的 ID，避免 void* 导致的编译错误
        seed = hash_combine(output, input, target, size_t(0), p, margin, reduction);
    }

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopMultiMarginLossDescriptor_t desc = nullptr;
    infiniopTensorDescriptor_t weight_desc = nullptr;
    const void* weight_data = nullptr;
    
    // 只有在 weight 有效时才去调用 ->desc() 和 ->data()
    if (has_weight) {
        weight_desc = weight->desc();
        weight_data = weight->data();
    }

    if (!desc_opt) {
        // 3. 创建描述符
        INFINICORE_CHECK_ERROR(infiniopCreateMultiMarginLossDescriptor(
            context::getInfiniopHandle(output->device()), 
            &desc,
            output->desc(), 
            input->desc(), 
            target->desc(), 
            weight_desc, 
            static_cast<int>(p),
            margin,
            static_cast<int>(reduction)
        ));
        
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 4. 获取 Workspace 并执行
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetMultiMarginLossWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopMultiMarginLoss(
        desc, 
        workspace->data(), 
        workspace_size,
        output->data(), 
        input->data(), 
        target->data(), 
        weight_data,
        context::getStream()
    ));
}

static bool registered = []() {
    MultiMarginLoss::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::multi_margin_loss_impl::infiniop