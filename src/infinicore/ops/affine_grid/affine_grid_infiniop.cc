#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/affine_grid.hpp" // 引用算子定义
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::affine_grid_impl::infiniop {

// 定义线程局部缓存，用于存储算子描述符
// Key: size_t (Hash)
// Value: infiniopAffineGridDescriptor_t
thread_local common::OpCache<size_t, infiniopAffineGridDescriptor_t> caches(
    100, // capacity
    [](infiniopAffineGridDescriptor_t &desc) {
        if (desc != nullptr) {
            // 销毁描述符
            INFINICORE_CHECK_ERROR(infiniopDestroyAffineGridDescriptor(desc));
            desc = nullptr;
        }
    });

// 计算函数实现
// 注意：参数必须匹配 schema 定义 -> (Tensor, Tensor, bool)
void calculate(Tensor output, Tensor theta, bool align_corners) {
    // 1. 计算 Hash Key
    // 必须将 align_corners 加入 Hash，因为它改变了计算逻辑（且改变了 Descriptor 内部状态）
    size_t seed = hash_combine(output, theta, align_corners);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    // 获取当前设备对应的缓存
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopAffineGridDescriptor_t desc = nullptr;

    // 2. 缓存未命中，创建描述符
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateAffineGridDescriptor(
            context::getInfiniopHandle(output->device()), 
            &desc,
            output->desc(), 
            theta->desc(), 
            align_corners)); // 传递 align_corners
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    // 3. 获取 Workspace 大小并分配
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetAffineGridWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // 4. 执行计算
    INFINICORE_CHECK_ERROR(infiniopAffineGrid(
        desc, 
        workspace->data(), 
        workspace_size,
        output->data(), 
        theta->data(), 
        context::getStream()));
}

// 5. 注册算子到 Dispatcher
static bool registered = []() {
    AffineGrid::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::affine_grid_impl::infiniop