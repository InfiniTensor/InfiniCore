// #include "../../utils.hpp"
// #include "infinicore/common/hash.hpp"
// #include "infinicore/ops/common/cache.hpp"
// #include "infinicore/ops/sqrt.hpp"
// #include <infiniop.h>

// namespace infinicore::op::sqrt_impl::infiniop {

// thread_local common::OpCache<size_t, infiniopSqrtDescriptor_t> caches(
//     100,
//     [](infiniopSqrtDescriptor_t &desc) {
//         if (desc != nullptr) {
//             INFINICORE_CHECK_ERROR(infiniopDestroySqrtDescriptor(desc));
//             desc = nullptr;
//         }
//     });

// void calculate(Tensor output, Tensor input) {
//     size_t seed = hash_combine(output, input);
//     auto device_type = context::getDevice().getType();
//     auto device_index = context::getDevice().getIndex();
//     auto &cache = caches.getCache(device_type, device_index);

//     auto desc_opt = cache.get(seed);
//     infiniopSqrtDescriptor_t desc = nullptr;

//     if (!desc_opt) {
//         INFINICORE_CHECK_ERROR(infiniopCreateSqrtDescriptor(
//             context::getInfiniopHandle(output->device()),
//             &desc,
//             output->desc(),
//             input->desc()));

//         cache.put(seed, desc);
//     } else {
//         desc = *desc_opt;
//     }

//     size_t workspace_size = 0;
//     INFINICORE_CHECK_ERROR(infiniopGetSqrtWorkspaceSize(desc, &workspace_size));
//     std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

//     INFINICORE_CHECK_ERROR(infiniopSqrt(
//         desc,
//         workspace->data(),
//         workspace_size,
//         output->data(),
//         input->data(),
//         context::getStream()));
// }

// static bool registered = []() {
//     Sqrt::dispatcher().registerAll(&calculate, false);
//     return true;
// }();

// } // namespace infinicore::op::sqrt_impl::infiniop

#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/sqrt.hpp"
#include <infiniop.h>

namespace infinicore::op::sqrt_impl::infiniop {

thread_local common::OpCache<size_t, infiniopSqrtDescriptor_t> caches(
    100,
    [](infiniopSqrtDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroySqrtDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    // 优化：只使用 input 的特征作为缓存键，不依赖 output 指针
    // 这样即使 output 是新创建的，也能复用 descriptor
    size_t seed = 0;
    hash_combine(seed, static_cast<size_t>(input->dtype()));
    
    // 手动遍历 shape 和 strides（因为 hash_combine 不支持 std::vector）
    for (Size shape_val : input->shape()) {
        hash_combine(seed, shape_val);
    }
    for (Stride stride_val : input->strides()) {
        hash_combine(seed, static_cast<size_t>(stride_val));
    }
    
    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopSqrtDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateSqrtDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc()));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetSqrtWorkspaceSize(desc, &workspace_size));
    
    // 优化：如果 workspace_size 为 0，跳过分配
    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopSqrt(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Sqrt::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::sqrt_impl::infiniop