#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/elu.hpp"
#include <infiniop.h>

namespace infinicore::op::elu_impl::infiniop {

thread_local common::OpCache<size_t, infiniopEluDescriptor_t> caches(
    100,
    [](infiniopEluDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyEluDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input, float alpha) {
    // 构建缓存键：需要同时考虑 output 和 input 的特征
    // 特别是要区分 inplace (output == input) 和 out-of-place 的情况
    size_t seed = 0;
    hash_combine(seed, static_cast<size_t>(input->dtype()));
    hash_combine(seed, static_cast<size_t>(*reinterpret_cast<const uint32_t *>(&alpha))); // 将 float 转换为 uint32_t 进行哈希
    
    // 检查是否为 inplace 操作
    bool is_inplace = (output->data() == input->data());
    hash_combine(seed, static_cast<size_t>(is_inplace ? 1 : 0));

    // 对于 inplace 操作，只需要 input 的特征（因为 output == input）
    // 对于 out-of-place 操作，需要同时考虑 output 和 input 的特征
    if (is_inplace) {
        // Inplace: 只使用 input 的特征
        for (Size shape_val : input->shape()) {
            hash_combine(seed, shape_val);
        }
        for (Stride stride_val : input->strides()) {
            hash_combine(seed, static_cast<size_t>(stride_val));
        }
    } else {
        // Out-of-place: 需要同时考虑 output 和 input
        for (Size shape_val : output->shape()) {
            hash_combine(seed, shape_val);
        }
        for (Stride stride_val : output->strides()) {
            hash_combine(seed, static_cast<size_t>(stride_val));
        }
        for (Size shape_val : input->shape()) {
            hash_combine(seed, shape_val);
        }
        for (Stride stride_val : input->strides()) {
            hash_combine(seed, static_cast<size_t>(stride_val));
        }
    }

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();
    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopEluDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateEluDescriptor(
            context::getInfiniopHandle(output->device()),
            &desc,
            output->desc(),
            input->desc(),
            alpha));

        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetEluWorkspaceSize(desc, &workspace_size));

    // 如果 workspace_size 为 0，跳过分配
    std::shared_ptr<Memory> workspace;
    void *workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = context::allocateMemory(workspace_size);
        workspace_ptr = workspace->data();
    }

    INFINICORE_CHECK_ERROR(infiniopElu(
        desc,
        workspace_ptr,
        workspace_size,
        output->data(),
        input->data(),
        context::getStream()));
}

static bool registered = []() {
    Elu::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::elu_impl::infiniop