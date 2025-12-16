#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/where.hpp"
#include <infiniop.h>

namespace infinicore::op::where_impl::infiniop {

thread_local common::OpCache<size_t, infiniopWhereIndicesDescriptor_t>
    indices_caches(100, [](infiniopWhereIndicesDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyWhereIndicesDescriptor(desc));
            desc = nullptr;
        }
    });

std::vector<Tensor> calculate(Tensor cond) {
    size_t seed = hash_combine(cond);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = indices_caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopWhereIndicesDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateWhereIndicesDescriptor(
            context::getInfiniopHandle(cond->device()), &desc, cond->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(
        infiniopGetWhereIndicesWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    size_t numel = cond->numel();
    int ndim = static_cast<int>(cond->ndim());

    // 先分配最大可能大小的输出（numel），实际大小会在 API 调用后确定
    std::vector<Tensor> outputs;
    std::vector<void *> output_ptrs;

    for (int dim = 0; dim < ndim; ++dim) {
        auto out = Tensor::empty({numel}, DataType::I64, cond->device());
        outputs.push_back(out);
        output_ptrs.push_back(out->data());
    }

    // 调用 infiniop API，它会计算 num_true
    size_t num_true = 0;
    INFINICORE_CHECK_ERROR(infiniopWhereIndices(
        desc, workspace->data(), workspace_size, output_ptrs.data(), cond->data(),
        context::getStream(), &num_true));

    // 同步流以确保计算完成
    context::syncStream();

    // 如果实际 num_true 小于 numel，需要调整输出张量的大小
    // 但 Tensor 可能不支持调整大小，所以我们需要创建新的张量并复制数据
    if (num_true < numel) {
        std::vector<Tensor> resized_outputs;
        for (int dim = 0; dim < ndim; ++dim) {
            auto resized = Tensor::empty({num_true}, DataType::I64, cond->device());
            // 复制前 num_true 个元素
            resized->copy_from(outputs[dim]->narrow({{0, 0, num_true}}));
            resized_outputs.push_back(resized);
        }
        return resized_outputs;
    }

    return outputs;
}

static bool registered = []() {
    WhereIndices::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::where_impl::infiniop
