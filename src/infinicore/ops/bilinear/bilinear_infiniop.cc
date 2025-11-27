// #include "../../utils.hpp"
// #include "infinicore/common/hash.hpp"
// #include "infinicore/ops/bilinear.hpp"
// #include "infinicore/ops/common/cache.hpp"
// #include <infiniop.h>

// namespace infinicore::op::bilinear_impl::infiniop {

// thread_local common::OpCache<size_t, infiniopBilinearDescriptor_t> caches(
//     100, // capacity
//     [](infiniopBilinearDescriptor_t &desc) {
//         if (desc != nullptr) {
//             INFINICORE_CHECK_ERROR(infiniopDestroyBilinearDescriptor(desc));
//             desc = nullptr;
//         }
//     });

// void calculate(Tensor out, Tensor x1, Tensor x2, Tensor weight, std::optional<Tensor> bias) {
//     size_t seed = hash_combine(out, x1, x2, weight);
//     if (bias) {
//         seed = hash_combine(out, x1, x2, weight,*bias);
//     }

//     auto device_type = context::getDevice().getType();
//     auto device_index = context::getDevice().getIndex();

//     auto &cache = caches.getCache(device_type, device_index);

//     auto desc_opt = cache.get(seed);
//     infiniopBilinearDescriptor_t desc = nullptr;

//     if (!desc_opt) {
//         INFINICORE_CHECK_ERROR(infiniopCreateBilinearDescriptor(
//             context::getInfiniopHandle(out->device()), &desc,
//             out->desc(), x1->desc(), x2->desc(), weight->desc(), 
//             bias ? (*bias)->desc() : nullptr));
//         cache.put(seed, desc);
//     } else {
//         desc = *desc_opt;
//     }

//     size_t workspace_size = 0;
//     INFINICORE_CHECK_ERROR(infiniopGetBilinearWorkspaceSize(desc, &workspace_size));
//     std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

//     INFINICORE_CHECK_ERROR(infiniopBilinear(
//         desc, workspace->data(), workspace_size,
//         out->data(), x1->data(), x2->data(),
//         weight->data(), bias ? (*bias)->data() : nullptr,
//         context::getStream()));
// }

// static bool registered = []() {
//     Bilinear::dispatcher().registerAll(&calculate, false);
//     return true;
// }();

// } // namespace infinicore::op::bilinear_impl::infiniop