#ifndef __INFINIOP_ELEMENTWISE_ASCEND_H__
#define __INFINIOP_ELEMENTWISE_ASCEND_H__

#include "../../devices/ascend/common_ascend.h"
#include "../elementwise.h"
#include <tuple>
#include <utility>

namespace op::elementwise::ascend {
// template <typename... TensorDescs>
class DeviceImpl final {
    struct Opaque;
    std::shared_ptr<Opaque> _opaque;

    DeviceImpl(std::shared_ptr<Opaque> opaque) : _opaque(std::move(opaque)) {}

public:
    ~DeviceImpl() = default;

    template <typename... Args>
    static utils::Result<DeviceImpl *> create(Args &&...args);
}

template <typename... TensorDescs,
          std::enable_if_t<(std::is_same_v<TensorDescs, aclnnTensorDescriptor_t> && ...), int> = 0>
struct Opaque {
    mutable aclOpExecutor *executor;
    size_t workspaceSize;
    TensorDescs outTensorDesc;
    std::tuple<TensorDescs...> inTensorDescs;

    explicit Opaque(aclOpExecutor *exec, size_t wsSize, TensorDescs outDesc, TensorDescs... descs)
        : executor(exec), workspaceSize(wsSize), outTensorDesc(outDesc), inTensorDescs(std::forward<TensorDescs>(descs)...) {}

    ~Opaque() {
        aclDestroyAclOpExecutor(executor);
        delete outDesc;
        // 遍历元组并释放每个 Tensor 描述符
        std::apply([](auto &&...args) {
            (..., (delete args));
        },
                   inTensorDescs);
    }

    // 获取输出 Tensor 描述符
    template <size_t N>
    auto getInTensor() -> decltype(auto) {
        return std::get<N>(inTensorDescs);
    }
}

} // namespace op::elementwise::ascend

#endif // __INFINIOP_ELEMENTWISE_ASCEND_H__
