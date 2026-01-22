#include "infinicore/ops/silu_and_mul.hpp"
#include "../../utils.hpp"

namespace infinicore::op {

// 实现分发器
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(SiluAndMul);

// 构造函数：校验设备并分发
SiluAndMul::SiluAndMul(Tensor out, Tensor x) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, x);
    // 根据设备类型（如 Moore, Cuda 等）路由到具体的实现
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(), out, x);
}

// 执行接口：在图模式下记录或在即时模式下运行
void SiluAndMul::execute(Tensor out, Tensor x) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(SiluAndMul, out, x);
}

// 非原地接口：负责推导输出形状并分配内存
Tensor silu_and_mul(Tensor x) {
    Shape shape = x->shape();
    size_t ndim = x->ndim();
    
    // SwiGLU 逻辑：输出最后一维是输入的一半
    if (shape[ndim - 1] % 2 != 0) {
        throw std::runtime_error("SiluAndMul input last dim must be even.");
    }
    shape[ndim - 1] /= 2;

    // 创建输出张量
    auto out = Tensor::empty(shape, x->dtype(), x->device());
    silu_and_mul_(out, x);
    return out;
}

// 原地/指定输出接口
void silu_and_mul_(Tensor out, Tensor x) {
    SiluAndMul::execute(out, x);
}

} // namespace infinicore::op
