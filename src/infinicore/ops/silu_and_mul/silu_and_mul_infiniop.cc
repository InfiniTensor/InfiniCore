#include "../infiniop_impl.hpp"
#include "infinicore/ops/silu_and_mul.hpp"

namespace infinicore::op::silu_and_mul_impl::infiniop {

// 定义可缓存的描述符，用于避免频繁创建/销毁 infiniopDescriptor
INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, SiluAndMul, 100);

// 定义图执行模式所需的元数据
struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, output, input;
};

// 预执行阶段：创建描述符并关联张量
void *plan(Tensor output, Tensor input) {
    // 根据张量的描述符（形状、类型等）生成唯一 Hash Seed
    size_t seed = hash_combine(output, input);

    // 获取缓存的描述符或创建新描述符
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, SiluAndMul,
        seed, output->desc(), input->desc());

    // 分配工作空间张量（SwiGLU 如果需要的话，由 descriptor->workspace_size 决定）
    INFINIOP_WORKSPACE_TENSOR(workspace, SiluAndMul, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input)};

    return planned;
}

// 实际执行阶段
void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    // 调用我们在之前步骤中实现的 infiniop 接口
    INFINICORE_CHECK_ERROR(infiniopSiluAndMul(
        planned->descriptor->desc, 
        planned->workspace->data(), 
        planned->workspace->numel(),
        planned->output->data(), 
        planned->input->data(), 
        context::getStream()));
}

// 清理逻辑
void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

// 注册算子到所有支持的设备
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(SiluAndMul, &plan, &run, &cleanup);

} // namespace infinicore::op::silu_and_mul_impl::infiniop
