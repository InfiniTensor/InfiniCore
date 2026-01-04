#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::binary_cross_entropy_with_logits {

struct Test::Attributes {
    std::shared_ptr<Tensor> logits;
    std::shared_ptr<Tensor> target;
    std::shared_ptr<Tensor> weight;     // 可选
    std::shared_ptr<Tensor> pos_weight; // 可选
    std::shared_ptr<Tensor> out;
    std::shared_ptr<Tensor> ans;
    int reduction; // 0: none, 1: mean, 2: sum
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    // 1. 校验必要张量是否存在
    if (tensors.find("logits") == tensors.end() ||
        tensors.find("target") == tensors.end() ||
        tensors.find("out") == tensors.end() ||
        tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid BCE Test: Missing mandatory tensors");
    }

    // 2. 获取 reduction 属性 (默认为 1: mean)
    test->_attributes->reduction = 1; 
    if (attributes.find("reduction") != attributes.end()) {
        test->_attributes->reduction = *reinterpret_cast<int *>(attributes["reduction"].data());
    }

    // 3. 填充张量（处理可选张量）
    test->_attributes->logits = tensors["logits"];
    test->_attributes->target = tensors["target"];
    test->_attributes->out = tensors["out"];
    test->_attributes->ans = tensors["ans"];
    
    // 如果 tensors 中存在则赋值，否则为 nullptr
    test->_attributes->weight = tensors.count("weight") ? tensors["weight"] : nullptr;
    test->_attributes->pos_weight = tensors.count("pos_weight") ? tensors["pos_weight"] : nullptr;

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    
    infiniopBCEWithLogitsDescriptor_t op_desc;
    
    // 4. 数据迁移
    auto logits = _attributes->logits->to(device, device_id);
    auto target = _attributes->target->to(device, device_id);
    auto out = _attributes->out->to(device, device_id);
    
    // 处理可选张量迁移
    std::shared_ptr<Tensor> weight = (_attributes->weight) ? _attributes->weight->to(device, device_id) : nullptr;
    std::shared_ptr<Tensor> pos_weight = (_attributes->pos_weight) ? _attributes->pos_weight->to(device, device_id) : nullptr;

    // 5. 创建描述符 (注意处理 NULL 描述符)
    auto w_desc = weight ? weight->desc() : nullptr;
    auto pw_desc = pos_weight ? pos_weight->desc() : nullptr;

    CHECK_OR(infiniopCreateBCEWithLogitsDescriptor(handle, &op_desc,
                                                     out->desc(),
                                                     logits->desc(),
                                                     target->desc(),
                                                     w_desc,
                                                     pw_desc,
                                                     static_cast<infiniopReduction_t>(_attributes->reduction)),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create BCE descriptor."));

    // 6. Workspace 管理
    size_t workspace_size;
    CHECK_OR(infiniopGetBCEWithLogitsWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));

    // 7. 执行计算
    auto w_data = weight ? weight->data() : nullptr;
    auto pw_data = pos_weight ? pos_weight->data() : nullptr;

    CHECK_OR(infiniopBCEWithLogits(op_desc, workspace, workspace_size,
                                    out->data(),
                                    logits->data(),
                                    target->data(),
                                    w_data,
                                    pw_data,
                                    nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    // 8. 结果验证
    try {
        allClose(out, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 9. 性能 Benchmark
    double elapsed_time = benchmark(
        [=]() {
            infiniopBCEWithLogits(op_desc, workspace, workspace_size,
                                   out->data(), logits->data(), target->data(),
                                   w_data, pw_data, nullptr);
        },
        warm_ups, iterations);

    // 10. 资源清理
    infinirtFree(workspace);
    infiniopDestroyBCEWithLogitsDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"reduction"};
}

std::vector<std::string> Test::tensor_names() {
    return {"logits", "target", "weight", "pos_weight", "out", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"out"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- reduction: " << _attributes->reduction << std::endl;
    oss << "- logits: " << _attributes->logits->info() << std::endl;
    if (_attributes->weight) oss << "- weight: " << _attributes->weight->info() << std::endl;
    oss << "- out: " << _attributes->out->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::binary_cross_entropy_with_logits