#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::atanh {
struct Test::Attributes {
    std::shared_ptr<Tensor> a;   // 输入
    std::shared_ptr<Tensor> y;   // 输出
    std::shared_ptr<Tensor> ans; // 参考结果
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    
    // atanh 只需要 a (input), y (output) 和 ans (reference)
    if (tensors.find("a") == tensors.end()
        || tensors.find("y") == tensors.end()
        || tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Atanh Test: Missing tensors.");
    }

    test->_attributes->a = tensors["a"];
    test->_attributes->y = tensors["y"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    
    infiniopAtanhDescriptor_t op_desc;
    auto a = _attributes->a->to(device, device_id);
    auto y = _attributes->y->to(device, device_id);

    // 调用修正后的 4 参数版本接口 (handle, desc, y, a)
    CHECK_OR(infiniopCreateAtanhDescriptor(handle, &op_desc,
                                           y->desc(),
                                           a->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create atanh descriptor."));
    
    size_t workspace_size;
    CHECK_OR(infiniopGetAtanhWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));

    // 执行计算 (移除 b 相关的参数)
    CHECK_OR(infiniopAtanh(op_desc, workspace, workspace_size,
                           y->data(),
                           a->data(),
                           nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during atanh execution."));

    // 验证结果
    try {
        allClose(y, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 性能测试 (Benchmark)
    double elapsed_time = 0.;
    elapsed_time = benchmark(
        [=]() {
            infiniopAtanh(
                op_desc, workspace, workspace_size,
                y->data(),
                a->data(),
                nullptr);
        },
        warm_ups, iterations);

    // 释放资源 (可选：根据框架决定是否在此释放 op_desc)
    // infiniopDestroyAtanhDescriptor(op_desc);
    // infinirtFree(workspace);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"a", "y", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"y"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- a: " << _attributes->a->info() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::atanh