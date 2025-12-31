#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::addcmul {

struct Test::Attributes {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> t1;
    std::shared_ptr<Tensor> t2;
    std::shared_ptr<Tensor> out;
    std::shared_ptr<Tensor> ans;
    float value;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    // 校验张量是否存在
    if (tensors.find("input") == tensors.end() ||
        tensors.find("t1") == tensors.end() ||
        tensors.find("t2") == tensors.end() ||
        tensors.find("out") == tensors.end() ||
        tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Addcmul Test: Missing tensors");
    }

    // 获取标量属性 value
    test->_attributes->value = 1.0f; // 默认值
    if (attributes.find("value") != attributes.end()) {
        test->_attributes->value = *reinterpret_cast<float *>(attributes["value"].data());
    }

    test->_attributes->input = tensors["input"];
    test->_attributes->t1 = tensors["t1"];
    test->_attributes->t2 = tensors["t2"];
    test->_attributes->out = tensors["out"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    
    infiniopAddcmulDescriptor_t op_desc;
    
    // 数据迁移至指定设备
    auto input = _attributes->input->to(device, device_id);
    auto t1 = _attributes->t1->to(device, device_id);
    auto t2 = _attributes->t2->to(device, device_id);
    auto out = _attributes->out->to(device, device_id);

    // 创建算子描述符
    CHECK_OR(infiniopCreateAddcmulDescriptor(handle, &op_desc,
                                             out->desc(),
                                             input->desc(),
                                             t1->desc(),
                                             t2->desc(),
                                             _attributes->value),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create addcmul descriptor."));

    // Workspace 处理
    size_t workspace_size;
    CHECK_OR(infiniopGetAddcmulWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));

    // 执行计算
    CHECK_OR(infiniopAddcmul(op_desc, workspace, workspace_size,
                             out->data(),
                             input->data(),
                             t1->data(),
                             t2->data(),
                             nullptr), // stream
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    // 结果验证
    try {
        allClose(out, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 性能测试
    double elapsed_time = benchmark(
        [=]() {
            infiniopAddcmul(op_desc, workspace, workspace_size,
                            out->data(),
                            input->data(),
                            t1->data(),
                            t2->data(),
                            nullptr);
        },
        warm_ups, iterations);

    // 资源清理
    infinirtFree(workspace);
    infiniopDestroyAddcmulDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"value"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "t1", "t2", "out", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"out"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- value: " << _attributes->value << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- t1: " << _attributes->t1->info() << std::endl;
    oss << "- t2: " << _attributes->t2->info() << std::endl;
    oss << "- out: " << _attributes->out->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::addcmul