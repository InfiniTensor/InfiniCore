#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::cross_entropy_loss {

struct Test::Attributes {
    // 输入张量
    std::shared_ptr<Tensor> logits;
    std::shared_ptr<Tensor> target;
    std::shared_ptr<Tensor> loss;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    // 检查必需的张量是否存在
    if (!check_names(tensors, Test::tensor_names()) || !check_names(attributes, Test::attribute_names())) {
        throw std::runtime_error("Invalid Test: Missing required tensors.");
    }

    test->_attributes->logits = tensors["logits"];
    test->_attributes->target = tensors["target"];
    test->_attributes->loss = tensors["loss"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopCrossEntropyLossDescriptor_t op_desc;

    // 将输入张量移动到目标设备
    auto logits = _attributes->logits->to(device, device_id);
    auto target = _attributes->target->to(device, device_id);
    auto loss = _attributes->loss;

    // 根据期望输出的形状创建实际输出张量
    auto output_shape = loss->shape();
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }
    output_size *= ggmlTypeSize(logits->ggml_type());

    auto output_memory = std::make_shared<Memory>(output_size, device, device_id);
    std::vector<ptrdiff_t> output_strides(static_cast<size_t>(output_shape.size()));
    if (output_shape.size() > 0) {
        output_strides[output_shape.size() - 1] = 1;
        for (int i = static_cast<int>(output_shape.size()) - 2; i >= 0; i--) {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }
    }
    auto actual_output = std::make_shared<Tensor>(
        output_memory, 0, output_shape, output_strides, logits->ggml_type());

    // 1. 创建算子描述符
    CHECK_OR(infiniopCreateCrossEntropyLossDescriptor(
                 handle, &op_desc,
                 actual_output->desc(),
                 logits->desc(),
                 target->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create cross entropy loss descriptor."));

    // 2. 获取并分配工作空间
    size_t workspace_size;
    CHECK_OR(infiniopGetCrossEntropyLossWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    // 3. 执行计算
    CHECK_OR(infiniopCrossEntropyLoss(
                 op_desc, workspace, workspace_size,
                 actual_output->data(),
                 logits->data(),
                 target->data(),
                 nullptr), // stream
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during cross entropy loss execution."));

    // 4. 验证结果
    try {
        allClose(actual_output, loss, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) {
            infinirtFree(workspace);
        }
        infiniopDestroyCrossEntropyLossDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 5. 性能测试
    double elapsed_time = benchmark(
        [=]() {
            infiniopCrossEntropyLoss(
                op_desc, workspace, workspace_size,
                actual_output->data(),
                logits->data(),
                target->data(),
                nullptr); // stream
        },
        warm_ups, iterations);

    // 6. 清理资源
    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyCrossEntropyLossDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

// 定义算子需要的属性名列表
std::vector<std::string> Test::attribute_names() {
    return {}; // CrossEntropyLoss 没有额外的属性
}

// 定义算子需要的张量名列表
std::vector<std::string> Test::tensor_names() {
    return {"logits", "target", "loss"};
}

std::vector<std::string> Test::output_names() {
    return {};
}

// 打印测试信息的辅助函数
std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- logits: " << _attributes->logits->info() << std::endl;
    oss << "- target: " << _attributes->target->info() << std::endl;
    oss << "- loss: " << _attributes->loss->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::cross_entropy_loss
