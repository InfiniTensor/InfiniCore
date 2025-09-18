#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::interpolate_nearest {

struct Test::Attributes {
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> expected_output;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        std::cout << "DEBUG: Name check failed" << std::endl;
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->input = tensors["input"];            // F32 输入数据
    test->_attributes->expected_output = tensors["output"]; // F64 期望结果

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopInterpolateNearestDescriptor_t op_desc;

    auto input = _attributes->input->to(device, device_id);
    auto expected_output = _attributes->expected_output; // F64 期望结果

    // 动态创建实际的输出张量，使用期望结果的形状，但使用输入的数据类型
    auto output_shape = expected_output->shape();
    auto input_dtype = input->ggml_type();

    // 创建输出张量的内存
    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }
    output_size *= ggmlTypeSize(input_dtype);

    auto output_memory = std::make_shared<Memory>(output_size, device, device_id);
    std::vector<ptrdiff_t> output_strides(output_shape.size());

    // 计算连续的步长
    if (output_shape.size() > 0) {
        output_strides[output_shape.size() - 1] = 1;
        for (int i = static_cast<int>(output_shape.size()) - 2; i >= 0; i--) {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }
    }

    auto actual_output = std::make_shared<Tensor>(
        output_memory, 0, output_shape, output_strides, input_dtype);

    // Create operator descriptor
    CHECK_OR(infiniopCreateInterpolateNearestDescriptor(
                 handle, &op_desc,
                 actual_output->desc(),
                 input->desc()),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create op descriptor."));

    // Get workspace size
    size_t workspace_size;
    CHECK_OR(infiniopGetInterpolateNearestWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    // Allocate workspace if needed
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    // Execute interpolate nearest
    CHECK_OR(infiniopInterpolateNearest(
                 op_desc, workspace, workspace_size,
                 actual_output->data(),
                 input->data(),
                 nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    // Verify result - 比较实际输出和期望结果
    try {
        allClose(actual_output, expected_output, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) {
            infinirtFree(workspace);
        }
        infiniopDestroyInterpolateNearestDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // Benchmark
    double elapsed_time = benchmark(
        [=]() {
            infiniopInterpolateNearest(
                op_desc, workspace, workspace_size,
                actual_output->data(),
                input->data(),
                nullptr);
        },
        warm_ups, iterations);

    // Cleanup
    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyInterpolateNearestDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "output"};
}

std::vector<std::string> Test::output_names() {
    return {};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- expected_output: " << _attributes->expected_output->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::interpolate_nearest
