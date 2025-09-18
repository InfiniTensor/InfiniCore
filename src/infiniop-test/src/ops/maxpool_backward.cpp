#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::maxpool_backward {

struct Test::Attributes {
    std::shared_ptr<Tensor> grad_output;
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> expected_grad_input;

    std::vector<size_t> kernel_size;
    std::vector<size_t> stride;
    std::vector<size_t> padding;
    bool ceil_mode;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test");
    }

    test->_attributes->grad_output = tensors["grad_output"];
    test->_attributes->input = tensors["input"];
    test->_attributes->expected_grad_input = tensors["grad_input"];

    // 获取池化维度
    size_t pool_ndim = test->_attributes->input->shape().size() - 2;
    if (pool_ndim == 0) {
        throw std::runtime_error("Input tensor must have at least 3 dimensions (N, C, ...)");
    }

    // 解析并广播 kernel_size
    auto kernel_size_data = attributes["kernel_size"];
    if (kernel_size_data.size() % sizeof(int) != 0) {
        throw std::runtime_error("Invalid kernel_size data size");
    }
    size_t kernel_size_count = kernel_size_data.size() / sizeof(int);
    const int *kernel_size_ptr = reinterpret_cast<const int *>(kernel_size_data.data());
    if (kernel_size_count == pool_ndim) {
        test->_attributes->kernel_size.clear();
        for (size_t i = 0; i < kernel_size_count; i++) {
            test->_attributes->kernel_size.push_back(static_cast<size_t>(kernel_size_ptr[i]));
        }
    } else {
        test->_attributes->kernel_size.assign(pool_ndim, static_cast<size_t>(kernel_size_ptr[0]));
    }

    // 解析并广播 stride
    auto stride_data = attributes["stride"];
    if (stride_data.size() % sizeof(int) != 0) {
        throw std::runtime_error("Invalid stride data size");
    }
    size_t stride_count = stride_data.size() / sizeof(int);
    const int *stride_ptr = reinterpret_cast<const int *>(stride_data.data());
    if (stride_count == pool_ndim) {
        test->_attributes->stride.clear();
        for (size_t i = 0; i < stride_count; i++) {
            test->_attributes->stride.push_back(static_cast<size_t>(stride_ptr[i]));
        }
    } else {
        test->_attributes->stride.assign(pool_ndim, static_cast<size_t>(stride_ptr[0]));
    }

    // 解析并广播 padding
    auto padding_data = attributes["padding"];
    if (padding_data.size() % sizeof(int) != 0) {
        throw std::runtime_error("Invalid padding data size");
    }
    size_t padding_count = padding_data.size() / sizeof(int);
    const int *padding_ptr = reinterpret_cast<const int *>(padding_data.data());
    if (padding_count == pool_ndim) {
        test->_attributes->padding.clear();
        for (size_t i = 0; i < padding_count; i++) {
            test->_attributes->padding.push_back(static_cast<size_t>(padding_ptr[i]));
        }
    } else {
        test->_attributes->padding.assign(pool_ndim, static_cast<size_t>(padding_ptr[0]));
    }

    // 解析 ceil_mode
    auto ceil_mode_data = attributes["ceil_mode"];
    if (ceil_mode_data.size() == sizeof(bool)) {
        test->_attributes->ceil_mode = *reinterpret_cast<const bool *>(ceil_mode_data.data());
    } else if (ceil_mode_data.size() == sizeof(uint8_t)) {
        test->_attributes->ceil_mode = *reinterpret_cast<const uint8_t *>(ceil_mode_data.data()) != 0;
    } else {
        throw std::runtime_error("Invalid ceil_mode data size");
    }

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopMaxPoolBackwardDescriptor_t op_desc;

    // 将输入张量移动到指定设备
    auto grad_output = _attributes->grad_output->to(device, device_id);
    auto input = _attributes->input->to(device, device_id);
    auto expected_grad_input = _attributes->expected_grad_input;

    // 获取输入数据类型
    auto input_dtype = input->ggml_type();

    // 手动创建 grad_input 张量（使用期望结果的形状，但使用输入的数据类型）
    auto grad_input_shape = expected_grad_input->shape();

    size_t grad_input_size = 1;
    for (auto dim : grad_input_shape) {
        grad_input_size *= dim;
    }
    grad_input_size *= ggmlTypeSize(input_dtype);

    auto grad_input_memory = std::make_shared<Memory>(grad_input_size, device, device_id);
    std::vector<ptrdiff_t> grad_input_strides(grad_input_shape.size());

    if (grad_input_shape.size() > 0) {
        grad_input_strides[grad_input_shape.size() - 1] = 1;
        for (int i = static_cast<int>(grad_input_shape.size()) - 2; i >= 0; i--) {
            grad_input_strides[i] = grad_input_strides[i + 1] * grad_input_shape[i + 1];
        }
    }

    auto actual_grad_input = std::make_shared<Tensor>(
        grad_input_memory, 0, grad_input_shape, grad_input_strides, input_dtype);

    // 准备参数指针
    void *kernel_size_ptr = _attributes->kernel_size.data();
    void *stride_ptr = _attributes->stride.data();
    void *padding_ptr = _attributes->padding.data();

    auto grad_output_shape = grad_output->shape();
    CHECK_OR(infiniopCreateMaxPoolBackwardDescriptor(
                 handle, &op_desc,
                 actual_grad_input->desc(),
                 grad_output->desc(),
                 input->desc(),
                 kernel_size_ptr,
                 stride_ptr,
                 padding_ptr,
                 _attributes->ceil_mode),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create maxpool_backward descriptor."));

    // 获取工作空间大小
    size_t workspace_size;
    CHECK_OR(infiniopGetMaxPoolBackwardWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    // 分配工作空间
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    // 执行最大池化反向传播
    CHECK_OR(infiniopMaxPoolBackward(
                 op_desc, workspace, workspace_size,
                 actual_grad_input->data(), // void *grad_input
                 grad_output->data(),       // const void *grad_output
                 input->data(),             // const void *input
                 nullptr),                  // void *stream
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during maxpool_backward execution."));

    // 验证结果
    try {
        allClose(actual_grad_input, expected_grad_input, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) {
            infinirtFree(workspace);
        }
        infiniopDestroyMaxPoolBackwardDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 性能测试
    double elapsed_time = benchmark(
        [=]() {
            infiniopMaxPoolBackward(
                op_desc, workspace, workspace_size,
                actual_grad_input->data(),
                grad_output->data(),
                input->data(),
                nullptr);
        },
        warm_ups, iterations);

    // 清理资源
    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyMaxPoolBackwardDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"kernel_size", "stride", "padding", "ceil_mode"};
}

std::vector<std::string> Test::tensor_names() {
    return {"grad_output", "input", "grad_input"};
}

std::vector<std::string> Test::output_names() {
    return {};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- grad_output: " << _attributes->grad_output->info() << std::endl;
    oss << "- input: " << _attributes->input->info() << std::endl;
    oss << "- expected_grad_input: " << _attributes->expected_grad_input->info() << std::endl;

    oss << "- kernel_size: [";
    for (size_t i = 0; i < _attributes->kernel_size.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << _attributes->kernel_size[i];
    }
    oss << "]" << std::endl;

    oss << "- stride: [";
    for (size_t i = 0; i < _attributes->stride.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << _attributes->stride[i];
    }
    oss << "]" << std::endl;

    oss << "- padding: [";
    for (size_t i = 0; i < _attributes->padding.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << _attributes->padding[i];
    }
    oss << "]" << std::endl;

    oss << "- ceil_mode: " << (_attributes->ceil_mode ? "true" : "false") << std::endl;

    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::maxpool_backward
