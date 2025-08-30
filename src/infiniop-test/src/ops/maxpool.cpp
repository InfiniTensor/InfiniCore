#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::maxpool {

struct Test::Attributes {
    // 输入张量
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> expected_output;

    // 最大池化参数
    std::vector<size_t> kernel_size;
    std::vector<size_t> stride;
    std::vector<size_t> padding;
    bool ceil_mode;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {

    std::cout << "DEBUG: maxpool::Test::build called" << std::endl;

    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    if (!check_names(attributes, Test::attribute_names()) || !check_names(tensors, Test::tensor_names())) {
        throw std::runtime_error("Invalid Test");
    }
    auto input_tensor = tensors["input"];
    if (input_tensor->ggml_type() == GGML_TYPE_BF16) {
        std::cout << "\n===== C++ Read BF16 Tensor (shape: ";
        for (size_t i = 0; i < input_tensor->shape().size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << input_tensor->shape()[i];
        }
        std::cout << ") =====" << std::endl;

        // 获取原始数据（BF16以uint16_t存储）
        const uint16_t *data_ptr = reinterpret_cast<const uint16_t *>(input_tensor->data());

        // 计算数据元素总数
        size_t total_elements = 1;
        for (auto dim : input_tensor->shape()) {
            total_elements *= dim;
        }

        // 取前10个元素打印（与Python对应）
        size_t print_count = std::min(total_elements, static_cast<size_t>(10));
        for (size_t i = 0; i < print_count; ++i) {
            uint16_t bf16_val = data_ptr[i];

            // 将BF16转换为float
            // BF16格式：1位符号位 + 8位指数 + 7位尾数位
            // 转换方法：将BF16的16位扩展为float的32位
            uint32_t float_bits = static_cast<uint32_t>(bf16_val) << 16;

            // 使用union避免strict-aliasing警告
            union {
                uint32_t bits;
                float value;
            } converter;
            converter.bits = float_bits;
            float float_val = converter.value;

            std::cout << "Index " << i << ": 十进制=" << std::fixed << std::setprecision(6)
                      << float_val << ", 十六进制=0x" << std::hex << std::setw(4) << std::setfill('0')
                      << bf16_val << std::dec << std::endl;
        }
        std::cout << "=================================================\n"
                  << std::endl;
    }

    test->_attributes->input = tensors["input"];
    test->_attributes->expected_output = tensors["output"];

    // 获取池化维度（输入张量维度 - 2，去掉batch和channel维度）
    size_t pool_ndim = test->_attributes->input->shape().size() - 2;
    if (pool_ndim == 0) {
        throw std::runtime_error("Input tensor must have at least 3 dimensions (N, C, ...)");
    }

    // 解析并广播 kernel_size - 修复类型转换
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
        // 直接使用提供的值
        test->_attributes->stride.clear();
        for (size_t i = 0; i < stride_count; i++) {
            test->_attributes->stride.push_back(static_cast<size_t>(stride_ptr[i]));
        }
    } else {
        // 广播单个值到所有维度
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

    // 添加调试信息
    std::cout << "DEBUG: Pool dimensions: " << pool_ndim << std::endl;
    std::cout << "DEBUG: After broadcasting:" << std::endl;
    std::cout << "  kernel_size: [";
    for (size_t i = 0; i < test->_attributes->kernel_size.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << test->_attributes->kernel_size[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "  stride: [";
    for (size_t i = 0; i < test->_attributes->stride.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << test->_attributes->stride[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "  padding: [";
    for (size_t i = 0; i < test->_attributes->padding.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << test->_attributes->padding[i];
    }
    std::cout << "]" << std::endl;

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopMaxPoolDescriptor_t op_desc;

    auto input = _attributes->input->to(device, device_id);
    auto expected_output = _attributes->expected_output;

    auto input_dtype = input->ggml_type();

    auto output_shape = expected_output->shape();

    std::cout << "DEBUG: output shape: [";
    for (size_t i = 0; i < output_shape.size(); i++) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << output_shape[i];
    }
    std::cout << "]" << std::endl;

    size_t output_size = 1;
    for (auto dim : output_shape) {
        output_size *= dim;
    }
    output_size *= ggmlTypeSize(input_dtype);

    auto output_memory = std::make_shared<Memory>(output_size, device, device_id);
    std::vector<ptrdiff_t> output_strides(output_shape.size());

    if (output_shape.size() > 0) {
        output_strides[output_shape.size() - 1] = 1;
        for (int i = output_shape.size() - 2; i >= 0; i--) {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }
    }

    auto actual_output = std::make_shared<Tensor>(
        output_memory, 0, output_shape, output_strides, input_dtype);

    std::cout << "DEBUG: actual_output created successfully" << std::endl;

    // 准备参数指针
    void *kernel_size_ptr = _attributes->kernel_size.data();
    void *stride_ptr = _attributes->stride.data();
    void *padding_ptr = _attributes->padding.data();

    // 调试信息
    std::cout << "DEBUG: MaxPool parameters:" << std::endl;
    std::cout << "  kernel_size: [";
    for (size_t i = 0; i < _attributes->kernel_size.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << _attributes->kernel_size[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "  stride: [";
    for (size_t i = 0; i < _attributes->stride.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << _attributes->stride[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "  padding: [";
    for (size_t i = 0; i < _attributes->padding.size(); ++i) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << _attributes->padding[i];
    }
    std::cout << "]" << std::endl;

    std::cout << "  ceil_mode: " << (_attributes->ceil_mode ? "true" : "false") << std::endl;

    std::cout << "DEBUG: input shape: [";
    auto input_shape = input->shape();
    for (size_t i = 0; i < input_shape.size(); i++) {
        if (i > 0) {
            std::cout << ", ";
        }
        std::cout << input_shape[i];
    }
    std::cout << "], dtype: " << input->ggml_type() << std::endl;

    // 创建算子描述符
    std::cout << "DEBUG: Creating max pool descriptor..." << std::endl;
    CHECK_OR(infiniopCreateMaxPoolDescriptor(
                 handle, &op_desc,
                 actual_output->desc(),
                 input->desc(),
                 kernel_size_ptr,
                 stride_ptr,
                 padding_ptr,
                 _attributes->ceil_mode),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create maxpool descriptor."));

    // 获取工作空间大小
    size_t workspace_size;
    CHECK_OR(infiniopGetMaxPoolWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));

    // 分配工作空间
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));
    }

    // 执行最大池化
    CHECK_OR(infiniopMaxPool(
                 op_desc, workspace, workspace_size,
                 actual_output->data(),
                 input->data(),
                 nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during maxpool execution."));

    // 验证结果
    try {
        allClose(actual_output, expected_output, _rtol, _atol);
    } catch (const std::exception &e) {
        if (workspace) {
            infinirtFree(workspace);
        }
        infiniopDestroyMaxPoolDescriptor(op_desc);
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 性能测试
    double elapsed_time = benchmark(
        [=]() {
            infiniopMaxPool(
                op_desc, workspace, workspace_size,
                actual_output->data(),
                input->data(),
                nullptr);
        },
        warm_ups, iterations);

    // 清理资源
    if (workspace) {
        infinirtFree(workspace);
    }
    infiniopDestroyMaxPoolDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"kernel_size", "stride", "padding", "ceil_mode"};
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

} // namespace infiniop_test::maxpool
