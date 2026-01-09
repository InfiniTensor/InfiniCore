#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::cdist {

struct Test::Attributes {
    std::shared_ptr<Tensor> x1;
    std::shared_ptr<Tensor> x2;
    std::shared_ptr<Tensor> out;
    std::shared_ptr<Tensor> ans;
    double p;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();

    // 1. 校验张量是否存在 (x1, x2, out, ans)
    if (tensors.find("x1") == tensors.end() ||
        tensors.find("x2") == tensors.end() ||
        tensors.find("out") == tensors.end() ||
        tensors.find("ans") == tensors.end()) {
        throw std::runtime_error("Invalid Cdist Test: Missing tensors");
    }

    // 2. 获取标量属性 p (注意 cdist 通常用 double)
    test->_attributes->p = 2.0; // 默认值
    if (attributes.find("p") != attributes.end()) {
        test->_attributes->p = *reinterpret_cast<double *>(attributes["p"].data());
    }

    test->_attributes->x1 = tensors["x1"];
    test->_attributes->x2 = tensors["x2"];
    test->_attributes->out = tensors["out"];
    test->_attributes->ans = tensors["ans"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id, size_t warm_ups, size_t iterations) {
    
    infiniopCdistDescriptor_t op_desc;
    
    // 3. 数据迁移至指定设备 (M x D, N x D)
    auto x1 = _attributes->x1->to(device, device_id);
    auto x2 = _attributes->x2->to(device, device_id);
    auto out = _attributes->out->to(device, device_id);

    // 4. 创建算子描述符
    CHECK_OR(infiniopCreateCdistDescriptor(handle, &op_desc,
                                             out->desc(),
                                             x1->desc(),
                                             x2->desc(),
                                             _attributes->p),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create cdist descriptor."));

    // 5. Workspace 动态内存分配
    size_t workspace_size;
    CHECK_OR(infiniopGetCdistWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size."));
    
    void *workspace;
    CHECK_OR(infinirtMalloc(&workspace, workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace."));

    // 6. 执行计算 (计算 M x N 距离矩阵)
    CHECK_OR(infiniopCdist(op_desc, workspace, workspace_size,
                             out->data(),
                             x1->data(),
                             x2->data(),
                             nullptr), // stream
             return TEST_FAILED(OP_EXECUTION_FAILED, "Failed during execution."));

    // 7. 结果数值验证
    try {
        allClose(out, _attributes->ans, _rtol, _atol);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    // 8. 性能 Benchmark
    double elapsed_time = benchmark(
        [=]() {
            infiniopCdist(op_desc, workspace, workspace_size,
                            out->data(),
                            x1->data(),
                            x2->data(),
                            nullptr);
        },
        warm_ups, iterations);

    // 9. 资源清理
    infinirtFree(workspace);
    infiniopDestroyCdistDescriptor(op_desc);

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"p"};
}

std::vector<std::string> Test::tensor_names() {
    return {"x1", "x2", "out", "ans"};
}

std::vector<std::string> Test::output_names() {
    return {"out"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- p: " << _attributes->p << std::endl;
    oss << "- x1: " << _attributes->x1->info() << std::endl;
    oss << "- x2: " << _attributes->x2->info() << std::endl;
    oss << "- out: " << _attributes->out->info() << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::cdist