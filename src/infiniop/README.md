# InfiniOP 开发者文档

InfiniOP 是 InfiniCore 下属的统一底层算子框架，为相同算子在不同平台提供统一的 C 语言多段式接口。

## 开发流程

1. 根据算子定义设计算子接口，在 [`InfiniCore文档`](https://github.com/InfiniTensor/InfiniCore-Documentation) 中添加算子文档。提交文档 PR 。

2. 在 `include/infiniop/` 中添加算子头文件，并 include 到 `include/infiniop.h` 中。每个算子暴露的接口包括：创建算子描述、获取工作空间大小、执行算子、销毁算子描述。比如：

    ```c
    #ifndef __INFINIOP_ADD_API_H__
    #define __INFINIOP_ADD_API_H__

    #include "../operator_descriptor.h"

    typedef struct InfiniopDescriptor *infiniopAddDescriptor_t;

    __C __export infiniStatus_t infiniopCreateAddDescriptor(infiniopHandle_t handle,
                                                            infiniopAddDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t c,
                                                            infiniopTensorDescriptor_t a,
                                                            infiniopTensorDescriptor_t b);

    __C __export infiniStatus_t infiniopGetAddWorkspaceSize(infiniopAddDescriptor_t desc, size_t *size);

    __C __export infiniStatus_t infiniopAdd(infiniopAddDescriptor_t desc,
                                            void *workspace,
                                            size_t workspace_size,
                                            void *c,
                                            const void *a,
                                            const void *b,
                                            void *stream);

    __C __export infiniStatus_t infiniopDestroyAddDescriptor(infiniopAddDescriptor_t desc);

    #endif
    ```

    在任何平台都不需要工作空间的算子也可以不提供获取工作空间大小接口。

3. 在 `src/infiniop/ops/` 中添加算子实现目录，并在目录中创建 `operator.cc` 文件实现头文件中的接口，并根据硬件环境分发至不同平台的核函数。你还可以在目录中创建该算子在全平台通用的代码，比如 `causal_softmax/info.h` 中就包含了对 Causal Softmax 算子在创建算子描述时的一些通用的信息获取和输入输出检查。像逐元素类的算子除了计算内核以外大部分逻辑都是一样的，你可以使用 `src/infiniop/elementwise/` 中的通用代码快速适配算子。

4. 在 `src/infiniop/ops/[op]/[device]/` 中添加平台算子实现。注意复用平台公共代码，比如规约计算（`src/infiniop/reduce/`），开发过程中把未来可复用的代码写在相应公用代码目录里。

    一些 CUDA kernel 可以被多个支持 CUDA 的平台公用，可以考虑在头文件中实现，并在多个源文件中使用。 比如 `mul/cuda/kernel.cuh` 中只有 device 测代码，会被多个支持 CUDA 的平台源代码引用。

5. 算子实现可以成功编译安装后，在 `test/infiniop/` 中添加单测脚本，与 PyTorch 实现进行正确性和性能比较。你可以仿照已有的测试脚本进行开发，以使用各种通用的测试功能。测例应覆盖算子常用类型和形状。测试成功之后可以将测例添加至 `scripts/python_test.py` 一键测试脚本中（这样 Github 自动测试也会包含该算子）。

## 添加 Elementwise 算子（Binary/Unary）

对于逐元素算子（Elementwise Operators），由于重构后的统一框架，添加新算子变得非常简单。以下步骤展示了如何添加一个新的 elementwise 算子。

### Binary Elementwise 算子示例（以 `pow` 为例）

#### 步骤 1: 在 `BinaryMode` 枚举中添加算子

在 `src/infiniop/elementwise/binary.h` 的 `BinaryMode` 枚举中添加新算子：

```cpp
enum class BinaryMode {
    // ... 其他算子
    Pow,  // 添加新算子
    // ...
};
```

#### 步骤 2: 在 `BinaryOp` 模板中添加计算逻辑

在同一文件的 `BinaryOp` 模板中添加对应的计算实现：

```cpp
template <BinaryMode Mode>
struct BinaryOp {
    template <typename T>
    T operator()(const T &a, const T &b) const {
        // ... 其他算子的实现
        else if constexpr (Mode == BinaryMode::Pow) {
            return std::pow(a, b);
        }
        // ...
    }
};
```

如果需要在 CUDA 端优化，还需要在 `namespace cuda` 的 `BinaryOp` 模板中添加对应的 CUDA 实现。

#### 步骤 3: 在 API 头文件中声明算子

在 `include/infiniop/ops/binary_ops_api.h` 中添加：

```cpp
BINARY_OP_API_DECLARE(pow, Pow)
```

#### 步骤 4: 创建算子目录和文件

创建目录结构 `src/infiniop/ops/pow/`，并创建以下文件：

**`operator.cc`** - 主实现文件：
```cpp
#include "../../operator_impl.h"
#include "infiniop/ops/binary_ops_api.h"

#ifdef ENABLE_CPU_API
#include "cpu/pow_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/pow_nvidia.cuh"
#endif

BINARY_OP_IMPL(pow, Pow)
```

**`cpu/pow_cpu.h`** - CPU 头文件：
```cpp
#ifndef __POW_CPU_H__
#define __POW_CPU_H__

#include "../../../elementwise/binary.h"
#include "../../../elementwise/cpu/elementwise_cpu.h"

BINARY_ELEMENTWISE_DESCRIPTOR(pow, cpu, op::elementwise::binary::BinaryMode::Pow)

#endif // __POW_CPU_H__
```

**`cpu/pow_cpu.cc`** - CPU 实现文件：
```cpp
#include "pow_cpu.h"
#include "../../../elementwise/cpu/elementwise_cpu_impl.h"

namespace op::pow::cpu {

ELEMENTWISE_CPU_IMPL_BINARY(pow)

} // namespace op::pow::cpu
```

**`nvidia/pow_nvidia.cuh`** - NVIDIA 头文件：
```cpp
#ifndef __POW_CUDA_API_H__
#define __POW_CUDA_API_H__

#include "../../../elementwise/nvidia/elementwise_nvidia_api.cuh"

ELEMENTWISE_DESCRIPTOR(pow, nvidia)

#endif // __POW_CUDA_API_H__
```

**`nvidia/pow_nvidia.cu`** - NVIDIA 实现文件：
```cpp
#include "../../../elementwise/nvidia/elementwise_nvidia_impl.cuh"

#include "../cuda/kernel.cuh"
#include "pow_nvidia.cuh"

namespace op::pow::nvidia {

ELEMENTWISE_NVIDIA_IMPL_BINARY(pow)

} // namespace op::pow::nvidia
```

**`cuda/kernel.cuh`**（可选）- 如果需要在 CUDA kernel 中实现特殊逻辑：
```cpp
// 通常不需要，除非有特殊的 CUDA 优化需求
```

### Unary Elementwise 算子示例（以 `abs` 为例）

Unary 算子的添加流程与 Binary 类似，主要区别如下：

#### 步骤 1: 在 `UnaryMode` 枚举中添加算子

在 `src/infiniop/elementwise/unary.h` 的 `UnaryMode` 枚举中添加：

```cpp
enum class UnaryMode {
    // ... 其他算子
    Abs,  // 添加新算子
    // ...
};
```

#### 步骤 2: 在 `UnaryOp` 模板中添加计算逻辑

```cpp
template <UnaryMode Mode>
struct UnaryOp {
    template <typename T>
    T operator()(const T &x) const {
        // ... 其他算子的实现
        else if constexpr (Mode == UnaryMode::Abs) {
            if constexpr (std::is_floating_point_v<T>) {
                return std::fabs(x);
            } else {
                return std::abs(x);
            }
        }
        // ...
    }
};
```

#### 步骤 3: 在 API 头文件中声明算子

在 `include/infiniop/ops/unary_ops_api.h` 中添加：

```cpp
UNARY_OP_API_DECLARE(abs, Abs)
```

#### 步骤 4: 创建算子目录和文件

文件结构与 Binary 类似，但使用 `UNARY_` 前缀的宏：

**`operator.cc`**:
```cpp
UNARY_OP_IMPL(abs, Abs)
```

**`cpu/abs_cpu.h`**:
```cpp
UNARY_ELEMENTWISE_DESCRIPTOR(abs, cpu, op::elementwise::unary::UnaryMode::Abs)
```

**`cpu/abs_cpu.cc`**:
```cpp
ELEMENTWISE_CPU_IMPL_UNARY(abs)
```

**`nvidia/abs_nvidia.cu`**:
```cpp
ELEMENTWISE_NVIDIA_IMPL_UNARY(abs)
```

### 总结

添加一个新的 elementwise 算子只需要：

1. ✅ 在对应的 `BinaryMode`/`UnaryMode` 枚举中添加算子
2. ✅ 在 `BinaryOp`/`UnaryOp` 模板中添加计算逻辑
3. ✅ 在 API 头文件中使用宏声明算子
4. ✅ 创建算子目录，使用统一的宏实现各平台代码

**关键优势**：
- 代码复用：所有平台共享相同的实现框架
- 最小改动：只需添加算子特定的计算逻辑
- 统一接口：自动生成标准的 C API
- 易于维护：修改框架代码即可影响所有算子

参考实现：
- Binary: `src/infiniop/ops/pow/`
- Unary: `src/infiniop/ops/abs/`
