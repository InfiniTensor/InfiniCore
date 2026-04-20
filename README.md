# InfiniCore

[![Doc](https://img.shields.io/badge/Document-ready-blue)](https://github.com/InfiniTensor/InfiniCore-Documentation)
[![CI](https://github.com/InfiniTensor/InfiniCore/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/InfiniTensor/InfiniCore/actions)
[![license](https://img.shields.io/github/license/InfiniTensor/InfiniCore)](https://mit-license.org/)
![GitHub repo size](https://img.shields.io/github/repo-size/InfiniTensor/InfiniCore)
![GitHub code size in bytes](https://img.shields.io/github/languages/code-size/InfiniTensor/InfiniCore)

[![GitHub Issues](https://img.shields.io/github/issues/InfiniTensor/InfiniCore)](https://github.com/InfiniTensor/InfiniCore/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/InfiniTensor/InfiniCore)](https://github.com/InfiniTensor/InfiniCore/pulls)
![GitHub contributors](https://img.shields.io/github/contributors/InfiniTensor/InfiniCore)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/InfiniTensor/InfiniCore)

InfiniCore 是一个跨平台统一编程工具集，为不同芯片平台的功能（包括计算、运行时、通信等）提供统一 C 语言接口。目前支持的硬件和后端包括：

- CPU；
- CUDA
  - 英伟达 GPU；
  - 摩尔线程 GPU；
  - 天数智芯 GPU；
  - 沐曦 GPU；
  - 海光 DCU；
  - 阿里 PPU；
- 华为昇腾 NPU；
- 寒武纪 MLU；
- 昆仑芯 XPU；

API 定义以及使用方式详见 [`InfiniCore文档`](https://github.com/InfiniTensor/InfiniCore-Documentation)。

## 项目依赖

- [Xmake](https://xmake.io/)：跨平台自动构建工具，用于编译 InfiniCore 项目。
- [gcc-11](https://gcc.gnu.org/) 以上或者 [clang-16](https://clang.llvm.org/)：基础编译器，需要支持 C++ 17 标准。
- [Python>=3.10](https://www.python.org/)
  - [PyTorch](https://pytorch.org/)：可选，用于对比测试。
- 各个硬件平台的工具包：请参考各厂商官方文档（如英伟达平台需要安装 CUDA Toolkit）。

## 配置和使用

### 一、克隆项目

由于仓库中含有子模块（如 `spdlog` / `nlohmann_json`），所以在克隆时请添加 `--recursive` 或 `--recurse-submodules`，如：

```shell
git clone --recursive https://github.com/InfiniTensor/InfiniCore.git
```

或者在普通克隆后进行更新：

```shell
git submodule update --init --recursive
```

> 注：InfLLM-V2 CUDA kernels（`infllmv2_cuda_impl`）为**可选依赖**，不会随仓库子模块默认拉取。
> 如需启用 `--infllmv2`（见下文），请自行在任意目录克隆/编译该项目，并将生成的 `infllm_v2/*.so` 路径传给 xmake；
> 或者将其手动放到 `InfiniCore/third_party/infllmv2_cuda_impl` 后再使用 `--infllmv2=y` 走自动探测。

配置`INFINI_ROOT` 和 `LD_LIBRARY_PATH` 环境变量。  
默认`INFINI_ROOT`为`$HOME/.infini`，可以使用以下命令自动配置：

```shell
source scripts/set_env_linux.sh
```

如果你需要在本地开发九齿算子（即需要对九齿算子库进行修改），推荐单独克隆[九齿算子库](https://github.com/InfiniTensor/ntops)，并从本地安装：

```shell
git clone https://github.com/InfiniTensor/ntops.git
cd ntops
pip install -e .
```

### 二、编译安装

InfiniCore 项目主要包括：

1. 底层 C 库（InfiniOP/InfiniRT/InfiniCCL）：[`一键安装`](#一键安装底层库)|[`手动安装`](#手动安装底层库)；
2. InfiniCore C++ 库：[`安装指令`](#2-安装-c-库)
3. InfiniCore Python 包（依赖[九齿算子库](https://github.com/InfiniTensor/ntops)）：[`安装指令`](#3-安装-python-包)

三者需要按照顺序进行编译安装。

#### 1. 安装底层库

##### 一键安装底层库

在 `script/` 目录中提供了 `install.py` 安装脚本。使用方式如下：

```shell
cd InfiniCore

python scripts/install.py [XMAKE_CONFIG_FLAGS]
```

参数 `XMAKE_CONFIG_FLAGS` 是 xmake 构建配置，可配置下列可选项：

| 选项                     | 功能                              | 默认值
|--------------------------|-----------------------------------|:-:
| `--omp=[y\|n]`           | 是否使用 OpenMP                   | y
| `--cpu=[y\|n]`           | 是否编译 CPU 接口实现             | y
| `--nv-gpu=[y\|n]`        | 是否编译英伟达 GPU 接口实现       | n
| `--ascend-npu=[y\|n]`    | 是否编译昇腾 NPU 接口实现         | n
| `--cambricon-mlu=[y\|n]` | 是否编译寒武纪 MLU 接口实现       | n
| `--metax-gpu=[y\|n]`     | 是否编译沐曦 GPU 接口实现         | n
| `--use-mc=[y\|n]`        | 是否沐曦 GPU 接口实现使用maca SDK | n
| `--moore-gpu=[y\|n]`     | 是否编译摩尔线程 GPU 接口实现     | n
| `--iluvatar-gpu=[y\|n]`  | 是否编译天数 GPU 接口实现         | n
| `--qy-gpu=[y\|n]`        | 是否编译QY GPU 接口实现           | n
| `--hygon-dcu=[y\|n]`     | 是否编译海光 DCU 接口实现         | n
| `--kunlun-xpu=[y\|n]`    | 是否编译昆仑 XPU 接口实现         | n
| `--ali-ppu=[y\|n]`       | 是否编译阿里 PPU 接口实现         | n
| `--ninetoothed=[y\|n]`   | 是否编译九齿实现                 | n
| `--ccl=[y\|n]`           | 是否编译 InfiniCCL 通信库接口实现 | n
| `--graph=[y\|n]`         | 是否编译 cuda graph 接口实现      | n
| `--aten=[y\|n]`          | 是否链接 ATen / PyTorch（用于部分算子/对比测试） | n
| `--infllmv2=[y\|PATH]`   | **可选**：启用 InfLLM-V2 attention（需 `--aten=y`）。值为 `y`（探测 `third_party/infllmv2_cuda_impl`）或指向 `libinfllm_v2.so` / `infllmv2_cuda_impl` 根目录 | (空)

##### 手动安装底层库

0. 生成九齿算子（可选）

   - 克隆并安装[九齿算子库](https://github.com/InfiniTensor/ntops)。

   - 在 `InfiniCore` 文件夹下运行以下命令 AOT 编译库中的九齿算子：

     ```shell
     PYTHONPATH=${PYTHONPATH}:src python scripts/build_ntops.py
     ```

1. 项目配置

   windows系统上，建议使用`xmake v2.8.9`编译项目。
   - 查看当前配置

     ```shell
     xmake f -v
     ```

   - 配置 CPU（默认配置）

     ```shell
     xmake f -cv
     ```

   - 配置加速卡

     ```shell
     # 英伟达
     # 可以指定 CUDA 路径， 一般环境变量为 `CUDA_HOME` 或者 `CUDA_ROOT`
     # window系统：--cuda="%CUDA_HOME%"
     # linux系统：--cuda=$CUDA_HOME
     xmake f --nv-gpu=true --cuda=$CUDA_HOME -cv

     # QY
     xmake f --qy-gpu=true --cuda=$CUDA_HOME -cv

     # 寒武纪
     xmake f --cambricon-mlu=true -cv

     # 华为昇腾
     xmake f --ascend-npu=true -cv
     ```

##### 试验功能 -- 使用flash attention库中的算子

  ```shell

  # 在third_party目录拉取cutlass和flash attn库的源码(不需要--recursive)

  # 设置cutlass路径的环境变量CUTLASS_ROOT(部分环境可选)
      export CUTLASS_ROOT=<path-to>/InfiniCore/third_party/cutlass

  # xmake配置环节额外打开 --aten 开关，并设置 --flash-attn 库位置，例(cuda路径部分环境可使用默认)：
      xmake f --nv-gpu=y --ccl=y --cuda=$CUDA_HOME --aten=y --flash-attn=<path-to>/InfiniCore/third_party/flash-attention -cv

  # 设置额外的环境变量
      export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

  # flash attenion库会伴随infinicore_cpp_api一同编译安装

  ```

##### 试验功能 -- 使用 InfLLM-V2 CUDA kernels（可选）

InfLLM-V2 的 varlen/kvcache attention 需要额外的 CUDA kernels（`infllm_v2/*.so`）。该依赖为**可选**，需要你自行克隆并编译。

如果你希望将 `infllmv2_cuda_impl` 放在本仓库 `third_party/` 下（但不作为子模块管理），可以按以下方式拉取并编译，然后使用 `--infllmv2=y` 让 xmake 自动探测：

```bash
cd InfiniCore

# Core submodules only (InfLLM-v2 不作为子模块强制拉取)
git submodule sync third_party/spdlog third_party/nlohmann_json
git submodule update --init third_party/spdlog third_party/nlohmann_json

# Fetch InfLLM-v2 into third_party if missing (NOT a git submodule).
INFLLMV2_DIR="$PWD/third_party/infllmv2_cuda_impl"
if [ ! -d "$INFLLMV2_DIR/.git" ]; then
  rm -rf "$INFLLMV2_DIR"
  git clone --depth 1 -b minicpm_sala_patches --recurse-submodules \
    https://github.com/Ceng23333/infllmv2_cuda_impl.git "$INFLLMV2_DIR"
fi

cd "$INFLLMV2_DIR"
git submodule update --init --recursive
python3 setup.py install

cd ..
python3 scripts/install.py --root --nv-gpu=y --cuda_arch=sm_80 --aten=y --infllmv2=y --ccl=y
xmake build -r _infinicore
xmake install _infinicore

export PYTHONPATH="$PWD/test/infinicore:$PWD/python:${PYTHONPATH:-}"
python3 "$PWD/test/infinicore/ops/infllmv2_attention.py" --nvidia
python3 "$PWD/test/infinicore/ops/simple_gla_prefill.py" --nvidia
python3 "$PWD/test/infinicore/ops/simple_gla_decode_recurrent.py" --nvidia
```

1. 构建 `infllmv2_cuda_impl`（示例，路径可自定义）：

```shell
git clone <your infllmv2_cuda_impl repo url> /abs/path/to/infllmv2_cuda_impl
cd /abs/path/to/infllmv2_cuda_impl
python setup.py install
```

2. 配置并编译 InfiniCore（需要 `--aten=y`）：

```shell
# 方式 A：直接给 .so 的绝对路径（推荐，更明确）
xmake f --nv-gpu=y --aten=y --infllmv2=/abs/path/to/libinfllm_v2.so -cv
xmake build && xmake install

# 方式 B：给 infllmv2_cuda_impl 根目录（会探测 build/lib.*/infllm_v2/*.so）
xmake f --nv-gpu=y --aten=y --infllmv2=/abs/path/to/infllmv2_cuda_impl -cv
xmake build && xmake install
```

运行时需要能找到该 `libinfllm_v2.so`（例如它的目录已在 rpath / `LD_LIBRARY_PATH` 中）。本项目在链接时会尝试写入 rpath 到对应目录，因此通常无需 `LD_PRELOAD`。

2. 编译安装

   默认安装路径为 `$HOME/.infini`。

   ```shell
   xmake build && xmake install
   ```

#### 2. 安装 C++ 库

```shell
xmake build _infinicore
xmake install _infinicore
```

#### 3. 安装 Python 包

```shell
pip install .
```

或

```shell
pip install -e .
```

注：开发时建议加入 `-e` 选项（即 `pip install -e .`），这样对 `python/infinicore` 做的更改将会实时得到反映，同时对 C++ 层所做的修改也只需要运行 `xmake build _infinicore && xmake install _infinicore` 便可以生效。

### 三、运行测试

#### 运行 InfiniCore Python算子接口测试

```bash
# 测试单算子
python test/infinicore/ops/[operator].py [--bench | --debug | --verbose] [--cpu | --nvidia | --cambricon | --ascend | --iluvatar | --metax | --moore | --kunlun | --Hygon | --ali]
# 测试全部算子
python test/infinicore/run.py [--bench | --debug | --verbose] [--cpu | --nvidia | --cambricon | --ascend | --iluvatar | --metax | --moore | --kunlun | --ali]
```

使用 -h 查看更多参数。

#### 运行 InfiniOP 算子测试

```shell
# 测试单算子
python test/infiniop/[operator].py [--cpu | --nvidia | --cambricon | --ascend | --iluvatar | --metax | --moore | --kunlun | --Hygon | --ali]
# 测试全部算子
python scripts/python_test.py [--cpu | --nvidia | --cambricon | --ascend | --iluvatar | --metax | --moore | --kunlun | --Hygon | --ali]
```

#### 通信库（InfiniCCL）测试

编译（需要先安装底层库中的 InfiniCCL 库）：

```shell
xmake build infiniccl-test
```

在英伟达平台运行测试（会自动使用所有可见的卡）：

```shell
infiniccl-test --nvidia
```

## 如何开源贡献

见 [`InfiniCore开发者手册`](DEV.md)。
