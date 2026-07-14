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

由于仓库中含有子模块，所以在克隆时请添加 `--recursive` 或 `--recurse-submodules`，如：

```shell
git clone --recursive https://github.com/InfiniTensor/InfiniCore.git
```

或者在普通克隆后进行更新：

```shell
git submodule update --init --recursive
```

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

在 `script/` 目录中提供了 `install` 安装脚本。使用方式如下：

linux 系统：在`terminal`中执行
```shell
cd InfiniCore
source scripts/install.sh [XMAKE_CONFIG_FLAGS]
# 例如
# source scripts/install.sh    配置 CPU（默认配置）
# source scripts/install.sh --nv-gpu=true --cuda=$CUDA_HOME   配置 英伟达 加速卡
```
windows 系统：在`CMD`中执行
```shell
cd InfiniCore
python scripts/install.py [XMAKE_CONFIG_FLAGS]
# 例如
# python scripts/install.py    配置 CPU（默认配置）
# python scripts/install.py --nv-gpu=true --cuda=\""%CUDA_HOME%"\"   配置 英伟达 加速卡
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
     # 需要指定环境变量QY_ROOT来确认库所在位置，比如说export QY_ROOT=/usr/local/XX
     xmake f --qy-gpu=true --cuda=$CUDA_HOME -cv

     # 寒武纪
     xmake f --cambricon-mlu=true -cv

     # 华为昇腾
     xmake f --ascend-npu=true -cv
     ```

##### 试验功能 -- 使用英伟达平台 flash attention 库中的算子

  ```shell

  # 该功能依赖 flash-attention 和 cutlass，默认不随仓库递归拉取。
  # 对应子模块固定为以下提交：
      ## flash-attention commit: 10846960ca0793b993446f6dbaf696479c127a9d
      ## cutlass commit: 087c84df83d254b5fb295a7a408f1a1d554085cf

  # 若需启用英伟达平台 flash attention 能力，请手动初始化对应子模块：
      git -c submodule.third_party/flash-attention.update=checkout \
          -c submodule.third_party/cutlass.update=checkout \
          submodule update --init third_party/flash-attention third_party/cutlass

  # 上述命令只初始化这两个顶层子模块，并会切换到仓库记录的固定提交。

  # 设置cutlass路径的环境变量CUTLASS_HOME(部分环境可选)
      export CUTLASS_HOME=<path-to>/InfiniCore/third_party/cutlass

  # xmake配置环节额外打开 --aten 开关，并设置 --flash-attn 库位置，例(cuda路径部分环境可使用默认)：
      xmake f --nv-gpu=y --ccl=y --aten=y [--graph=y] [--cuda=$CUDA_HOME] --flash-attn=<path-to>/InfiniCore/third_party/flash-attention -cv

  # 设置额外的环境变量
      export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

  # flash attention库会伴随infinicore_cpp_api一同编译安装

  ```

##### 试验功能 -- 使用摩尔线程开源 mate 提供的 flash attention 能力
  ```shell
  #该功能依赖摩尔线程开源项目 mate（https://github.com/MooreThreads/mate） v0.1.3 版本，默认不随仓库递归拉取。
  
  #若需启用摩尔线程开源项目 mate 提供的 flash attention 能力，请手动初始化对应子模块：
  git -c submodule.third_party/mate.update=checkout submodule update --init --recursive third_party/mate

  #随后参考 mate v0.1.3 README 进行编译，之后在 xmake 配置环节额外打开 --aten 开关和 --flash-attn 使用 mate 提供的 flash attention 能力，可参考：
  xmake f --moore-gpu=y --aten=y --flash-attn=y -cv
  ```


##### 试验功能 -- 使用海光 DCU 平台预编译 flash-attn 能力

  ```shell
  # 海光 DCU 不在 InfiniCore 内现场编译 flash-attn，而是链接 Python 环境中已经安装好的 flash-attn 运行库。
  # 因此 --flash-attn 需要指向 flash-attn 的 Python 安装根目录，通常是当前 Python 的 site-packages/dist-packages 目录。
  # 该目录下需要能找到以下两个文件：
  #   1. flash_attn_2_cuda*.so
  #   2. flash_attn/lib/libflash_attention.so
  # 例如：
  #   /usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so
  #   /usr/local/lib/python3.10/dist-packages/flash_attn/lib/libflash_attention.so

  # 若 flash_attn_2_cuda*.so 不在 --flash-attn 指定目录下，可通过 FLASH_ATTN_2_CUDA_SO 显式指定。
  export FLASH_ATTN_2_CUDA_SO=/usr/local/lib/python3.10/dist-packages/flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so

  # xmake 配置环节需要同时打开海光 DCU、ATen 和 flash-attn：
  xmake f --hygon-dcu=y --aten=y --flash-attn=/usr/local/lib/python3.10/dist-packages -cv

  # 编译 Python/C++ 封装：
  xmake build _infinicore
  xmake install _infinicore
  ```

##### 试验功能 -- 编译marlin相关算子

  ```shell

  # 需要从github上克隆tvm_ffi仓库，克隆命令参考
  ## tvm-ffi commit: 35c99d0ac4cb784862115d0089f60c603acec8f9
      git clone https://github.com/apache/tvm-ffi.git --recursive

  # 设置TVM_ROOT
      export TVM_ROOT=<path-to>/tvm-ffi #用来搜索tvm相关头文件
  # 注意，编译gptq_marlin_gemm算子的时候除了指定TVM_ROOT以外，还需要指定cuda_arch
  ```

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
