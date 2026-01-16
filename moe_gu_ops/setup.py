import os
import sys

try:
    import torch
    from torch.utils import cpp_extension
    def no_op_check(compiler_name, compiler_version):
        return True
    cpp_extension._check_cuda_version = no_op_check
    os.environ["TORCH_DONT_CHECK_CUDA_VERSION_COMPATIBILITY"] = "1"
except ImportError:
    pass
# =================================================================================

if hasattr(torch, '_C'):
    torch._C._GLIBCXX_USE_CXX11_ABI = True

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pybind11

# 获取当前目录 (即 .../InfiniCore/gu_moe_ops)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# -------------------------------------------------------------------------
# 1. 路径配置 (动态推导)
# -------------------------------------------------------------------------
# 优先使用环境变量，如果没有，则根据目录结构自动推导
# 假设结构是：
# InfiniCore/
#   ├── gu_moe_ops/  <-- 我们在这里
#   ├── include/
#   └── build/
INFINI_SRC_ROOT = os.getenv("INFINICORE_ROOT")
if not INFINI_SRC_ROOT:
    # 向上找一级，就是 InfiniCore 根目录
    INFINI_SRC_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../"))

INFINI_LIB_DIR = os.getenv("INFINICORE_BUILD_DIR")
if not INFINI_LIB_DIR:
    # 默认构建路径
    INFINI_LIB_DIR = os.path.join(INFINI_SRC_ROOT, 'build/linux/x86_64/release')

print(f"[Info] InfiniCore Root: {INFINI_SRC_ROOT}")
print(f"[Info] InfiniCore Lib Dir: {INFINI_LIB_DIR}")

# 检查一下库目录是否存在，防止后面报错看不懂
if not os.path.exists(INFINI_LIB_DIR):
    print(f"[Warning] Library directory not found: {INFINI_LIB_DIR}")
    print("Please check if InfiniCore is built or set INFINICORE_BUILD_DIR env var.")

# -------------------------------------------------------------------------
# 2. 库列表
# -------------------------------------------------------------------------
libs = [
    os.path.join(INFINI_LIB_DIR, 'libinfini-utils.a'), 
    os.path.join(INFINI_LIB_DIR, 'libinfiniop-nvidia.a'),
    os.path.join(INFINI_LIB_DIR, 'libinfiniccl-nvidia.a'),
    os.path.join(INFINI_LIB_DIR, 'libinfinirt-nvidia.a') 
]

setup(
    name='gu_moe_ops',
    version='0.1.0',
    ext_modules=[
        CUDAExtension(
            name='gu_moe_ops',
            sources=[
                'pybind_gumoe.cc',          
                'src/gumoe.cpp',            
                'src/gu_mul.cc',            
                'src/gu_topk_softmax.cc',
                'src/nvidia_kernels/gu_reduce.cu', 
                'src/nvidia_kernels/gu_sort.cu', 
            ],
            include_dirs=[
                pybind11.get_include(),
                os.path.join(INFINI_SRC_ROOT, 'include'), # 动态引用父目录的 include
                os.path.join(CURRENT_DIR, 'src'),
                "/usr/local/cuda/include" 
            ],
            extra_objects=libs,
            
            extra_link_args=[
                '-Wl,--allow-shlib-undefined' 
            ],
            
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17', '-D_GLIBCXX_USE_CXX11_ABI=1'],
                'nvcc': [
                    '-O3', '--use_fast_math', 
                    # 注意：移除了硬编码的 -gencode 参数
                    # 编译时会自动读取环境变量 TORCH_CUDA_ARCH_LIST
                ]
            }
        )
    ],
    cmdclass={ 'build_ext': BuildExtension }
)