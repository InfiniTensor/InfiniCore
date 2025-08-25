from .structs import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    infiniopOperatorDescriptor_t,
)

from ctypes import c_int32, c_void_p, c_size_t, POINTER, c_float


class OpRegister:
    registry = []

    @classmethod
    def operator(cls, op):
        cls.registry.append(op)
        return op

    @classmethod
    def register_lib(cls, lib):
        for op in cls.registry:
            op(lib)


@OpRegister.operator
def add_(lib):
    lib.infiniopCreateAddDescriptor.restype = c_int32
    lib.infiniopCreateAddDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetAddWorkspaceSize.restype = c_int32
    lib.infiniopGetAddWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAdd.restype = c_int32
    lib.infiniopAdd.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAddDescriptor.restype = c_int32
    lib.infiniopDestroyAddDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def attention_(lib):
    lib.infiniopCreateAttentionDescriptor.restype = c_int32
    lib.infiniopCreateAttentionDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_size_t,
    ]

    lib.infiniopGetAttentionWorkspaceSize.restype = c_int32
    lib.infiniopGetAttentionWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAttention.restype = c_int32
    lib.infiniopAttention.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAttentionDescriptor.restype = c_int32
    lib.infiniopDestroyAttentionDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def cast_(lib):
    lib.infiniopCreateCastDescriptor.restype = c_int32
    lib.infiniopCreateCastDescriptor.argtypes = [
        infiniopHandle_t,  # 句柄
        POINTER(infiniopOperatorDescriptor_t),  # 输出：算子描述符指针
        infiniopTensorDescriptor_t,  # 输出张量描述符
        infiniopTensorDescriptor_t   # 输入张量描述符
    ]

    lib.infiniopGetCastWorkspaceSize.restype = c_int32
    lib.infiniopGetCastWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        POINTER(c_size_t)  # 输出：工作空间大小指针
    ]

    lib.infiniopCast.restype = c_int32
    lib.infiniopCast.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        c_void_p,  # 工作空间地址
        c_size_t,  # 工作空间大小
        c_void_p,  # 输出数据地址
        c_void_p,  # 输入数据地址
        c_void_p   # 额外参数（通常为nullptr）
    ]

    lib.infiniopDestroyCastDescriptor.restype = c_int32
    lib.infiniopDestroyCastDescriptor.argtypes = [
        infiniopOperatorDescriptor_t  # 算子描述符
    ]

@OpRegister.operator
def causal_softmax_(lib):
    lib.infiniopCreateCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateCausalSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetCausalSoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetCausalSoftmaxWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopCausalSoftmax.restype = c_int32
    lib.infiniopCausalSoftmax.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyCausalSoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroyCausalSoftmaxDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def clip_(lib):
    lib.infiniopCreateClipDescriptor.restype = c_int32
    lib.infiniopCreateClipDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetClipWorkspaceSize.restype = c_int32
    lib.infiniopGetClipWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopClip.restype = c_int32
    lib.infiniopClip.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyClipDescriptor.restype = c_int32
    lib.infiniopDestroyClipDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


# @OpRegister.operator
# def conv_(lib):
#     pass


@OpRegister.operator
def cos_(lib):
    lib.infiniopCreateCosDescriptor.restype = c_int32
    lib.infiniopCreateCosDescriptor.argtypes = [
        infiniopHandle_t,  # 句柄
        POINTER(infiniopOperatorDescriptor_t),  # 输出：算子描述符指针
        infiniopTensorDescriptor_t,  # 输出张量描述符
        infiniopTensorDescriptor_t   # 输入张量描述符（cos为单输入）
    ]

    # 2. 获取cos算子工作空间大小的函数
    lib.infiniopGetCosWorkspaceSize.restype = c_int32
    lib.infiniopGetCosWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        POINTER(c_size_t)  # 输出：工作空间大小指针
    ]

    # 3. 执行cos算子的函数
    lib.infiniopCos.restype = c_int32
    lib.infiniopCos.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        c_void_p,  # 工作空间地址
        c_size_t,  # 工作空间大小
        c_void_p,  # 输出数据地址
        c_void_p,  # 输入数据地址（cos为单输入）
        c_void_p   # 额外参数（通常为nullptr）
    ]

    # 4. 销毁cos算子描述符的函数
    lib.infiniopDestroyCosDescriptor.restype = c_int32
    lib.infiniopDestroyCosDescriptor.argtypes = [
        infiniopOperatorDescriptor_t  # 算子描述符
    ]


@OpRegister.operator
def exp_(lib):
    # 1. 创建exp算子描述符的函数
    lib.infiniopCreateExpDescriptor.restype = c_int32
    lib.infiniopCreateExpDescriptor.argtypes = [
        infiniopHandle_t,  # 句柄
        POINTER(infiniopOperatorDescriptor_t),  # 输出：算子描述符指针
        infiniopTensorDescriptor_t,  # 输出张量描述符
        infiniopTensorDescriptor_t   # 输入张量描述符（exp为单输入）
    ]

    # 2. 获取exp算子工作空间大小的函数
    lib.infiniopGetExpWorkspaceSize.restype = c_int32
    lib.infiniopGetExpWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        POINTER(c_size_t)  # 输出：工作空间大小指针
    ]

    # 3. 执行exp算子的函数
    lib.infiniopExp.restype = c_int32
    lib.infiniopExp.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        c_void_p,  # 工作空间地址
        c_size_t,  # 工作空间大小
        c_void_p,  # 输出数据地址
        c_void_p,  # 输入数据地址（exp为单输入）
        c_void_p   # 额外参数（通常为nullptr）
    ]

    # 4. 销毁exp算子描述符的函数
    lib.infiniopDestroyExpDescriptor.restype = c_int32
    lib.infiniopDestroyExpDescriptor.argtypes = [
        infiniopOperatorDescriptor_t  # 算子描述符
    ]

@OpRegister.operator
def gemm_(lib):
    lib.infiniopCreateGemmDescriptor.restype = c_int32
    lib.infiniopCreateGemmDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetGemmWorkspaceSize.restype = c_int32
    lib.infiniopGetGemmWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopGemm.restype = c_int32
    lib.infiniopGemm.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_float,
        c_float,
        c_void_p,
    ]

    lib.infiniopDestroyGemmDescriptor.restype = c_int32
    lib.infiniopDestroyGemmDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]



@OpRegister.operator
def hard_swish_(lib):
    # 1. 创建hard_swish算子描述符的函数
    lib.infiniopCreateHardSwishDescriptor.restype = c_int32
    lib.infiniopCreateHardSwishDescriptor.argtypes = [
        infiniopHandle_t,  # 句柄
        POINTER(infiniopOperatorDescriptor_t),  # 输出：算子描述符指针
        infiniopTensorDescriptor_t,  # 输出张量描述符
        infiniopTensorDescriptor_t   # 输入张量描述符（hard_swish为单输入）
    ]

    # 2. 获取hard_swish算子工作空间大小的函数
    lib.infiniopGetHardSwishWorkspaceSize.restype = c_int32
    lib.infiniopGetHardSwishWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        POINTER(c_size_t)  # 输出：工作空间大小指针
    ]

    # 3. 执行hard_swish算子的函数
    lib.infiniopHardSwish.restype = c_int32
    lib.infiniopHardSwish.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        c_void_p,  # 工作空间地址
        c_size_t,  # 工作空间大小
        c_void_p,  # 输出数据地址
        c_void_p,  # 输入数据地址（hard_swish为单输入）
        c_void_p   # 额外参数（通常为nullptr）
    ]

    # 4. 销毁hard_swish算子描述符的函数
    lib.infiniopDestroyHardSwishDescriptor.restype = c_int32
    lib.infiniopDestroyHardSwishDescriptor.argtypes = [
        infiniopOperatorDescriptor_t  # 算子描述符
    ]

@OpRegister.operator
def leaky_relu_(lib):
    # 1. 创建 LeakyReLU 算子描述符的函数
    lib.infiniopCreateLeakyReluDescriptor.restype = c_int32
    lib.infiniopCreateLeakyReluDescriptor.argtypes = [
        infiniopHandle_t,  # 句柄
        POINTER(infiniopOperatorDescriptor_t),  # 输出：算子描述符指针
        infiniopTensorDescriptor_t,  # 输出张量描述符
        infiniopTensorDescriptor_t,  # 输入张量描述符
    ]
    
    # 2. 获取 LeakyReLU 算子工作空间大小的函数
    lib.infiniopGetLeakyReluWorkspaceSize.restype = c_int32
    lib.infiniopGetLeakyReluWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        POINTER(c_size_t)  # 输出：工作空间大小指针
    ]
    
    # 3. 执行 LeakyReLU 算子的函数
    lib.infiniopLeakyRelu.restype = c_int32
    lib.infiniopLeakyRelu.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        c_void_p,  # 工作空间地址
        c_size_t,  # 工作空间大小
        c_void_p,  # 输出数据地址
        c_void_p,  # 输入数据地址
        c_float,   # negative_slope 参数
        c_void_p   # 额外参数（通常为 nullptr）
    ]
    
    # 4. 销毁 LeakyReLU 算子描述符的函数
    lib.infiniopDestroyLeakyReluDescriptor.restype = c_int32
    lib.infiniopDestroyLeakyReluDescriptor.argtypes = [
        infiniopOperatorDescriptor_t  # 算子描述符
    ]
    

@OpRegister.operator
def mul_(lib):
    lib.infiniopCreateMulDescriptor.restype = c_int32
    lib.infiniopCreateMulDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetMulWorkspaceSize.restype = c_int32
    lib.infiniopGetMulWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopMul.restype = c_int32
    lib.infiniopMul.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyMulDescriptor.restype = c_int32
    lib.infiniopDestroyMulDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def random_sample_(lib):
    lib.infiniopCreateRandomSampleDescriptor.restype = c_int32
    lib.infiniopCreateRandomSampleDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRandomSampleWorkspaceSize.restype = c_int32
    lib.infiniopGetRandomSampleWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRandomSample.restype = c_int32
    lib.infiniopRandomSample.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_size_t,
        c_void_p,
        c_float,
        c_float,
        c_int32,
        c_float,
        c_void_p,
    ]

    lib.infiniopDestroyRandomSampleDescriptor.restype = c_int32
    lib.infiniopDestroyRandomSampleDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def rearrange_(lib):
    lib.infiniopCreateRearrangeDescriptor.restype = c_int32
    lib.infiniopCreateRearrangeDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopRearrange.restype = c_int32
    lib.infiniopRearrange.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRearrangeDescriptor.restype = c_int32
    lib.infiniopDestroyRearrangeDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def relu_(lib):
    lib.infiniopCreateReluDescriptor.restype = c_int32
    lib.infiniopCreateReluDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopRelu.restype = c_int32
    lib.infiniopRelu.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReluDescriptor.restype = c_int32
    lib.infiniopDestroyReluDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def rms_norm_(lib):
    lib.infiniopCreateRMSNormDescriptor.restype = c_int32
    lib.infiniopCreateRMSNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopGetRMSNormWorkspaceSize.restype = c_int32
    lib.infiniopGetRMSNormWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRMSNorm.restype = c_int32
    lib.infiniopRMSNorm.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRMSNormDescriptor.restype = c_int32
    lib.infiniopDestroyRMSNormDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def rope_(lib):
    lib.infiniopCreateRoPEDescriptor.restype = c_int32
    lib.infiniopCreateRoPEDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRoPEWorkspaceSize.restype = c_int32
    lib.infiniopGetRoPEWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRoPE.restype = c_int32
    lib.infiniopRoPE.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRoPEDescriptor.restype = c_int32
    lib.infiniopDestroyRoPEDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]



@OpRegister.operator
def sigmoid_backward_(lib):
    lib.infiniopCreateSigmoidBackwardDescriptor.restype = c_int32
    lib.infiniopCreateSigmoidBackwardDescriptor.argtypes = [
        infiniopHandle_t,  # 句柄
        POINTER(infiniopOperatorDescriptor_t),  # 输出：算子描述符指针
        infiniopTensorDescriptor_t,  
        infiniopTensorDescriptor_t,  
        infiniopTensorDescriptor_t   
    ]

    lib.infiniopGetSigmoidBackwardWorkspaceSize.restype = c_int32
    lib.infiniopGetSigmoidBackwardWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        POINTER(c_size_t)  # 输出：工作空间大小指针
    ]

    lib.infiniopSigmoidBackward.restype = c_int32
    lib.infiniopSigmoidBackward.argtypes = [
        infiniopOperatorDescriptor_t,  # 算子描述符
        c_void_p,  # 工作空间地址
        c_size_t,  # 工作空间大小
        c_void_p,  # 输出数据地址
        c_void_p,  # 输入数据地址
        c_void_p,  # 输入数据地址
        c_void_p   # 额外参数（通常为nullptr）
    ]

    lib.infiniopDestroySigmoidBackwardDescriptor.restype = c_int32
    lib.infiniopDestroySigmoidBackwardDescriptor.argtypes = [
        infiniopOperatorDescriptor_t  # 算子描述符
    ]



@OpRegister.operator
def sub_(lib):
    lib.infiniopCreateSubDescriptor.restype = c_int32
    lib.infiniopCreateSubDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSubWorkspaceSize.restype = c_int32
    lib.infiniopGetSubWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSub.restype = c_int32
    lib.infiniopSub.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySubDescriptor.restype = c_int32
    lib.infiniopDestroySubDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def swiglu_(lib):
    lib.infiniopCreateSwiGLUDescriptor.restype = c_int32
    lib.infiniopCreateSwiGLUDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSwiGLUWorkspaceSize.restype = c_int32
    lib.infiniopGetSwiGLUWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSwiGLU.restype = c_int32
    lib.infiniopSwiGLU.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySwiGLUDescriptor.restype = c_int32
    lib.infiniopDestroySwiGLUDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]

@OpRegister.operator
def conv_(lib):
    lib.infiniopCreateConvDescriptor.restype = c_int32
    lib.infiniopCreateConvDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_size_t,
    ]
    lib.infiniopGetConvWorkspaceSize.restype = c_int32
    lib.infiniopGetConvWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopConv.restype = c_int32
    lib.infiniopConv.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyConvDescriptor.restype = c_int32
    lib.infiniopDestroyConvDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]

@OpRegister.operator
def sin_(lib):
    lib.infiniopCreateSinDescriptor.restype = c_int32
    lib.infiniopCreateSinDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # output descriptor
        infiniopTensorDescriptor_t,  # input descriptor
    ]

    lib.infiniopGetSinWorkspaceSize.restype = c_int32
    lib.infiniopGetSinWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSin.restype = c_int32
    lib.infiniopSin.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,  # output data
        c_void_p,  # input data
        c_void_p,  # stream or reserved
    ]

    lib.infiniopDestroySinDescriptor.restype = c_int32
    lib.infiniopDestroySinDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]

@OpRegister.operator
def tanh_(lib):
    lib.infiniopCreateTanhDescriptor.restype = c_int32
    lib.infiniopCreateTanhDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # output descriptor
        infiniopTensorDescriptor_t,  # input descriptor
    ]

    lib.infiniopGetTanhWorkspaceSize.restype = c_int32
    lib.infiniopGetTanhWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopTanh.restype = c_int32
    lib.infiniopTanh.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,  # output data
        c_void_p,  # input data
        c_void_p,  # stream or reserved
    ]

    lib.infiniopDestroyTanhDescriptor.restype = c_int32
    lib.infiniopDestroyTanhDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]
@OpRegister.operator
def where_(lib):
    lib.infiniopCreateWhereDescriptor.restype = c_int32
    lib.infiniopCreateWhereDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetWhereWorkspaceSize.restype = c_int32
    lib.infiniopGetWhereWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopWhere.restype = c_int32
    lib.infiniopWhere.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyWhereDescriptor.restype = c_int32
    lib.infiniopDestroyWhereDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]