from .structs import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    infiniopOperatorDescriptor_t,
)

from ctypes import c_int32, c_void_p, c_size_t, POINTER, c_float, c_bool


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


@OpRegister.operator
def conv_(lib):
    pass


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
def rope_v2_(lib):
    lib.infiniopCreateRoPEv2Descriptor.restype = c_int32
    lib.infiniopCreateRoPEv2Descriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRoPEv2WorkspaceSize.restype = c_int32
    lib.infiniopGetRoPEv2WorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRoPEv2.restype = c_int32
    lib.infiniopRoPEv2.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRoPEv2Descriptor.restype = c_int32
    lib.infiniopDestroyRoPEv2Descriptor.argtypes = [
        infiniopOperatorDescriptor_t,
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
def sigmoid_(lib):
    lib.infiniopCreateSigmoidDescriptor.restype = c_int32
    lib.infiniopCreateSigmoidDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetSigmoidWorkspaceSize.restype = c_int32
    lib.infiniopGetSigmoidWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopSigmoid.restype = c_int32
    lib.infiniopSigmoid.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroySigmoidDescriptor.restype = c_int32
    lib.infiniopDestroySigmoidDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def topksoftmax_(lib):
    lib.infiniopCreateTopksoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateTopksoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetTopksoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetTopksoftmaxWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopTopksoftmax.restype = c_int32
    lib.infiniopTopksoftmax.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_size_t,
        c_bool , 
        c_void_p,
    ]
    lib.infiniopDestroyTopksoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroyTopksoftmaxDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def topkrouter_(lib):
    lib.infiniopCreateTopkrouterDescriptor.restype = c_int32
    lib.infiniopCreateTopkrouterDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t
    ]

    lib.infiniopGetTopkrouterWorkspaceSize.restype = c_int32
    lib.infiniopGetTopkrouterWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopTopkrouter.restype = c_int32
    lib.infiniopTopkrouter.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_float,
        c_size_t,
        c_void_p,
    ]
    lib.infiniopDestroyTopkrouterDescriptor.restype = c_int32
    lib.infiniopDestroyTopkrouterDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def dequantize_(lib):
    lib.infiniopCreateDequantizeDescriptor.restype = c_int32
    lib.infiniopCreateDequantizeDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetDequantizeWorkspaceSize.restype = c_int32
    lib.infiniopGetDequantizeWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopDequantize.restype = c_int32
    lib.infiniopDequantize.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_size_t,
        c_size_t,
        c_size_t,
        c_void_p,
    ]
    lib.infiniopDestroyDequantizeDescriptor.restype = c_int32
    lib.infiniopDestroyDequantizeDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def exp_(lib):
    lib.infiniopCreateExpDescriptor.restype = c_int32
    lib.infiniopCreateExpDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopExp.restype = c_int32
    lib.infiniopExp.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyExpDescriptor.restype = c_int32
    lib.infiniopDestroyExpDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def sin_(lib):
    lib.infiniopCreateSinDescriptor.restype = c_int32
    lib.infiniopCreateSinDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopSin.restype = c_int32
    lib.infiniopSin.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroySinDescriptor.restype = c_int32
    lib.infiniopDestroySinDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def cos_(lib):
    lib.infiniopCreateCosDescriptor.restype = c_int32
    lib.infiniopCreateCosDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopCos.restype = c_int32
    lib.infiniopCos.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyCosDescriptor.restype = c_int32
    lib.infiniopDestroyCosDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def tanh_(lib):
    lib.infiniopCreateTanhDescriptor.restype = c_int32
    lib.infiniopCreateTanhDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopTanh.restype = c_int32
    lib.infiniopTanh.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyTanhDescriptor.restype = c_int32
    lib.infiniopDestroyTanhDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def sigmoid_backward_(lib):
    lib.infiniopCreateSigmoidBackwardDescriptor.restype = c_int32
    lib.infiniopCreateSigmoidBackwardDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSigmoidBackwardWorkspaceSize.restype = c_int32
    lib.infiniopGetSigmoidBackwardWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSigmoidBackward.restype = c_int32
    lib.infiniopSigmoidBackward.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySigmoidBackwardDescriptor.restype = c_int32
    lib.infiniopDestroySigmoidBackwardDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def hardswish_(lib):
    lib.infiniopCreateHardSwishDescriptor.restype = c_int32
    lib.infiniopCreateHardSwishDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopHardSwish.restype = c_int32
    lib.infiniopHardSwish.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyHardSwishDescriptor.restype = c_int32
    lib.infiniopDestroyHardSwishDescriptor.argtypes = [infiniopOperatorDescriptor_t]


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


@OpRegister.operator
def leaky_relu_(lib):
    lib.infiniopCreateLeakyReLUDescriptor.restype = c_int32
    lib.infiniopCreateLeakyReLUDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopGetLeakyReLUWorkspaceSize.restype = c_int32
    lib.infiniopGetLeakyReLUWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopLeakyReLU.restype = c_int32
    lib.infiniopLeakyReLU.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyLeakyReLUDescriptor.restype = c_int32
    lib.infiniopDestroyLeakyReLUDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def div_(lib):
    lib.infiniopCreateDivDescriptor.restype = c_int32
    lib.infiniopCreateDivDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetDivWorkspaceSize.restype = c_int32
    lib.infiniopGetDivWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopDiv.restype = c_int32
    lib.infiniopDiv.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyDivDescriptor.restype = c_int32
    lib.infiniopDestroyDivDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def silu_(lib):
    lib.infiniopCreateSiluDescriptor.restype = c_int32
    lib.infiniopCreateSiluDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopSilu.restype = c_int32
    lib.infiniopSilu.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroySiluDescriptor.restype = c_int32
    lib.infiniopDestroySiluDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def and_(lib):
    lib.infiniopCreateAndDescriptor.restype = c_int32
    lib.infiniopCreateAndDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetAndWorkspaceSize.restype = c_int32
    lib.infiniopGetAndWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAnd.restype = c_int32
    lib.infiniopAnd.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAndDescriptor.restype = c_int32
    lib.infiniopDestroyAndDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def or_(lib):
    lib.infiniopCreateOrDescriptor.restype = c_int32
    lib.infiniopCreateOrDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetOrWorkspaceSize.restype = c_int32
    lib.infiniopGetOrWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopOr.restype = c_int32
    lib.infiniopOr.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyOrDescriptor.restype = c_int32
    lib.infiniopDestroyOrDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def equal_(lib):
    lib.infiniopCreateEqualDescriptor.restype = c_int32
    lib.infiniopCreateEqualDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetEqualWorkspaceSize.restype = c_int32
    lib.infiniopGetEqualWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopEqual.restype = c_int32
    lib.infiniopEqual.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyEqualDescriptor.restype = c_int32
    lib.infiniopDestroyEqualDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def relu_backward_(lib):
    lib.infiniopCreateReluBackwardDescriptor.restype = c_int32
    lib.infiniopCreateReluBackwardDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetReluBackwardWorkspaceSize.restype = c_int32
    lib.infiniopGetReluBackwardWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopReluBackward.restype = c_int32
    lib.infiniopReluBackward.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyReluBackwardDescriptor.restype = c_int32
    lib.infiniopDestroyReluBackwardDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def gelu_backward_(lib):
    lib.infiniopCreateGeluBackwardDescriptor.restype = c_int32
    lib.infiniopCreateGeluBackwardDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetGeluBackwardWorkspaceSize.restype = c_int32
    lib.infiniopGetGeluBackwardWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopGeluBackward.restype = c_int32
    lib.infiniopGeluBackward.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyGeluBackwardDescriptor.restype = c_int32
    lib.infiniopDestroyGeluBackwardDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def gelu_(lib):
    lib.infiniopCreateGeluDescriptor.restype = c_int32
    lib.infiniopCreateGeluDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGelu.restype = c_int32
    lib.infiniopGelu.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyGeluDescriptor.restype = c_int32
    lib.infiniopDestroyGeluDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def cross_entropy_loss_backward_(lib):
    lib.infiniopCreateCrossEntropyLossBackwardDescriptor.restype = c_int32
    lib.infiniopCreateCrossEntropyLossBackwardDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetCrossEntropyLossBackwardWorkspaceSize.restype = c_int32
    lib.infiniopGetCrossEntropyLossBackwardWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopCrossEntropyLossBackward.restype = c_int32
    lib.infiniopCrossEntropyLossBackward.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyCrossEntropyLossBackwardDescriptor.restype = c_int32
    lib.infiniopDestroyCrossEntropyLossBackwardDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]

@OpRegister.operator
def index_copy_inplace_(lib):
    lib.infiniopCreateIndexCopyInplaceDescriptor.restype = c_int32
    lib.infiniopCreateIndexCopyInplaceDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_size_t,
    ]
    lib.infiniopGetIndexCopyInplaceWorkspaceSize.restype = c_int32
    lib.infiniopGetIndexCopyInplaceWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]    
    lib.infiniopIndexCopyInplace.restype = c_int32
    lib.infiniopIndexCopyInplace.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyIndexCopyInplaceDescriptor.restype = c_int32
    lib.infiniopDestroyIndexCopyInplaceDescriptor.argtypes = [infiniopOperatorDescriptor_t]

@OpRegister.operator
def gather_(lib):
    lib.infiniopCreateGatherDescriptor.restype = c_int32
    lib.infiniopCreateGatherDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_size_t,
    ]
    lib.infiniopGetGatherWorkspaceSize.restype = c_int32
    lib.infiniopGetGatherWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]    
    lib.infiniopGather.restype = c_int32
    lib.infiniopGather.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyGatherDescriptor.restype = c_int32
    lib.infiniopDestroyGatherDescriptor.argtypes = [infiniopOperatorDescriptor_t]

@OpRegister.operator
def scatter_(lib):
    lib.infiniopCreateScatterDescriptor.restype = c_int32
    lib.infiniopCreateScatterDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_size_t,
    ]
    lib.infiniopGetScatterWorkspaceSize.restype = c_int32
    lib.infiniopGetScatterWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]    
    lib.infiniopScatter.restype = c_int32
    lib.infiniopScatter.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyScatterDescriptor.restype = c_int32
    lib.infiniopDestroyScatterDescriptor.argtypes = [infiniopOperatorDescriptor_t]

@OpRegister.operator
def tril_(lib):
    lib.infiniopCreateTrilDescriptor.restype = c_int32
    lib.infiniopCreateTrilDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]
    lib.infiniopGetTrilWorkspaceSize.restype = c_int32
    lib.infiniopGetTrilWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]    
    lib.infiniopTril.restype = c_int32
    lib.infiniopTril.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyTrilDescriptor.restype = c_int32
    lib.infiniopDestroyTrilDescriptor.argtypes = [infiniopOperatorDescriptor_t]

@OpRegister.operator
def triu_(lib):
    lib.infiniopCreateTriuDescriptor.restype = c_int32
    lib.infiniopCreateTriuDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]
    lib.infiniopGetTriuWorkspaceSize.restype = c_int32
    lib.infiniopGetTriuWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]    
    lib.infiniopTriu.restype = c_int32
    lib.infiniopTriu.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyTriuDescriptor.restype = c_int32
    lib.infiniopDestroyTriuDescriptor.argtypes = [infiniopOperatorDescriptor_t]

@OpRegister.operator
def linear_(lib):
    lib.infiniopCreateLinearDescriptor.restype = c_int32
    lib.infiniopCreateLinearDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetLinearWorkspaceSize.restype = c_int32
    lib.infiniopGetLinearWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]    
    lib.infiniopLinear.restype = c_int32
    lib.infiniopLinear.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyLinearDescriptor.restype = c_int32
    lib.infiniopDestroyLinearDescriptor.argtypes = [infiniopOperatorDescriptor_t]

@OpRegister.operator
def linear_backward_(lib):
    lib.infiniopCreateLinearBackwardDescriptor.restype = c_int32
    lib.infiniopCreateLinearBackwardDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetLinearBackwardWorkspaceSize.restype = c_int32
    lib.infiniopGetLinearBackwardWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]    
    lib.infiniopLinearBackward.restype = c_int32
    lib.infiniopLinearBackward.argtypes = [
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
    lib.infiniopDestroyLinearBackwardDescriptor.restype = c_int32
    lib.infiniopDestroyLinearBackwardDescriptor.argtypes = [infiniopOperatorDescriptor_t]

