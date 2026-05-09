from .structs import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    infiniopOperatorDescriptor_t,
)

from ctypes import (
    c_int32,
    c_void_p,
    c_size_t,
    POINTER,
    c_float,
    c_double,
    c_int64,
    c_bool,
    c_int64,
)


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
def atanh_(lib):
    lib.infiniopCreateAtanhDescriptor.restype = c_int32
    lib.infiniopCreateAtanhDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetAtanhWorkspaceSize.restype = c_int32
    lib.infiniopGetAtanhWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAtanh.restype = c_int32
    lib.infiniopAtanh.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # y_data
        c_void_p,  # a_data
        c_void_p,  # stream
    ]

    lib.infiniopDestroyAtanhDescriptor.restype = c_int32
    lib.infiniopDestroyAtanhDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


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
def addcmul_(lib):
    lib.infiniopCreateAddcmulDescriptor.restype = c_int32
    lib.infiniopCreateAddcmulDescriptor.argtypes = [
        infiniopHandle_t,  # handle
        POINTER(infiniopOperatorDescriptor_t),  # desc_ptr
        infiniopTensorDescriptor_t,  # out_desc
        infiniopTensorDescriptor_t,  # input_desc
        infiniopTensorDescriptor_t,  # t1_desc
        infiniopTensorDescriptor_t,  # t2_desc
        c_float,  # value (标量系数)
    ]

    lib.infiniopGetAddcmulWorkspaceSize.restype = c_int32
    lib.infiniopGetAddcmulWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        POINTER(c_size_t),  # size_ptr
    ]

    lib.infiniopAddcmul.restype = c_int32
    lib.infiniopAddcmul.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # out_ptr
        c_void_p,  # input_ptr
        c_void_p,  # t1_ptr
        c_void_p,  # t2_ptr
        c_void_p,  # stream
    ]

    lib.infiniopDestroyAddcmulDescriptor.restype = c_int32
    lib.infiniopDestroyAddcmulDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
    ]


@OpRegister.operator
def cdist_(lib):
    # 1. 创建描述符接口
    # 接口通常接收 handle, 输出 desc, 两个输入 desc, 以及范数 p
    lib.infiniopCreateCdistDescriptor.restype = c_int32
    lib.infiniopCreateCdistDescriptor.argtypes = [
        infiniopHandle_t,  # handle
        POINTER(infiniopOperatorDescriptor_t),  # desc_ptr
        infiniopTensorDescriptor_t,  # y_desc (输出)
        infiniopTensorDescriptor_t,  # x1_desc
        infiniopTensorDescriptor_t,  # x2_desc
        c_double,  # p (范数阶数)
    ]

    # 2. 获取 Workspace 大小接口
    lib.infiniopGetCdistWorkspaceSize.restype = c_int32
    lib.infiniopGetCdistWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        POINTER(c_size_t),  # size_ptr
    ]

    # 3. 执行算子接口
    lib.infiniopCdist.restype = c_int32
    lib.infiniopCdist.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # y_ptr
        c_void_p,  # x1_ptr
        c_void_p,  # x2_ptr
        c_void_p,  # stream
    ]

    # 4. 销毁描述符接口
    lib.infiniopDestroyCdistDescriptor.restype = c_int32
    lib.infiniopDestroyCdistDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
    ]


@OpRegister.operator
def binary_cross_entropy_with_logits_(lib):
    # 1. 创建描述符 (Descriptor Creation)
    lib.infiniopCreateBCEWithLogitsDescriptor.restype = c_int32
    lib.infiniopCreateBCEWithLogitsDescriptor.argtypes = [
        infiniopHandle_t,  # handle
        POINTER(infiniopOperatorDescriptor_t),  # desc_ptr
        infiniopTensorDescriptor_t,  # out_desc
        infiniopTensorDescriptor_t,  # input_desc (logits)
        infiniopTensorDescriptor_t,  # target_desc
        infiniopTensorDescriptor_t,  # weight_desc (可选，不可用则传 NULL)
        infiniopTensorDescriptor_t,  # pos_weight_desc (可选，不可用则传 NULL)
        c_int32,  # reduction (0:none, 1:mean, 2:sum)
    ]

    # 2. 获取工作空间大小 (Workspace Size)
    lib.infiniopGetBCEWithLogitsWorkspaceSize.restype = c_int32
    lib.infiniopGetBCEWithLogitsWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        POINTER(c_size_t),  # size_ptr
    ]

    # 3. 执行算子 (Execution)
    lib.infiniopBCEWithLogits.restype = c_int32
    lib.infiniopBCEWithLogits.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # out_ptr
        c_void_p,  # input_ptr (logits)
        c_void_p,  # target_ptr
        c_void_p,  # weight_ptr (可选)
        c_void_p,  # pos_weight_ptr (可选)
        c_void_p,  # stream
    ]

    # 4. 销毁描述符 (Destruction)
    lib.infiniopDestroyBCEWithLogitsDescriptor.restype = c_int32
    lib.infiniopDestroyBCEWithLogitsDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,  # descriptor
    ]


@OpRegister.operator
def reciprocal_(lib):
    lib.infiniopCreateReciprocalDescriptor.restype = c_int32
    lib.infiniopCreateReciprocalDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # Output descriptor
        infiniopTensorDescriptor_t,  # Input descriptor
    ]

    # 获取工作空间大小接口
    lib.infiniopGetReciprocalWorkspaceSize.restype = c_int32
    lib.infiniopGetReciprocalWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    # 最后的 c_void_p 通常对应 stream 或其他异步句柄，保持一致即可
    lib.infiniopReciprocal.restype = c_int32
    lib.infiniopReciprocal.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,  # Workspace pointer
        c_size_t,  # Workspace size
        c_void_p,  # Output data pointer
        c_void_p,  # Input data pointer
        c_void_p,  # Stream pointer (optional)
    ]

    # 销毁描述符接口
    lib.infiniopDestroyReciprocalDescriptor.restype = c_int32
    lib.infiniopDestroyReciprocalDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def equal_(lib):
    # =========================================================
    # 1. 注册 Create 函数
    # C函数签名: (handle, &desc, output_desc, input_a_desc, input_b_desc)
    # =========================================================
    lib.infiniopCreateEqualDescriptor.restype = c_int32
    lib.infiniopCreateEqualDescriptor.argtypes = [
        infiniopHandle_t,  # handle
        POINTER(infiniopOperatorDescriptor_t),  # desc_ptr (输出)
        infiniopTensorDescriptor_t,  # output (c)
        infiniopTensorDescriptor_t,  # input_a
        infiniopTensorDescriptor_t,  # input_b
    ]

    # =========================================================
    # 2. 注册 GetWorkspaceSize 函数
    # C函数签名: (desc, &size)
    # =========================================================
    lib.infiniopGetEqualWorkspaceSize.restype = c_int32
    lib.infiniopGetEqualWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    # =========================================================
    # 3. 注册 Execute (计算) 函数
    # C函数签名: (desc, workspace, size, output_data, input_a_data, input_b_data, stream)
    # =========================================================
    lib.infiniopEqual.restype = c_int32
    lib.infiniopEqual.argtypes = [
        infiniopOperatorDescriptor_t,  # desc
        c_void_p,  # workspace ptr
        c_size_t,  # workspace size
        c_void_p,  # output data ptr
        c_void_p,  # input a data ptr
        c_void_p,  # input b data ptr
        c_void_p,  # stream
    ]

    # =========================================================
    # 4. 注册 Destroy 函数
    # C函数签名: (desc)
    # =========================================================
    lib.infiniopDestroyEqualDescriptor.restype = c_int32
    lib.infiniopDestroyEqualDescriptor.argtypes = [
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
def cross_entropy_(lib):
    lib.infiniopCreateCrossEntropyDescriptor.restype = c_int32
    lib.infiniopCreateCrossEntropyDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetCrossEntropyWorkspaceSize.restype = c_int32
    lib.infiniopGetCrossEntropyWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopCrossEntropy.restype = c_int32
    lib.infiniopCrossEntropy.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyCrossEntropyDescriptor.restype = c_int32
    lib.infiniopDestroyCrossEntropyDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def logsoftmax_(lib):
    lib.infiniopCreateLogSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateLogSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]

    lib.infiniopGetLogSoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetLogSoftmaxWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopLogSoftmax.restype = c_int32
    lib.infiniopLogSoftmax.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyLogSoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroyLogSoftmaxDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def log10_(lib):
    lib.infiniopCreateLog10Descriptor.restype = c_int32
    lib.infiniopCreateLog10Descriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetLog10WorkspaceSize.restype = c_int32
    lib.infiniopGetLog10WorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopLog10.restype = c_int32
    lib.infiniopLog10.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyLog10Descriptor.restype = c_int32
    lib.infiniopDestroyLog10Descriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def log1p_(lib):
    lib.infiniopCreateLog1pDescriptor.restype = c_int32
    lib.infiniopCreateLog1pDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetLog1pWorkspaceSize.restype = c_int32
    lib.infiniopGetLog1pWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopLog1p.restype = c_int32
    lib.infiniopLog1p.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyLog1pDescriptor.restype = c_int32
    lib.infiniopDestroyLog1pDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def histc_(lib):
    lib.infiniopCreateHistcDescriptor.restype = c_int32
    lib.infiniopCreateHistcDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int64,
        c_double,
        c_double,
    ]

    lib.infiniopGetHistcWorkspaceSize.restype = c_int32
    lib.infiniopGetHistcWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopHistc.restype = c_int32
    lib.infiniopHistc.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyHistcDescriptor.restype = c_int32
    lib.infiniopDestroyHistcDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def dot_(lib):
    lib.infiniopCreateDotDescriptor.restype = c_int32
    lib.infiniopCreateDotDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetDotWorkspaceSize.restype = c_int32
    lib.infiniopGetDotWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopDot.restype = c_int32
    lib.infiniopDot.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyDotDescriptor.restype = c_int32
    lib.infiniopDestroyDotDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def avg_pool3d_(lib):
    lib.infiniopCreateAvgPool3dDescriptor.restype = c_int32
    lib.infiniopCreateAvgPool3dDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,  # kernel_size
        c_void_p,  # stride (nullable)
        c_void_p,  # padding (nullable)
    ]

    lib.infiniopGetAvgPool3dWorkspaceSize.restype = c_int32
    lib.infiniopGetAvgPool3dWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAvgPool3d.restype = c_int32
    lib.infiniopAvgPool3d.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAvgPool3dDescriptor.restype = c_int32
    lib.infiniopDestroyAvgPool3dDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
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
def add_rms_norm_(lib):
    lib.infiniopCreateAddRMSNormDescriptor.restype = c_int32
    lib.infiniopCreateAddRMSNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopGetAddRMSNormWorkspaceSize.restype = c_int32
    lib.infiniopGetAddRMSNormWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAddRMSNorm.restype = c_int32
    lib.infiniopAddRMSNorm.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAddRMSNormDescriptor.restype = c_int32
    lib.infiniopDestroyAddRMSNormDescriptor.argtypes = [
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
        infiniopTensorDescriptor_t,
        c_int32,
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
        c_void_p,
    ]

    lib.infiniopDestroyRoPEDescriptor.restype = c_int32
    lib.infiniopDestroyRoPEDescriptor.argtypes = [
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
def softmax_(lib):
    lib.infiniopCreateSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]

    lib.infiniopGetSoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetSoftmaxWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSoftmax.restype = c_int32
    lib.infiniopSoftmax.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroySoftmaxDescriptor.argtypes = [
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
        c_int32,
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
        infiniopTensorDescriptor_t,
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
    lib.infiniopCreateDequantizeAWQDescriptor.restype = c_int32
    lib.infiniopCreateDequantizeAWQDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetDequantizeAWQWorkspaceSize.restype = c_int32
    lib.infiniopGetDequantizeAWQWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopDequantizeAWQ.restype = c_int32
    lib.infiniopDequantizeAWQ.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyDequantizeAWQDescriptor.restype = c_int32
    lib.infiniopDestroyDequantizeAWQDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def dequantize_gptq_(lib):
    lib.infiniopCreateDequantizeGPTQDescriptor.restype = c_int32
    lib.infiniopCreateDequantizeGPTQDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetDequantizeGPTQWorkspaceSize.restype = c_int32
    lib.infiniopGetDequantizeGPTQWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopDequantizeGPTQ.restype = c_int32
    lib.infiniopDequantizeGPTQ.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyDequantizeGPTQDescriptor.restype = c_int32
    lib.infiniopDestroyDequantizeGPTQDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def per_channel_quant_int8_(lib):
    lib.infiniopCreatePerChannelQuantI8Descriptor.restype = c_int32
    lib.infiniopCreatePerChannelQuantI8Descriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetPerChannelQuantI8WorkspaceSize.restype = c_int32
    lib.infiniopGetPerChannelQuantI8WorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopPerChannelQuantI8.restype = c_int32
    lib.infiniopPerChannelQuantI8.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyPerChannelQuantI8Descriptor.restype = c_int32
    lib.infiniopDestroyPerChannelQuantI8Descriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def per_tensor_quant_int8_(lib):
    lib.infiniopCreatePerTensorQuantI8Descriptor.restype = c_int32
    lib.infiniopCreatePerTensorQuantI8Descriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetPerTensorQuantI8WorkspaceSize.restype = c_int32
    lib.infiniopGetPerTensorQuantI8WorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopPerTensorQuantI8.restype = c_int32
    lib.infiniopPerTensorQuantI8.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_bool,
        c_void_p,
    ]

    lib.infiniopDestroyPerTensorQuantI8Descriptor.restype = c_int32
    lib.infiniopDestroyPerTensorQuantI8Descriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def per_tensor_dequant_int8_(lib):
    lib.infiniopCreatePerTensorDequantI8Descriptor.restype = c_int32
    lib.infiniopCreatePerTensorDequantI8Descriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetPerTensorDequantI8WorkspaceSize.restype = c_int32
    lib.infiniopGetPerTensorDequantI8WorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopPerTensorDequantI8.restype = c_int32
    lib.infiniopPerTensorDequantI8.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyPerTensorDequantI8Descriptor.restype = c_int32
    lib.infiniopDestroyPerTensorDequantI8Descriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def gptq_qyblas_gemm_(lib):
    lib.infiniopCreateGptqQyblasGemmDescriptor.restype = c_int32
    lib.infiniopCreateGptqQyblasGemmDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetGptqQyblasGemmWorkspaceSize.restype = c_int32
    lib.infiniopGetGptqQyblasGemmWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopGptqQyblasGemm.restype = c_int32
    lib.infiniopGptqQyblasGemm.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_int64,
        c_int64,
        c_void_p,
    ]

    lib.infiniopDestroyGptqQyblasGemmDescriptor.restype = c_int32
    lib.infiniopDestroyGptqQyblasGemmDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def softplus_(lib):
    lib.infiniopCreateSoftplusDescriptor.restype = c_int32
    lib.infiniopCreateSoftplusDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
        c_float,
    ]
    lib.infiniopSoftplus.restype = c_int32
    lib.infiniopSoftplus.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroySoftplusDescriptor.restype = c_int32
    lib.infiniopDestroySoftplusDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def zeros_(lib):
    lib.infiniopCreateZerosDescriptor.restype = c_int32
    lib.infiniopCreateZerosDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetZerosWorkspaceSize.restype = c_int32
    lib.infiniopGetZerosWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopZeros.restype = c_int32
    lib.infiniopZeros.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyZerosDescriptor.restype = c_int32
    lib.infiniopDestroyZerosDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def ones_(lib):
    lib.infiniopCreateOnesDescriptor.restype = c_int32
    lib.infiniopCreateOnesDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetOnesWorkspaceSize.restype = c_int32
    lib.infiniopGetOnesWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopOnes.restype = c_int32
    lib.infiniopOnes.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyOnesDescriptor.restype = c_int32
    lib.infiniopDestroyOnesDescriptor.argtypes = [
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

    lib.infiniopGetGeluWorkspaceSize.restype = c_int32
    lib.infiniopGetGeluWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
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
    lib.infiniopDestroyGeluDescriptor.argtypes = [
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

    lib.infiniopGetSiluWorkspaceSize.restype = c_int32
    lib.infiniopGetSiluWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
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
    lib.infiniopDestroySiluDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def hardtanh_(lib):
    # 1. Create Descriptor - 注意增加了两个 c_float 参数
    lib.infiniopCreateHardTanhDescriptor.restype = c_int32
    lib.infiniopCreateHardTanhDescriptor.argtypes = [
        infiniopHandle_t,  # handle
        POINTER(infiniopOperatorDescriptor_t),  # desc_ptr
        infiniopTensorDescriptor_t,  # output
        infiniopTensorDescriptor_t,  # input
        c_float,  # min_val
        c_float,  # max_val
    ]

    # 2. Get Workspace Size
    lib.infiniopGetHardTanhWorkspaceSize.restype = c_int32
    lib.infiniopGetHardTanhWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,  # desc
        POINTER(c_size_t),  # size
    ]

    # 3. Execute Operator
    lib.infiniopHardTanh.restype = c_int32
    lib.infiniopHardTanh.argtypes = [
        infiniopOperatorDescriptor_t,  # desc
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # output
        c_void_p,  # input
        c_void_p,  # stream
    ]

    # 4. Destroy Descriptor
    lib.infiniopDestroyHardTanhDescriptor.restype = c_int32
    lib.infiniopDestroyHardTanhDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,  # desc
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

    lib.infiniopGetHardSwishWorkspaceSize.restype = c_int32
    lib.infiniopGetHardSwishWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
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
    lib.infiniopDestroyHardSwishDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def avg_pool1d_(lib):
    # 1. Create 函数
    # C签名: (handle, *desc, y, x, kernel_size, stride, padding)
    lib.infiniopCreateAvgPool1dDescriptor.restype = c_int32
    lib.infiniopCreateAvgPool1dDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # y_desc (Output)
        infiniopTensorDescriptor_t,  # x_desc (Input)
        c_size_t,  # kernel_size
        c_size_t,  # stride
        c_size_t,  # padding
    ]

    # 2. GetWorkspaceSize 函数
    lib.infiniopGetAvgPool1dWorkspaceSize.restype = c_int32
    lib.infiniopGetAvgPool1dWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    # 3. Execute 函数
    lib.infiniopAvgPool1d.restype = c_int32
    lib.infiniopAvgPool1d.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # y (output pointer)
        c_void_p,  # x (input pointer)
        c_void_p,  # stream
    ]

    # 4. Destroy 函数
    lib.infiniopDestroyAvgPool1dDescriptor.restype = c_int32
    lib.infiniopDestroyAvgPool1dDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def layer_norm_(lib):
    lib.infiniopCreateLayerNormDescriptor.restype = c_int32
    lib.infiniopCreateLayerNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]
    lib.infiniopGetLayerNormWorkspaceSize.restype = c_int32
    lib.infiniopGetLayerNormWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]
    lib.infiniopLayerNorm.restype = c_int32
    lib.infiniopLayerNorm.argtypes = [
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

    lib.infiniopDestroyLayerNormDescriptor.restype = c_int32
    lib.infiniopDestroyLayerNormDescriptor.argtypes = [infiniopOperatorDescriptor_t]


@OpRegister.operator
def lp_norm_(lib):
    lib.infiniopCreateLPNormDescriptor.restype = c_int32
    lib.infiniopCreateLPNormDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
        c_int32,
        c_float,
    ]

    lib.infiniopGetLPNormWorkspaceSize.restype = c_int32
    lib.infiniopGetLPNormWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopLPNorm.restype = c_int32
    lib.infiniopLPNorm.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyLPNormDescriptor.restype = c_int32
    lib.infiniopDestroyLPNormDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def tanh_(lib):
    lib.infiniopCreateTanhDescriptor.restype = c_int32
    lib.infiniopCreateTanhDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
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
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyTanhDescriptor.restype = c_int32
    lib.infiniopDestroyTanhDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def scaled_mm_int8_(lib):
    lib.infiniopCreateI8GemmDescriptor.restype = c_int32
    lib.infiniopCreateI8GemmDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetI8GemmWorkspaceSize.restype = c_int32
    lib.infiniopGetI8GemmWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopI8Gemm.restype = c_int32
    lib.infiniopI8Gemm.argtypes = [
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

    lib.infiniopDestroyI8GemmDescriptor.restype = c_int32
    lib.infiniopDestroyI8GemmDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def kv_caching_(lib):
    lib.infiniopCreateKVCachingDescriptor.restype = c_int32
    lib.infiniopCreateKVCachingDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetKVCachingWorkspaceSize.restype = c_int32
    lib.infiniopGetKVCachingWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopKVCaching.restype = c_int32
    lib.infiniopKVCaching.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyKVCachingDescriptor.restype = c_int32
    lib.infiniopDestroyKVCachingDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def paged_attention_(lib):
    lib.infiniopCreatePagedAttentionDescriptor.restype = c_int32
    lib.infiniopCreatePagedAttentionDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_float,
    ]

    lib.infiniopGetPagedAttentionWorkspaceSize.restype = c_int32
    lib.infiniopGetPagedAttentionWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopPagedAttention.restype = c_int32
    lib.infiniopPagedAttention.argtypes = [
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
        c_void_p,
    ]

    lib.infiniopDestroyPagedAttentionDescriptor.restype = c_int32
    lib.infiniopDestroyPagedAttentionDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def paged_caching_(lib):
    lib.infiniopCreatePagedCachingDescriptor.restype = c_int32
    lib.infiniopCreatePagedCachingDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,  # k_cache_desc
        infiniopTensorDescriptor_t,  # v_cache_desc
        infiniopTensorDescriptor_t,  # k_desc
        infiniopTensorDescriptor_t,  # v_desc
        infiniopTensorDescriptor_t,  # slot_mapping_desc
    ]

    # infiniopGetPagedCachingWorkspaceSize
    lib.infiniopGetPagedCachingWorkspaceSize.restype = c_int32
    lib.infiniopGetPagedCachingWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    # infiniopPagedCaching
    lib.infiniopPagedCaching.restype = c_int32
    lib.infiniopPagedCaching.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,  # workspace
        c_size_t,  # workspace_size
        c_void_p,  # k_cache
        c_void_p,  # v_cache
        c_void_p,  # k
        c_void_p,  # v
        c_void_p,  # slot_mapping
        c_void_p,  # stream
    ]

    # infiniopDestroyPagedCachingDescriptor
    lib.infiniopDestroyPagedCachingDescriptor.restype = c_int32
    lib.infiniopDestroyPagedCachingDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def paged_attention_prefill_(lib):
    lib.infiniopCreatePagedAttentionPrefillDescriptor.restype = c_int32
    lib.infiniopCreatePagedAttentionPrefillDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopGetPagedAttentionPrefillWorkspaceSize.restype = c_int32
    lib.infiniopGetPagedAttentionPrefillWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopPagedAttentionPrefill.restype = c_int32
    lib.infiniopPagedAttentionPrefill.argtypes = [
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
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyPagedAttentionPrefillDescriptor.restype = c_int32
    lib.infiniopDestroyPagedAttentionPrefillDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def silu_and_mul(lib):
    lib.infiniopCreateSiluAndMulDescriptor.restype = c_int32
    lib.infiniopCreateSiluAndMulDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSiluAndMulWorkspaceSize.restype = c_int32
    lib.infiniopGetSiluAndMulWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSiluAndMul.restype = c_int32
    lib.infiniopSiluAndMul.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySiluAndMulDescriptor.restype = c_int32
    lib.infiniopDestroySiluAndMulDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def erf(lib):
    lib.infiniopCreateErfDescriptor.restype = c_int32
    lib.infiniopCreateErfDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetErfWorkspaceSize.restype = c_int32
    lib.infiniopGetErfWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopErf.restype = c_int32
    lib.infiniopErf.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyErfDescriptor.restype = c_int32
    lib.infiniopDestroyErfDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def erfc(lib):
    lib.infiniopCreateErfcDescriptor.restype = c_int32
    lib.infiniopCreateErfcDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetErfcWorkspaceSize.restype = c_int32
    lib.infiniopGetErfcWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopErfc.restype = c_int32
    lib.infiniopErfc.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyErfcDescriptor.restype = c_int32
    lib.infiniopDestroyErfcDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def erfinv(lib):
    lib.infiniopCreateErfinvDescriptor.restype = c_int32
    lib.infiniopCreateErfinvDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetErfinvWorkspaceSize.restype = c_int32
    lib.infiniopGetErfinvWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopErfinv.restype = c_int32
    lib.infiniopErfinv.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyErfinvDescriptor.restype = c_int32
    lib.infiniopDestroyErfinvDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def matrix_power(lib):
    lib.infiniopCreateMatrixPowerDescriptor.restype = c_int32
    lib.infiniopCreateMatrixPowerDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]

    lib.infiniopGetMatrixPowerWorkspaceSize.restype = c_int32
    lib.infiniopGetMatrixPowerWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopMatrixPower.restype = c_int32
    lib.infiniopMatrixPower.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyMatrixPowerDescriptor.restype = c_int32
    lib.infiniopDestroyMatrixPowerDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def pixel_shuffle(lib):
    lib.infiniopCreatePixelShuffleDescriptor.restype = c_int32
    lib.infiniopCreatePixelShuffleDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]

    lib.infiniopGetPixelShuffleWorkspaceSize.restype = c_int32
    lib.infiniopGetPixelShuffleWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopPixelShuffle.restype = c_int32
    lib.infiniopPixelShuffle.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyPixelShuffleDescriptor.restype = c_int32
    lib.infiniopDestroyPixelShuffleDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def fused_ffn_(lib):
    lib.infiniopCreateFusedFFNDescriptor.restype = c_int32
    lib.infiniopCreateFusedFFNDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_float,
    ]

    lib.infiniopGetFusedFFNWorkspaceSize.restype = c_int32
    lib.infiniopGetFusedFFNWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopFusedFFN.restype = c_int32
    lib.infiniopFusedFFN.argtypes = [
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

    lib.infiniopDestroyFusedFFNDescriptor.restype = c_int32
    lib.infiniopDestroyFusedFFNDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def blas_amax_(lib):
    lib.infiniopCreateBlasAmaxDescriptor.restype = c_int32
    lib.infiniopCreateBlasAmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetBlasAmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetBlasAmaxWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopBlasAmax.restype = c_int32
    lib.infiniopBlasAmax.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyBlasAmaxDescriptor.restype = c_int32
    lib.infiniopDestroyBlasAmaxDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def blas_amin_(lib):
    lib.infiniopCreateBlasAminDescriptor.restype = c_int32
    lib.infiniopCreateBlasAminDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetBlasAminWorkspaceSize.restype = c_int32
    lib.infiniopGetBlasAminWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopBlasAmin.restype = c_int32
    lib.infiniopBlasAmin.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyBlasAminDescriptor.restype = c_int32
    lib.infiniopDestroyBlasAminDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def asum_(lib):
    lib.infiniopCreateAsumDescriptor.restype = c_int32
    lib.infiniopCreateAsumDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetAsumWorkspaceSize.restype = c_int32
    lib.infiniopGetAsumWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAsum.restype = c_int32
    lib.infiniopAsum.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAsumDescriptor.restype = c_int32
    lib.infiniopDestroyAsumDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def axpy_(lib):
    lib.infiniopCreateAxpyDescriptor.restype = c_int32
    lib.infiniopCreateAxpyDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetAxpyWorkspaceSize.restype = c_int32
    lib.infiniopGetAxpyWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopAxpy.restype = c_int32
    lib.infiniopAxpy.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyAxpyDescriptor.restype = c_int32
    lib.infiniopDestroyAxpyDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def blas_copy_(lib):
    lib.infiniopCreateBlasCopyDescriptor.restype = c_int32
    lib.infiniopCreateBlasCopyDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetBlasCopyWorkspaceSize.restype = c_int32
    lib.infiniopGetBlasCopyWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopBlasCopy.restype = c_int32
    lib.infiniopBlasCopy.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyBlasCopyDescriptor.restype = c_int32
    lib.infiniopDestroyBlasCopyDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def blas_dot_(lib):
    lib.infiniopCreateBlasDotDescriptor.restype = c_int32
    lib.infiniopCreateBlasDotDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetBlasDotWorkspaceSize.restype = c_int32
    lib.infiniopGetBlasDotWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopBlasDot.restype = c_int32
    lib.infiniopBlasDot.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyBlasDotDescriptor.restype = c_int32
    lib.infiniopDestroyBlasDotDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def nrm2_(lib):
    lib.infiniopCreateNrm2Descriptor.restype = c_int32
    lib.infiniopCreateNrm2Descriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetNrm2WorkspaceSize.restype = c_int32
    lib.infiniopGetNrm2WorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopNrm2.restype = c_int32
    lib.infiniopNrm2.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyNrm2Descriptor.restype = c_int32
    lib.infiniopDestroyNrm2Descriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def rot_(lib):
    lib.infiniopCreateRotDescriptor.restype = c_int32
    lib.infiniopCreateRotDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRotWorkspaceSize.restype = c_int32
    lib.infiniopGetRotWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRot.restype = c_int32
    lib.infiniopRot.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRotDescriptor.restype = c_int32
    lib.infiniopDestroyRotDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def rotg_(lib):
    lib.infiniopCreateRotgDescriptor.restype = c_int32
    lib.infiniopCreateRotgDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRotgWorkspaceSize.restype = c_int32
    lib.infiniopGetRotgWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRotg.restype = c_int32
    lib.infiniopRotg.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRotgDescriptor.restype = c_int32
    lib.infiniopDestroyRotgDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def rotm_(lib):
    lib.infiniopCreateRotmDescriptor.restype = c_int32
    lib.infiniopCreateRotmDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRotmWorkspaceSize.restype = c_int32
    lib.infiniopGetRotmWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRotm.restype = c_int32
    lib.infiniopRotm.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRotmDescriptor.restype = c_int32
    lib.infiniopDestroyRotmDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def rotmg_(lib):
    lib.infiniopCreateRotmgDescriptor.restype = c_int32
    lib.infiniopCreateRotmgDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetRotmgWorkspaceSize.restype = c_int32
    lib.infiniopGetRotmgWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopRotmg.restype = c_int32
    lib.infiniopRotmg.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyRotmgDescriptor.restype = c_int32
    lib.infiniopDestroyRotmgDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def scal_(lib):
    lib.infiniopCreateScalDescriptor.restype = c_int32
    lib.infiniopCreateScalDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetScalWorkspaceSize.restype = c_int32
    lib.infiniopGetScalWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopScal.restype = c_int32
    lib.infiniopScal.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroyScalDescriptor.restype = c_int32
    lib.infiniopDestroyScalDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]


@OpRegister.operator
def swap_(lib):
    lib.infiniopCreateSwapDescriptor.restype = c_int32
    lib.infiniopCreateSwapDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopOperatorDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]

    lib.infiniopGetSwapWorkspaceSize.restype = c_int32
    lib.infiniopGetSwapWorkspaceSize.argtypes = [
        infiniopOperatorDescriptor_t,
        POINTER(c_size_t),
    ]

    lib.infiniopSwap.restype = c_int32
    lib.infiniopSwap.argtypes = [
        infiniopOperatorDescriptor_t,
        c_void_p,
        c_size_t,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySwapDescriptor.restype = c_int32
    lib.infiniopDestroySwapDescriptor.argtypes = [
        infiniopOperatorDescriptor_t,
    ]
