from .liboperators import (open_lib, infiniopHandle_t, infiniopTensorDescriptor_t)
from ctypes import POINTER, Structure, c_int32, c_size_t, c_uint64, c_void_p, c_float

lib = open_lib()

#########################################################
# Gemm
class GemmDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopGemmDescriptor_t = POINTER(GemmDescriptor)

lib.infiniopCreateGemmDescriptor.restype = c_int32
lib.infiniopCreateGemmDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopGemmDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
]

lib.infiniopGetGemmWorkspaceSize.restype = c_int32
lib.infiniopGetGemmWorkspaceSize.argtypes = [
    infiniopGemmDescriptor_t,
    POINTER(c_size_t),
]

lib.infiniopGemm.restype = c_int32
lib.infiniopGemm.argtypes = [
    infiniopGemmDescriptor_t,
    c_void_p,
    c_uint64,
    c_void_p,
    c_void_p,
    c_void_p,
    c_float,
    c_float,
    c_void_p,
]

lib.infiniopDestroyGemmDescriptor.restype = c_int32
lib.infiniopDestroyGemmDescriptor.argtypes = [
    infiniopGemmDescriptor_t,
]

#########################################################
# casual softmax
class CausalSoftmaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopCausalSoftmaxDescriptor_t = POINTER(CausalSoftmaxDescriptor)

lib.infiniopCreateCausalSoftmaxDescriptor.restype = c_int32
lib.infiniopCreateCausalSoftmaxDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopCausalSoftmaxDescriptor_t),
    infiniopTensorDescriptor_t,
]

lib.infiniopGetCausalSoftmaxWorkspaceSize.restype = c_int32
lib.infiniopGetCausalSoftmaxWorkspaceSize.argtypes = [
    infiniopCausalSoftmaxDescriptor_t,
    POINTER(c_uint64),
]

lib.infiniopCausalSoftmax.restype = c_int32
lib.infiniopCausalSoftmax.argtypes = [
    infiniopCausalSoftmaxDescriptor_t,
    c_void_p,
    c_uint64,
    c_void_p,
    c_void_p,
]

lib.infiniopDestroyCausalSoftmaxDescriptor.restype = c_int32
lib.infiniopDestroyCausalSoftmaxDescriptor.argtypes = [
    infiniopCausalSoftmaxDescriptor_t,
]

#########################################################
# rms_norm
class RMSNormDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopRMSNormDescriptor_t = POINTER(RMSNormDescriptor)
    
lib.infiniopCreateRMSNormDescriptor.restype = c_int32
lib.infiniopCreateRMSNormDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopRMSNormDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    c_float,
]

lib.infiniopGetRMSNormWorkspaceSize.restype = c_int32
lib.infiniopGetRMSNormWorkspaceSize.argtypes = [
    infiniopRMSNormDescriptor_t,
    POINTER(c_uint64),
]

lib.infiniopRMSNorm.restypes = c_int32
lib.infiniopRMSNorm.argtypes = [
    infiniopRMSNormDescriptor_t,
    c_void_p,
    c_uint64,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]

lib.infiniopDestroyRMSNormDescriptor.restype = c_int32
lib.infiniopDestroyRMSNormDescriptor.argtypes = [
    infiniopRMSNormDescriptor_t,
]

#########################################################
# swiglu
class SwiGLUDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSwiGLUDescriptor_t = POINTER(SwiGLUDescriptor)

lib.infiniopCreateSwiGLUDescriptor.restype = c_int32
lib.infiniopCreateSwiGLUDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopSwiGLUDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
]

lib.infiniopGetSwiGLUWorkspaceSize.restype = c_int32
lib.infiniopGetSwiGLUWorkspaceSize.argtypes = [
    infiniopSwiGLUDescriptor_t,
    POINTER(c_uint64),
]

lib.infiniopSwiGLU.restype = c_int32
lib.infiniopSwiGLU.argtypes = [
    infiniopSwiGLUDescriptor_t,
    c_void_p,
    c_uint64,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]

lib.infiniopDestroySwiGLUDescriptor.restype = c_int32
lib.infiniopDestroySwiGLUDescriptor.argtypes = [
    infiniopSwiGLUDescriptor_t,
]

#########################################################
# random_sample
class RandomSampleDescriptor(Structure):
    _fields_ = [("device", c_int32)]

infiniopRandomSampleDescriptor_t = POINTER(RandomSampleDescriptor)

lib.infiniopCreateRandomSampleDescriptor.restype = c_int32
lib.infiniopCreateRandomSampleDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopRandomSampleDescriptor_t),
    infiniopTensorDescriptor_t,
]

lib.infiniopGetRandomSampleWorkspaceSize.restype = c_int32
lib.infiniopGetRandomSampleWorkspaceSize.argtypes = [
    infiniopRandomSampleDescriptor_t,
    POINTER(c_uint64),
]

lib.infiniopRandomSample.restype = c_int32
lib.infiniopRandomSample.argtypes = [
    infiniopRandomSampleDescriptor_t,
    c_void_p,
    c_uint64,
    c_uint64,
    c_void_p,
    c_float,
    c_float,
    c_int32,
    c_float,
    c_void_p,
]

lib.infiniopDestroyRandomSampleDescriptor.restype = c_int32
lib.infiniopDestroyRandomSampleDescriptor.argtypes = [
    infiniopRandomSampleDescriptor_t,
]

__all__ = ['lib', 'infiniopGemmDescriptor_t', 'infiniopCausalSoftmaxDescriptor_t',
           'infiniopRMSNormDescriptor_t', 'infiniopSwiGLUDescriptor_t', 'infiniopRandomSampleDescriptor_t']
