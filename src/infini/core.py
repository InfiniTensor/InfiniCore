from ctypes import POINTER, byref, c_int

import libinfiniop

import _infini

lib = libinfiniop.LIBINFINIOP

infiniStatus_t = c_int
lib.INFINI_STATUS_SUCCESS = _infini.Status.SUCCESS.value
lib.INFINI_STATUS_INTERNAL_ERROR = _infini.Status.INTERNAL_ERROR.value
lib.INFINI_STATUS_NOT_IMPLEMENTED = _infini.Status.NOT_IMPLEMENTED.value
lib.INFINI_STATUS_BAD_PARAM = _infini.Status.BAD_PARAM.value
lib.INFINI_STATUS_NULL_POINTER = _infini.Status.NULL_POINTER.value
lib.INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED = (
    _infini.Status.DEVICE_TYPE_NOT_SUPPORTED.value
)
lib.INFINI_STATUS_DEVICE_NOT_FOUND = _infini.Status.DEVICE_NOT_FOUND.value
lib.INFINI_STATUS_DEVICE_NOT_INITIALIZED = _infini.Status.DEVICE_NOT_INITIALIZED.value
lib.INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED = (
    _infini.Status.DEVICE_ARCHITECTURE_NOT_SUPPORTED.value
)
lib.INFINI_STATUS_BAD_TENSOR_DTYPE = _infini.Status.BAD_TENSOR_DTYPE.value
lib.INFINI_STATUS_BAD_TENSOR_SHAPE = _infini.Status.BAD_TENSOR_SHAPE.value
lib.INFINI_STATUS_BAD_TENSOR_STRIDES = _infini.Status.BAD_TENSOR_STRIDES.value
lib.INFINI_STATUS_INSUFFICIENT_WORKSPACE = _infini.Status.INSUFFICIENT_WORKSPACE.value

infiniDevice_t = c_int
lib.INFINI_DEVICE_CPU = _infini.Device.CPU.value
lib.INFINI_DEVICE_NVIDIA = _infini.Device.NVIDIA.value
lib.INFINI_DEVICE_CAMBRICON = _infini.Device.CAMBRICON.value
lib.INFINI_DEVICE_ASCEND = _infini.Device.ASCEND.value
lib.INFINI_DEVICE_METAX = _infini.Device.METAX.value
lib.INFINI_DEVICE_MOORE = _infini.Device.MOORE.value
lib.INFINI_DEVICE_ILUVATAR = _infini.Device.ILUVATAR.value
lib.INFINI_DEVICE_KUNLUN = _infini.Device.KUNLUN.value
lib.INFINI_DEVICE_SUGON = _infini.Device.SUGON.value
lib.INFINI_DEVICE_TYPE_COUNT = _infini.Device.TYPE_COUNT.value

infiniDtype_t = c_int
lib.INFINI_DTYPE_INVALID = _infini.Dtype.INVALID.value
lib.INFINI_DTYPE_BYTE = _infini.Dtype.BYTE.value
lib.INFINI_DTYPE_BOOL = _infini.Dtype.BOOL.value
lib.INFINI_DTYPE_I8 = _infini.Dtype.I8.value
lib.INFINI_DTYPE_I16 = _infini.Dtype.I16.value
lib.INFINI_DTYPE_I32 = _infini.Dtype.I32.value
lib.INFINI_DTYPE_I64 = _infini.Dtype.I64.value
lib.INFINI_DTYPE_U8 = _infini.Dtype.U8.value
lib.INFINI_DTYPE_U16 = _infini.Dtype.U16.value
lib.INFINI_DTYPE_U32 = _infini.Dtype.U32.value
lib.INFINI_DTYPE_U64 = _infini.Dtype.U64.value
lib.INFINI_DTYPE_F8 = _infini.Dtype.F8.value
lib.INFINI_DTYPE_F16 = _infini.Dtype.F16.value
lib.INFINI_DTYPE_F32 = _infini.Dtype.F32.value
lib.INFINI_DTYPE_F64 = _infini.Dtype.F64.value
lib.INFINI_DTYPE_C16 = _infini.Dtype.C16.value
lib.INFINI_DTYPE_C32 = _infini.Dtype.C32.value
lib.INFINI_DTYPE_C64 = _infini.Dtype.C64.value
lib.INFINI_DTYPE_C128 = _infini.Dtype.C128.value
lib.INFINI_DTYPE_BF16 = _infini.Dtype.BF16.value

lib.infinirtGetAllDeviceCount.argtypes = [POINTER(c_int)]
lib.infinirtGetAllDeviceCount.restype = infiniStatus_t

lib.infinirtGetDeviceCount.argtypes = [infiniDevice_t, POINTER(c_int)]
lib.infinirtGetDeviceCount.restype = infiniStatus_t

lib.infinirtSetDevice.argtypes = [infiniDevice_t, c_int]
lib.infinirtSetDevice.restype = infiniStatus_t

lib.infinirtGetDevice.argtypes = [POINTER(infiniDevice_t), POINTER(c_int)]
lib.infinirtGetDevice.restype = infiniStatus_t

lib.infinirtDeviceSynchronize.argtypes = []
lib.infinirtDeviceSynchronize.restype = infiniStatus_t

infiniopHandle_t = libinfiniop.infiniopHandle_t

infiniopOperatorDescriptor_t = libinfiniop.infiniopOperatorDescriptor_t

infiniopTensorDescriptor_t = libinfiniop.infiniopTensorDescriptor_t


def get_all_device_count():
    count_array = (c_int * lib.INFINI_DEVICE_TYPE_COUNT)()

    # TODO: Add error handling.
    lib.infinirtGetAllDeviceCount(count_array)

    return tuple(count_array[i] for i in range(len(count_array)))


def get_device_count(device):
    count = c_int()

    # TODO: Add error handling.
    lib.infinirtGetDeviceCount(device, byref(count))

    return count.value


def set_device(device, device_id):
    # TODO: Add error handling.
    lib.infinirtSetDevice(device, device_id)


def get_device():
    device = infiniDevice_t()
    device_id = c_int()

    # TODO: Add error handling.
    lib.infinirtGetDevice(byref(device), byref(device_id))

    return device.value, device_id.value


def device_synchronize():
    # TODO: Add error handling.
    lib.infinirtDeviceSynchronize()
