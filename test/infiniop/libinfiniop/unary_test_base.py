"""
Base test template for unary operators.

This module provides a unified test framework for all unary operators,
eliminating code duplication across individual test scripts.

Usage:
    from libinfiniop.unary_test_base import UnaryTestBase
    
    class AbsTest(UnaryTestBase):
        OP_NAME = "Abs"
        OP_NAME_LOWER = "abs"
        
        @staticmethod
        def torch_op(x):
            return torch.abs(x).to(x.dtype)
        
        @staticmethod
        def generate_input(shape, dtype, device):
            # Generate test tensors with values in range [-1, 1) for abs operation
            return torch.rand(shape, dtype=dtype, device=device) * 2 - 1
        
        TOLERANCE_MAP = {
            InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
            InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
        }
    
    if __name__ == "__main__":
        AbsTest.run()
"""

import ctypes
from ctypes import c_uint64
from enum import Enum, auto

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)
from libinfiniop.utils import to_torch_dtype
from libinfiniop.devices import torch_device_map


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


# Common test cases for unary operators
_UNARY_TEST_CASES_ = [
    # tensor_shape, inplace
    ((1, 3),),
    ((3, 3),),
    ((32, 20, 512),),
    ((33, 333, 333),),
    ((32, 256, 112, 112),),
    ((3, 3, 13, 9, 17),),
]

# Inplace options applied for each test case
_UNARY_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_UNARY_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _UNARY_TEST_CASES_
    for inplace_item in _UNARY_INPLACE
]

# Data types used for testing (matching old operators library: only F16 and F32)
_UNARY_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]


class UnaryTestBase:
    """
    Base class for unary operator tests.
    
    Subclasses must define:
    - OP_NAME: Uppercase operator name (e.g., "Abs", "Log")
    - OP_NAME_LOWER: Lowercase operator name (e.g., "abs", "log")
    - torch_op: Static method that performs the PyTorch operation
    - generate_input: Static method that generates input tensor
    - TOLERANCE_MAP: Dictionary mapping dtype to tolerance values
    """
    
    OP_NAME = None
    OP_NAME_LOWER = None
    
    # Default tolerance map (can be overridden)
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    }
    
    # Test cases (can be overridden)
    TEST_CASES = _UNARY_TEST_CASES
    TENSOR_DTYPES = _UNARY_TENSOR_DTYPES
    
    DEBUG = False
    PROFILE = False
    NUM_PRERUN = 10
    NUM_ITERATIONS = 1000
    
    @staticmethod
    def torch_op(x):
        """PyTorch operation - must be implemented by subclass"""
        raise NotImplementedError("Subclass must implement torch_op")
    
    @staticmethod
    def generate_input(shape, dtype, device):
        """
        Generate input tensor - must be implemented by subclass.
        
        Args:
            shape: Tensor shape tuple
            dtype: PyTorch dtype (e.g., torch.float16, torch.float32)
            device: PyTorch device string (e.g., "cpu", "cuda")
        
        Returns:
            torch.Tensor: Generated input tensor
        """
        raise NotImplementedError("Subclass must implement generate_input")
    
    @classmethod
    def test(cls, handle, device, shape, inplace=Inplace.OUT_OF_PLACE, dtype=InfiniDtype.F16, sync=None):
        """Common test function for unary operators"""
        from libinfiniop.devices import torch_device_map
        from libinfiniop.utils import to_torch_dtype
        
        # Generate input tensor
        torch_dtype = to_torch_dtype(dtype)
        torch_device = torch_device_map[device]
        x_torch_tensor = cls.generate_input(shape, torch_dtype, torch_device)
        
        x = TestTensor(
            shape,
            x_torch_tensor.stride(),
            dtype,
            device,
            mode="manual",
            set_tensor=x_torch_tensor,
        )
        
        if inplace == Inplace.INPLACE_X:
            y = x
        else:
            y = TestTensor(shape, None, dtype, device)
        
        if y.is_broadcast():
            return
        
        print(
            f"Testing {cls.OP_NAME} on {InfiniDeviceNames[device]} with shape:{shape} dtype:{InfiniDtypeNames[dtype]} inplace: {inplace}"
        )
        
        ans = cls.torch_op(x.torch_tensor())
        
        if sync is not None:
            sync()
        
        descriptor = infiniopOperatorDescriptor_t()
        create_func = getattr(LIBINFINIOP, f"infiniopCreate{cls.OP_NAME}Descriptor")
        check_error(
            create_func(
                handle, ctypes.byref(descriptor), y.descriptor, x.descriptor
            )
        )
        
        # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
        for tensor in [x, y]:
            tensor.destroy_desc()
        
        workspace_size = c_uint64(0)
        get_workspace_func = getattr(LIBINFINIOP, f"infiniopGet{cls.OP_NAME}WorkspaceSize")
        check_error(
            get_workspace_func(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, y.device)
        
        def lib_op():
            op_func = getattr(LIBINFINIOP, f"infiniop{cls.OP_NAME}")
            check_error(
                op_func(
                    descriptor, workspace.data(), workspace_size.value, y.data(), x.data(), None
                )
            )
        
        lib_op()
        if sync is not None:
            sync()
        
        atol, rtol = get_tolerance(cls.TOLERANCE_MAP, dtype)
        equal_nan = getattr(cls, 'EQUAL_NAN', False)
        
        if cls.DEBUG:
            debug(y.actual_tensor(), ans, atol=atol, rtol=rtol, equal_nan=equal_nan)
        
        assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol, equal_nan=equal_nan)
        
        # Profiling workflow
        if cls.PROFILE:
            # fmt: off
            profile_operation("PyTorch", lambda: cls.torch_op(x.torch_tensor()), device, cls.NUM_PRERUN, cls.NUM_ITERATIONS)
            profile_operation("    lib", lambda: lib_op(), device, cls.NUM_PRERUN, cls.NUM_ITERATIONS)
            # fmt: on
        
        destroy_func = getattr(LIBINFINIOP, f"infiniopDestroy{cls.OP_NAME}Descriptor")
        check_error(destroy_func(descriptor))
    
    @classmethod
    def run(cls):
        """Run the test"""
        args = get_args()
        
        # Configure testing options
        cls.DEBUG = args.debug
        cls.PROFILE = args.profile
        cls.NUM_PRERUN = args.num_prerun
        cls.NUM_ITERATIONS = args.num_iterations
        
        for device in get_test_devices(args):
            test_operator(device, cls.test, cls.TEST_CASES, cls.TENSOR_DTYPES)
        
        print("\033[92mTest passed!\033[0m")
