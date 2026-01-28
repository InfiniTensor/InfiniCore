"""
Base test template for binary operators.

This module provides a unified test framework for all binary operators,
eliminating code duplication across individual test scripts.

Usage:
    from libinfiniop.binary_test_base import BinaryTestBase
    
    class DivTest(BinaryTestBase):
        OP_NAME = "Div"
        OP_NAME_LOWER = "div"
        
        @staticmethod
        def torch_op(c, a, b):
            torch.div(a, b, out=c)
        
        @staticmethod
        def generate_input_a(shape, dtype, device):
            return TestTensor(shape, None, dtype, device)
        
        @staticmethod
        def generate_input_b(shape, dtype, device):
            # For division, ensure b doesn't contain zeros
            return TestTensor(shape, None, dtype, device, scale=2, bias=0.1)
        
        TOLERANCE_MAP = {
            InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
            InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
        }
    
    if __name__ == "__main__":
        DivTest.run()
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


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


# Common test cases for binary operators
_BINARY_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((13, 16, 2), (128, 4, 1), (0, 2, 1), (64, 4, 1)),
    ((13, 16, 2), (128, 4, 1), (2, 0, 1), (64, 4, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]

# Inplace options applied for each test case
_BINARY_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_A,
    Inplace.INPLACE_B,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_BINARY_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _BINARY_TEST_CASES_
    for inplace_item in _BINARY_INPLACE
]

# Data types used for testing (matching old operators library: only F16 and F32)
_BINARY_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]


class BinaryTestBase:
    """
    Base class for binary operator tests.
    
    Subclasses must define:
    - OP_NAME: Uppercase operator name (e.g., "Div", "Pow")
    - OP_NAME_LOWER: Lowercase operator name (e.g., "div", "pow")
    - torch_op: Static method that performs the PyTorch operation
    - generate_input_a: Static method that generates first input tensor
    - generate_input_b: Static method that generates second input tensor
    - TOLERANCE_MAP: Dictionary mapping dtype to tolerance values
    """
    
    OP_NAME = None
    OP_NAME_LOWER = None
    
    # Default tolerance map (can be overridden)
    TOLERANCE_MAP = {
        InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
        InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    }
    
    # Test cases (can be overridden)
    TEST_CASES = _BINARY_TEST_CASES
    TENSOR_DTYPES = _BINARY_TENSOR_DTYPES
    
    DEBUG = False
    PROFILE = False
    NUM_PRERUN = 10
    NUM_ITERATIONS = 1000
    
    @staticmethod
    def torch_op(c, a, b):
        """PyTorch operation - must be implemented by subclass"""
        raise NotImplementedError("Subclass must implement torch_op")
    
    @staticmethod
    def generate_input_a(shape, a_stride, dtype, device):
        """
        Generate first input tensor - must be implemented by subclass.
        
        Args:
            shape: Tensor shape tuple
            a_stride: Stride tuple or None
            dtype: InfiniDtype enum value
            device: InfiniDeviceEnum value
        
        Returns:
            TestTensor: Generated first input tensor
        """
        raise NotImplementedError("Subclass must implement generate_input_a")
    
    @staticmethod
    def generate_input_b(shape, b_stride, dtype, device):
        """
        Generate second input tensor - must be implemented by subclass.
        
        Args:
            shape: Tensor shape tuple
            b_stride: Stride tuple or None
            dtype: InfiniDtype enum value
            device: InfiniDeviceEnum value
        
        Returns:
            TestTensor: Generated second input tensor
        """
        raise NotImplementedError("Subclass must implement generate_input_b")
    
    @classmethod
    def test(cls, handle, device, shape, a_stride=None, b_stride=None, c_stride=None, 
             inplace=Inplace.OUT_OF_PLACE, dtype=InfiniDtype.F16, sync=None):
        """Common test function for binary operators"""
        a = cls.generate_input_a(shape, a_stride, dtype, device)
        b = cls.generate_input_b(shape, b_stride, dtype, device)
        
        if inplace == Inplace.INPLACE_A:
            if c_stride is not None and c_stride != a_stride:
                return
            c = a
        elif inplace == Inplace.INPLACE_B:
            if c_stride is not None and c_stride != b_stride:
                return
            c = b
        else:
            c = TestTensor(shape, c_stride, dtype, device)
        
        if c.is_broadcast():
            return
        
        print(
            f"Testing {cls.OP_NAME} on {InfiniDeviceNames[device]} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
            f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
        )
        
        cls.torch_op(c.torch_tensor(), a.torch_tensor(), b.torch_tensor())
        
        if sync is not None:
            sync()
        
        descriptor = infiniopOperatorDescriptor_t()
        create_func = getattr(LIBINFINIOP, f"infiniopCreate{cls.OP_NAME}Descriptor")
        check_error(
            create_func(
                handle,
                ctypes.byref(descriptor),
                c.descriptor,
                a.descriptor,
                b.descriptor,
            )
        )
        
        # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
        for tensor in [a, b, c]:
            tensor.destroy_desc()
        
        workspace_size = c_uint64(0)
        get_workspace_func = getattr(LIBINFINIOP, f"infiniopGet{cls.OP_NAME}WorkspaceSize")
        check_error(
            get_workspace_func(
                descriptor, ctypes.byref(workspace_size)
            )
        )
        workspace = TestWorkspace(workspace_size.value, device)
        
        def lib_op():
            op_func = getattr(LIBINFINIOP, f"infiniop{cls.OP_NAME}")
            check_error(
                op_func(
                    descriptor,
                    workspace.data(),
                    workspace_size.value,
                    c.data(),
                    a.data(),
                    b.data(),
                    None,
                )
            )
        
        lib_op()
        if sync is not None:
            sync()
        
        atol, rtol = get_tolerance(cls.TOLERANCE_MAP, dtype)
        if cls.DEBUG:
            debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
        
        equal_nan = getattr(cls, 'EQUAL_NAN', False)
        assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol, equal_nan=equal_nan)
        
        # Profiling workflow
        if cls.PROFILE:
            # fmt: off
            profile_operation("PyTorch", lambda: cls.torch_op(c.torch_tensor(), a.torch_tensor(), b.torch_tensor()), device, cls.NUM_PRERUN, cls.NUM_ITERATIONS)
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
