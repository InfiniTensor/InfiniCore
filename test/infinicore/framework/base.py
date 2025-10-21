import torch
import infinicore
from .devices import InfiniDeviceNames
from .utils import synchronize_device


class TestCase:
    """Base test case class"""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __str__(self):
        return f"TestCase{self.args}"


class TestConfig:
    """Test configuration"""

    def __init__(
        self,
        tensor_dtypes,
        tolerance_map,
        debug=False,
        bench=False,
        num_prerun=10,
        num_iterations=1000,
    ):
        self.tensor_dtypes = tensor_dtypes
        self.tolerance_map = tolerance_map
        self.debug = debug
        self.bench = bench
        self.num_prerun = num_prerun
        self.num_iterations = num_iterations


class TestRunner:
    """Test runner"""

    def __init__(self, test_cases, test_config):
        self.test_cases = test_cases
        self.config = test_config
        self.failed_tests = []  # Track failures

    def run_tests(self, devices, test_func):
        """Run tests and track failures"""
        for device in devices:
            print(f"\n{'='*60}")
            print(f"Testing on {InfiniDeviceNames[device]}")
            print(f"{'='*60}")

            # filter unsupported data types
            tensor_dtypes = self._filter_tensor_dtypes_by_device(
                device, self.config.tensor_dtypes
            )

            for test_case in self.test_cases:
                for dtype in tensor_dtypes:
                    try:
                        test_func(device, test_case, dtype, self.config)
                        print(f"✓ {test_case} with {dtype} passed")
                    except Exception as e:
                        error_msg = f"{test_case} with {dtype} on {InfiniDeviceNames[device]}: {e}"
                        print(f"✗ {error_msg}")
                        self.failed_tests.append(error_msg)
                        if self.config.debug:
                            raise

        # Return whether any tests failed
        return len(self.failed_tests) == 0

    def _filter_tensor_dtypes_by_device(self, device, tensor_dtypes):
        """Filter data types based on device"""
        if device in ():
            # Filter out unsupported data types on specified devices
            return [dt for dt in tensor_dtypes if dt != infinicore.bfloat16]
        else:
            return tensor_dtypes

    def print_summary(self):
        """Print test summary"""
        if self.failed_tests:
            print(f"\n\033[91m{len(self.failed_tests)} tests failed:\033[0m")
            for failure in self.failed_tests:
                print(f"  - {failure}")
            return False
        else:
            print("\n\033[92mAll tests passed!\033[0m")
            return True
