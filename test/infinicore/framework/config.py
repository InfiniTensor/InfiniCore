import argparse
from .devices import InfiniDeviceEnum

"""
Shared hardware platform information for the InfiniCore testing framework
"""


def get_supported_hardware_platforms():
    """
    Get list of supported hardware platforms with descriptions.

    Returns:
        List of tuples (flag, description)
    """
    return [
        ("--cpu", "Standard CPU execution"),
        ("--nvidia", "NVIDIA GPUs with CUDA support"),
        ("--cambricon", "Cambricon MLU accelerators (requires torch_mlu)"),
        ("--ascend", "Huawei Ascend NPUs (requires torch_npu)"),
        ("--iluvatar", "Iluvatar GPUs"),
        ("--metax", "Metax GPUs"),
        ("--moore", "Moore Threads GPUs (requires torch_musa)"),
        ("--kunlun", "Kunlun XPUs (requires torch_xmlir)"),
        ("--hygon", "Hygon DCUs"),
    ]


def get_hardware_args_group(parser):
    """
    Add hardware platform arguments to an argument parser.

    Args:
        parser: argparse.ArgumentParser instance

    Returns:
        The argument group for hardware platforms
    """
    hardware_group = parser.add_argument_group("Hardware Platform Options")

    for flag, description in get_supported_hardware_platforms():
        hardware_group.add_argument(flag, action="store_true", help=description)

    return hardware_group


def get_args():
    """Parse command line arguments for operator testing"""
    parser = argparse.ArgumentParser(
        description="Test InfiniCore operators across multiple hardware platforms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run all tests on CPU only
  python test_operator.py --cpu

  # Run with benchmarking on NVIDIA GPU
  python test_operator.py --nvidia --bench

  # Run with debug mode on multiple devices
  python test_operator.py --cpu --nvidia --debug

  # Run with verbose mode to stop on first error with full traceback
  python test_operator.py --cpu --nvidia --verbose

  # Run performance profiling with custom iterations
  python test_operator.py --nvidia --bench --num_prerun 50 --num_iterations 5000

        """,
    )

    # Core testing options
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Enable performance benchmarking mode",
    )
    parser.add_argument(
        "--num_prerun",
        type=lambda x: max(0, int(x)),
        default=10,
        help="Number of warm-up runs before benchmarking (default: 10)",
    )
    parser.add_argument(
        "--num_iterations",
        type=lambda x: max(0, int(x)),
        default=1000,
        help="Number of iterations for benchmarking (default: 1000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for detailed tensor comparison",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode to stop on first error with full traceback",
    )

    # Device options using shared hardware info
    hardware_group = get_hardware_args_group(parser)
    args, unknown = parser.parse_known_args()

    return args


def get_test_devices(args):
    """
    Determine which devices to test based on command line arguments

    Returns:
        List[InfiniDeviceEnum]: List of devices to test
    """
    devices_to_test = []

    # Check each hardware platform with proper dependency validation
    if args.cpu:
        devices_to_test.append(InfiniDeviceEnum.CPU)

    if args.nvidia:
        try:
            import torch.cuda

            devices_to_test.append(InfiniDeviceEnum.NVIDIA)
        except ImportError:
            print("Warning: CUDA not available, skipping NVIDIA tests")

    if args.cambricon:
        try:
            import torch_mlu

            devices_to_test.append(InfiniDeviceEnum.CAMBRICON)
        except ImportError:
            print("Warning: torch_mlu not available, skipping Cambricon tests")

    if args.ascend:
        try:
            import torch
            import torch_npu

            torch.npu.set_device(0)  # Ascend NPU needs explicit device initialization
            devices_to_test.append(InfiniDeviceEnum.ASCEND)
        except ImportError:
            print("Warning: torch_npu not available, skipping Ascend tests")

    if args.metax:
        try:
            import torch

            devices_to_test.append(InfiniDeviceEnum.METAX)
        except ImportError:
            print("Warning: Metax GPU support not available")

    if args.moore:
        try:
            import torch
            import torch_musa

            devices_to_test.append(InfiniDeviceEnum.MOORE)
        except ImportError:
            print("Warning: torch_musa not available, skipping Moore tests")

    if args.iluvatar:
        try:
            # Iluvatar GPU detection
            import torch

            devices_to_test.append(InfiniDeviceEnum.ILUVATAR)
        except ImportError:
            print("Warning: Iluvatar GPU support not available")

    if args.kunlun:
        try:
            import torch_xmlir

            devices_to_test.append(InfiniDeviceEnum.KUNLUN)
        except ImportError:
            print("Warning: torch_xmlir not available, skipping Kunlun tests")

    if args.hygon:
        try:
            import torch

            devices_to_test.append(InfiniDeviceEnum.HYGON)
        except ImportError:
            print("Warning: Hygon DCU support not available")

    # Default to CPU if no devices specified
    if not devices_to_test:
        devices_to_test = [InfiniDeviceEnum.CPU]
        print("No devices specified, defaulting to CPU")

    return devices_to_test
