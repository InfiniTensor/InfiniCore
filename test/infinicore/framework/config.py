import argparse
from .devices import InfiniDeviceEnum


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Operator")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Whether to benchmark performance",
    )
    parser.add_argument(
        "--num_prerun",
        type=lambda x: max(0, int(x)),
        default=10,
        help="Set the number of pre-runs before benchmarking. Default is 10.",
    )
    parser.add_argument(
        "--num_iterations",
        type=lambda x: max(0, int(x)),
        default=1000,
        help="Set the number of iterations for benchmarking. Default is 1000.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to turn on debug mode.",
    )

    # Device options
    device_group = parser.add_argument_group("Device options")
    device_group.add_argument("--cpu", action="store_true", help="Run CPU test")
    device_group.add_argument(
        "--nvidia", action="store_true", help="Run NVIDIA GPU test"
    )
    device_group.add_argument(
        "--cambricon", action="store_true", help="Run Cambricon MLU test"
    )
    device_group.add_argument(
        "--ascend", action="store_true", help="Run ASCEND NPU test"
    )
    device_group.add_argument(
        "--iluvatar", action="store_true", help="Run Iluvatar GPU test"
    )
    device_group.add_argument("--metax", action="store_true", help="Run METAX GPU test")
    device_group.add_argument(
        "--moore", action="store_true", help="Run MTHREADS GPU test"
    )
    device_group.add_argument(
        "--kunlun", action="store_true", help="Run KUNLUN XPU test"
    )

    return parser.parse_args()


def get_test_devices(args):
    """
    Determine which devices to test based on command line arguments
    """
    devices_to_test = []

    if args.cpu:
        devices_to_test.append(InfiniDeviceEnum.CPU)
    if args.nvidia:
        devices_to_test.append(InfiniDeviceEnum.NVIDIA)
    if args.iluvatar:
        devices_to_test.append(InfiniDeviceEnum.ILUVATAR)
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
        import torch

        devices_to_test.append(InfiniDeviceEnum.METAX)
    if args.moore:
        try:
            import torch
            import torch_musa

            devices_to_test.append(InfiniDeviceEnum.MOORE)
        except ImportError:
            print("Warning: torch_musa not available, skipping Moore tests")
    if args.kunlun:
        try:
            import torch_xmlir

            devices_to_test.append(InfiniDeviceEnum.KUNLUN)
        except ImportError:
            print("Warning: torch_xmlir not available, skipping Kunlun tests")

    # Default to CPU if no devices specified
    if not devices_to_test:
        devices_to_test = [InfiniDeviceEnum.CPU]

    return devices_to_test
