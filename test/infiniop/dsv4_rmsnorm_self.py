from _dsv4_common import (
    _RMS_CASES,
    _TENSOR_DTYPES,
    get_args,
    get_test_devices,
    test_operator,
    test_rms,
)

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_rms, _RMS_CASES, _TENSOR_DTYPES)
    print("\033[92mDSV4 rmsnorm_self Test passed!\033[0m")
