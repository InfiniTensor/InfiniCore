from _dsv4_common import (
    _SILU_CASES,
    _TENSOR_DTYPES,
    get_args,
    get_test_devices,
    test_operator,
    test_silu,
)

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_silu, _SILU_CASES, _TENSOR_DTYPES)
    print("\033[92mDSV4 silu_and_mul Test passed!\033[0m")
