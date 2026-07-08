from _dsv4_common import (
    _ROPE_CASES,
    _TENSOR_DTYPES,
    get_args,
    get_test_devices,
    test_operator,
    test_rope,
)

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_rope, _ROPE_CASES, _TENSOR_DTYPES)
    print("\033[92mDSV4 fused_rope Test passed!\033[0m")
