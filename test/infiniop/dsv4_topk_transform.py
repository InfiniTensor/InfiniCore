from _dsv4_common import (
    _TOPK_CASES,
    InfiniDtype,
    get_args,
    get_test_devices,
    test_operator,
    test_topk,
)

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_topk, _TOPK_CASES, [InfiniDtype.F32])
    print("\033[92mDSV4 topk_transform Test passed!\033[0m")
