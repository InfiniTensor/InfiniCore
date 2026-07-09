from _dsv4_common import (
    _QUANT_CASES,
    _TENSOR_DTYPES,
    get_args,
    get_test_devices,
    test_operator,
    test_quant,
)

if __name__ == "__main__":
    args = get_args()
    for device in get_test_devices(args):
        test_operator(device, test_quant, _QUANT_CASES, _TENSOR_DTYPES)
    print("\033[92mDSV4 per_token_quant_int8 Test passed!\033[0m")
