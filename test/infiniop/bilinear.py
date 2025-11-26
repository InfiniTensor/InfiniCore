import torch
import ctypes
from ctypes import c_uint64

from libinfiniop import (
	LIBINFINIOP,
	TestTensor,
	get_test_devices,
	check_error,
	test_operator,
	get_args,
	debug,
	get_tolerance,
	profile_operation,
	TestWorkspace,
	InfiniDtype,
	InfiniDtypeNames,
	InfiniDeviceNames,
	infiniopOperatorDescriptor_t,
)


_TEST_CASES = [
	# batch, in1, in2, out, use_bias
	(4, 3, 5, 2, True),
	(1, 6, 7, 3, True),
	(8, 2, 4, 5, False),
	(2, 3, 3, 4, True),
	(6, 10, 12, 7, False),
	(3, 1, 1, 2, True),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
	InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
	InfiniDtype.BF16: {"atol": 1e-2, "rtol": 5e-2},
	InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-4},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def reference_bilinear(x1, x2, weight, bias):
	"""Compute bilinear output in FP32 for stability and cast back."""
	out = torch.einsum(
		"ni,oij,nj->no",
		x1.to(torch.float32),
		weight.to(torch.float32),
		x2.to(torch.float32),
	)
	if bias is not None:
		out = out + bias.to(torch.float32)
	return out.to(x1.dtype)


def test(
	handle,
	device,
	batch,
	in1_features,
	in2_features,
	out_features,
	use_bias,
	dtype=InfiniDtype.F16,
	sync=None,
):
	print(
		f"Testing Bilinear on {InfiniDeviceNames[device]} with N:{batch} in1:{in1_features} in2:{in2_features} "
		f"out:{out_features} bias:{use_bias} dtype:{InfiniDtypeNames[dtype]}"
	)

	out_tensor = TestTensor((batch, out_features), None, dtype, device, mode="zeros")
	x1 = TestTensor((batch, in1_features), None, dtype, device, scale=0.1, bias=-0.05)
	x2 = TestTensor((batch, in2_features), None, dtype, device, scale=0.1, bias=-0.05)
	weight = TestTensor(
		(out_features, in1_features, in2_features),
		None,
		dtype,
		device,
		scale=0.1,
		bias=-0.05,
	)
	bias_tensor = (
		TestTensor((out_features,), None, dtype, device, scale=0.1, bias=-0.05)
		if use_bias
		else None
	)

	ref = reference_bilinear(
		x1.torch_tensor(),
		x2.torch_tensor(),
		weight.torch_tensor(),
		bias_tensor.torch_tensor() if bias_tensor else None,
	)

	if sync is not None:
		sync()

	descriptor = infiniopOperatorDescriptor_t()
	check_error(
		LIBINFINIOP.infiniopCreateBilinearDescriptor(
			handle,
			ctypes.byref(descriptor),
			out_tensor.descriptor,
			x1.descriptor,
			x2.descriptor,
			weight.descriptor,
			bias_tensor.descriptor if bias_tensor else None,
		)
	)

	tensors = [out_tensor, x1, x2, weight]
	if bias_tensor:
		tensors.append(bias_tensor)
	for tensor in tensors:
		tensor.destroy_desc()

	workspace_size = c_uint64(0)
	check_error(
		LIBINFINIOP.infiniopGetBilinearWorkspaceSize(
			descriptor, ctypes.byref(workspace_size)
		)
	)
	workspace = TestWorkspace(workspace_size.value, device)

	def lib_bilinear():
		check_error(
			LIBINFINIOP.infiniopBilinear(
				descriptor,
				workspace.data(),
				workspace_size.value,
				out_tensor.data(),
				x1.data(),
				x2.data(),
				weight.data(),
				bias_tensor.data() if bias_tensor else None,
				None,
			)
		)
	print(f"workspace size: {workspace_size.value} bytes")
	print(f"tensor shapes: x1{tuple(x1.shape)} x2{tuple(x2.shape)} weight{tuple(weight.shape)} bias{tuple(bias_tensor.shape) if bias_tensor else None} out{tuple(out_tensor.shape)}")
	lib_bilinear()
	print(f"new workspace size: {workspace_size.value} bytes")
	print(f"new tensor shapes: x1{tuple(x1.shape)} x2{tuple(x2.shape)} weight{tuple(weight.shape)} bias{tuple(bias_tensor.shape) if bias_tensor else None} out{tuple(out_tensor.shape)}")
	print(f"--")
 
	atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
	if DEBUG:
		debug(out_tensor.actual_tensor(), ref, atol=atol, rtol=rtol)
	assert torch.allclose(out_tensor.actual_tensor(), ref, atol=atol, rtol=rtol)

	if PROFILE:
		profile_operation(
			"PyTorch",
			lambda: reference_bilinear(
				x1.torch_tensor(),
				x2.torch_tensor(),
				weight.torch_tensor(),
				bias_tensor.torch_tensor() if bias_tensor else None,
			),
			device,
			NUM_PRERUN,
			NUM_ITERATIONS,
		)
		profile_operation("    lib", lambda: lib_bilinear(), device, NUM_PRERUN, NUM_ITERATIONS)

	check_error(LIBINFINIOP.infiniopDestroyBilinearDescriptor(descriptor))


if __name__ == "__main__":
	args = get_args()

	DEBUG = args.debug
	PROFILE = args.profile
	NUM_PRERUN = args.num_prerun
	NUM_ITERATIONS = args.num_iterations

	for device in get_test_devices(args):
		test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

	print("\033[92mTest passed!\033[0m")
