import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
import infinicore.nn.functional as F
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.tensor import TensorInitializer

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases: (voc, random_val, topp, topk, temperature)
# Aligned with test/infiniop/random_sample.py
_TEST_CASES_DATA = [
    (512, 0.8, 0.8, 3, 0.5),
    (4096, 0.05, 0.9, 5, 1.0),
    (16384, 0.15, 0.85, 10, 2.0),
    (512, 0.08, 0.0, 3, 0.5),
    (4096, 0.5, 0.9, 1, 1.0),
    (16384, 0.15, 0.0, 1, 2.0),  # Duplicate as in infiniop test
    (32000, 0.08, 0.8, 50, 1.0),
    (32000, 0.08, 1.0, 25, 1.0),
    # (119696, 0.01, 1.0, 100, 1.0),  # Commented out in infiniop test
]

# Data types - note: infiniop random_sample supports F16/BF16/F32/F64 for logits
# But NVIDIA backend may have restrictions, adjust based on actual device support
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.bfloat16: {"atol": 0, "rtol": 0},
}


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for all operation types.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        voc, random_val, topp, topk, temperature = data
        base_kwargs = {
            "voc": voc,
            "random_val": random_val,
            "topp": topp,
            "topk": topk,
            "temperature": temperature,
        }

        for tensor_dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(tensor_dtype, {"atol": 0, "rtol": 0})

            # Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[TensorSpec.from_tensor((voc,), dtype=tensor_dtype)],
                    kwargs=base_kwargs.copy(),
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"RandomSample - OUT_OF_PLACE",
                )
            )

            # With explicit output tensor
            test_cases.append(
                TestCase(
                    inputs=[TensorSpec.from_tensor((voc,), dtype=tensor_dtype)],
                    kwargs=base_kwargs.copy(),
                    output_spec=TensorSpec.from_tensor(
                        (), dtype=infinicore.int32, init_mode=TensorInitializer.ZEROS
                    ),
                    comparison_target="out",
                    tolerance=tolerance,
                    description=f"RandomSample - OUT",
                )
            )

    return test_cases


def torch_random_sample(data, random_val, topp, topk, voc, temperature):
    if topp > 0 and topk > 1:
        sorted_vals, sorted_indices = torch.sort(data, descending=True)

        scaled_vals = (sorted_vals - sorted_vals[0]) / temperature
        try:
            probs = torch.softmax(scaled_vals, dim=0)
        except RuntimeError as e:
            if "not implemented for 'Half'" in str(e):
                scaled_vals = scaled_vals.to(torch.float32)
                probs = torch.softmax(scaled_vals, dim=0)
            else:
                raise
        cum_probs = torch.cumsum(probs, dim=0)

        k_index = min(topk, voc) - 1
        threshold = min(cum_probs[k_index], topp) * random_val

        try:
            idx = torch.searchsorted(cum_probs, threshold)
        except Exception:
            indices = (cum_probs >= threshold).nonzero(as_tuple=True)[0]
            idx = (
                indices[0]
                if indices.numel() > 0
                else torch.tensor(len(cum_probs) - 1, device=cum_probs.device)
            )
        return sorted_indices[idx]

    return torch.argmax(data)


class OpTest(BaseOperatorTest):
    """RandomSample operator test with simplified implementation"""

    def __init__(self):
        super().__init__("RandomSample")
        self._current_logits = None  # Store logits for special comparison

    def get_test_cases(self):
        return parse_test_cases()

    def prepare_inputs_and_kwargs(self, test_case, device):
        """Prepare inputs and kwargs, replacing TensorSpec objects with actual tensors"""
        inputs, kwargs = super().prepare_inputs_and_kwargs(test_case, device)

        # If we already have stored logits (from a previous call), reuse them
        # to ensure consistency across multiple calls for the same test case
        if self._current_logits is not None:
            inputs[0] = self._current_logits
            return inputs, kwargs

        voc = kwargs["voc"]
        from framework.devices import torch_device_map

        if device not in torch_device_map:
            raise ValueError(f"Unsupported device: {device}")
        torch_device = torch.device(torch_device_map[device])

        tensor_dtype = inputs[0].dtype

        # Match infiniop test: torch.arange(voc)[_perm].float() * 0.0001
        _perm = torch.randperm(voc, device=torch_device)
        logits = (
            torch.arange(voc, dtype=torch.float32, device=torch_device)[_perm] * 0.0001
        ).to(tensor_dtype)
        inputs[0] = logits
        self._current_logits = logits  # Store for special comparison

        return inputs, kwargs

    def torch_operator(self, logits, out=None, **kwargs):
        """PyTorch random_sample implementation"""
        idx = torch_random_sample(
            logits,
            kwargs["random_val"],
            kwargs["topp"],
            kwargs["topk"],
            kwargs["voc"],
            kwargs["temperature"],
        ).to(torch.int32)
        if out is None:
            return idx
        out.copy_(idx)
        return out

    def infinicore_operator(self, logits, out=None, **kwargs):
        """InfiniCore random_sample implementation"""
        if out is None:
            return F.random_sample(
                logits,
                kwargs["random_val"],
                kwargs["topp"],
                kwargs["topk"],
                kwargs["temperature"],
            )
        return F.random_sample(
            logits,
            kwargs["random_val"],
            kwargs["topp"],
            kwargs["topk"],
            kwargs["temperature"],
            out=out,
        )

    def run_test(self, device, test_case, config):
        """
        Override run_test to handle random_sample's special comparison logic.

        For random_sample, if the indices differ but the logits values at those
        indices are equal, the result is still considered valid. This handles
        cases where multiple valid indices exist due to floating-point precision.

        This is necessary because random_sample can return different valid indices
        when multiple positions have the same logits value, especially with
        low-precision types like bfloat16 due to floating-point rounding.
        """
        # Clear stored logits before test to ensure fresh generation
        self._current_logits = None

        try:
            # Try the standard comparison first
            # This will call prepare_inputs_and_kwargs which will set self._current_logits
            return super().run_test(device, test_case, config)
        except AssertionError as original_error:
            # If standard comparison fails, check if this is a valid case where
            # indices differ but logits values are equal

            # Only handle if we have stored logits (from prepare_inputs_and_kwargs)
            if self._current_logits is None:
                raise

            logits_tensor = self._current_logits

            # Re-run operations with the same logits to get results for comparison
            # prepare_inputs_and_kwargs will reuse self._current_logits if it exists
            from framework.utils import (
                infinicore_tensor_from_torch,
                convert_infinicore_to_torch,
            )

            inputs, kwargs = self.prepare_inputs_and_kwargs(test_case, device)

            # Prepare infinicore inputs
            infini_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    cloned_inp = inp.clone().detach()
                    infini_tensor = infinicore_tensor_from_torch(cloned_inp)
                    infini_inputs.append(infini_tensor)
                else:
                    infini_inputs.append(inp)

            infini_kwargs = kwargs.copy()
            if "out" in infini_kwargs and isinstance(
                infini_kwargs["out"], torch.Tensor
            ):
                cloned_out = infini_kwargs["out"].clone().detach()
                infini_kwargs["out"] = infinicore_tensor_from_torch(cloned_out)

            # Run both operators
            torch_result = self.torch_operator(*inputs, **kwargs)
            infini_result = self.infinicore_operator(*infini_inputs, **infini_kwargs)

            # Extract indices from results
            comparison_target = test_case.comparison_target
            if comparison_target == "out":
                # Compare output tensor from kwargs
                ref_idx = kwargs["out"].item()
                torch_result_from_infini = convert_infinicore_to_torch(
                    infini_kwargs["out"]
                )
                ic_idx = torch_result_from_infini.item()
            else:
                # Compare return values
                ref_idx = torch_result.item()
                torch_result_from_infini = convert_infinicore_to_torch(infini_result)
                ic_idx = torch_result_from_infini.item()

            # Check if indices are equal (standard case)
            if ic_idx == ref_idx:
                return True, "passed"

            # Special case: indices differ but logits values are equal
            # This is valid for random_sample when multiple indices have the same logits value
            try:
                logits_ref = logits_tensor[ref_idx].item()
                logits_ic = logits_tensor[ic_idx].item()
                if logits_ic == logits_ref:
                    # Valid: different indices but same logits value
                    return True, "passed"
            except (IndexError, RuntimeError):
                # If we can't access the logits, fall through to raise the original error
                pass

            # If we get here, the results are truly different
            raise original_error


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
