import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from infinicore.ops.random_sample import random_sample as ic_random_sample
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


def parse_test_cases(data):
    voc, random_val, topp, topk, temperature = data

    inputs = []
    # logits: will be set in prepare_inputs to match infiniop pattern
    # Use RANDOM as placeholder, will be replaced
    inputs.append(TensorSpec.from_tensor((voc,)))

    # output: scalar int32 (required by backend), use zeros init to avoid torch.rand(int) error
    output = TensorSpec.from_tensor(
        (), dtype=infinicore.int32, init_mode=TensorInitializer.ZEROS
    )
    return TestCase(
        TestCase.BOTH,
        inputs,
        output,
        voc=voc,
        random_val=random_val,
        topp=topp,
        topk=topk,
        temperature=temperature,
    )


_TEST_CASES = [parse_test_cases(d) for d in _TEST_CASES_DATA]

# Data types - note: infiniop random_sample supports F16/BF16/F32/F64 for logits
# But NVIDIA backend may have restrictions, adjust based on actual device support
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.bfloat16: {"atol": 0, "rtol": 0},
}


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
            idx = indices[0] if indices.numel() > 0 else torch.tensor(len(cum_probs) - 1, device=cum_probs.device)
        return sorted_indices[idx]

    return torch.argmax(data)


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("RandomSample")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def prepare_inputs(self, test_case, device, dtype_config):
        """Create logits matching infiniop test pattern: torch.arange(voc)[_perm].float() * 0.0001"""
        inputs, kwargs = super().prepare_inputs(test_case, device, dtype_config)
        
        voc = test_case.kwargs["voc"]
        from framework.devices import torch_device_map
        if device not in torch_device_map:
            raise ValueError(f"Unsupported device: {device}")
        torch_device = torch.device(torch_device_map[device])
        
        # Get dtype for logits
        if isinstance(dtype_config, dict) and "input_0" in dtype_config:
            tensor_dtype = dtype_config["input_0"]
        else:
            tensor_dtype = dtype_config if not isinstance(dtype_config, (list, tuple)) else dtype_config[0]
        
        from framework.datatypes import to_torch_dtype
        torch_dtype = to_torch_dtype(tensor_dtype)
        
        # Match infiniop test: torch.arange(voc)[_perm].float() * 0.0001
        _perm = torch.randperm(voc, device=torch_device)
        inputs[0] = (torch.arange(voc, dtype=torch.float32, device=torch_device)[_perm] * 0.0001).to(torch_dtype)
        
        return inputs, kwargs

    def torch_operator(self, logits, out=None, **kwargs):
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
        if out is None:
            return ic_random_sample(
                logits,
                kwargs["random_val"],
                kwargs["topp"],
                kwargs["topk"],
                kwargs["temperature"],
            )
        return ic_random_sample(
            logits,
            kwargs["random_val"],
            kwargs["topp"],
            kwargs["topk"],
            kwargs["temperature"],
            out=out,
        )

    def _run_single_test(self, device, test_case, dtype_config, config, mode_name):
        """Override to add fallback comparison: indices match OR logits values match (matches infiniop test)"""
        from framework.utils import infinicore_tensor_from_torch, convert_infinicore_to_torch
        
        # Store logits for fallback comparison
        inputs, kwargs = self.prepare_inputs(test_case, device, dtype_config)
        logits_tensor = inputs[0]
        
        # Try parent comparison first
        try:
            super()._run_single_test(device, test_case, dtype_config, config, mode_name)
            return  # Success with normal comparison
        except AssertionError:
            # Fallback: check if logits values match when indices differ
            infini_inputs = [infinicore_tensor_from_torch(inp) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
            
            if test_case.operation_mode == TestCase.OUT_OF_PLACE:
                torch_result = self.torch_operator(*inputs, **kwargs)
                infini_result = self.infinicore_operator(*infini_inputs, **kwargs)
                torch_result_from_infini = convert_infinicore_to_torch(infini_result, torch_result)
                ic_idx = torch_result_from_infini.item()
                ref_idx = torch_result.item()
            else:  # IN_PLACE - need to manually handle
                from framework.tensor import TensorSpec
                from framework.devices import torch_device_map
                from framework.datatypes import to_torch_dtype
                
                output_dtype = self.get_output_dtype(test_case, dtype_config)
                if test_case.output.is_contiguous or test_case.output.strides is None:
                    output_spec = TensorSpec.from_tensor(test_case.output.shape, output_dtype, init_mode=test_case.output.init_mode)
                else:
                    output_spec = TensorSpec.from_strided_tensor(test_case.output.shape, test_case.output.strides, output_dtype, init_mode=test_case.output.init_mode)
                
                torch_output = output_spec.create_torch_tensor(device, output_dtype)
                if not test_case.output.is_contiguous and test_case.output.strides is not None:
                    torch_output.zero_()
                
                torch_output_ref = torch_output.clone()
                self.torch_operator(*inputs, out=torch_output_ref, **kwargs)
                
                torch_dummy = torch.zeros(test_case.output.shape, dtype=to_torch_dtype(output_dtype), device=torch_device_map[device])
                if not test_case.output.is_contiguous and test_case.output.strides is not None:
                    from framework.utils import rearrange_tensor
                    rearrange_tensor(torch_dummy, list(torch_output.stride()))
                infini_output = infinicore_tensor_from_torch(torch_dummy)
                self.infinicore_operator(*infini_inputs, out=infini_output, **kwargs)
                
                torch_result_from_infini = convert_infinicore_to_torch(infini_output, torch_output)
                ic_idx = torch_result_from_infini.item()
                ref_idx = torch_output_ref.item()
            
            # Fallback comparison: indices match OR logits values match
            if ic_idx != ref_idx and logits_tensor[ic_idx] != logits_tensor[ref_idx]:
                raise AssertionError(f"RandomSample {mode_name}: indices differ ({ic_idx} vs {ref_idx}) and logits values differ")


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
