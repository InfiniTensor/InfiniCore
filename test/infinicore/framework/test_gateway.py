import sys
import os
import json
import importlib
import inspect
import argparse
from typing import Any, Optional, Tuple, Union, Dict

import infinicore
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from framework.base import BaseOperatorTest, TestCase, TensorSpec
from framework.config import get_args
from framework.runner import GenericTestRunner

class TestExecutionGateway:
    """
    Test Execution Gateway
    """

    SUPPORTED_HARDWARE_FLAGS = [
        "cpu", "nvidia", "cambricon", "ascend", "iluvatar", 
        "metax", "moore", "kunlun", "hygon", "qy"
    ]

    def run(self, json_file_path: str, config: Union[str, Dict[str, Any], argparse.Namespace, None] = None) -> Any:
        print(f"🚀 Gateway: Start processing...")

        if not json_file_path or not os.path.exists(json_file_path):
            raise FileNotFoundError(f"❌ JSON file not found: {json_file_path}")

        # Normalize Config Override
        override_dict = self._normalize_override_config(config)

        print(f"📄 Source: Loading {json_file_path}")
        try:
            op_name, test_case, final_args, op_funcs, op_paths = self._load(json_file_path, override_config=override_dict)
        except Exception as e:
            import traceback; traceback.print_exc()
            raise RuntimeError(f"❌ Failed to load configuration: {e}") from e

        # Identify active devices for cleaner logging
        active_devices = [hw.upper() for hw in self.SUPPORTED_HARDWARE_FLAGS if getattr(final_args, hw, False)]
        device_str = ", ".join(active_devices) if active_devices else "None"

        print(f"⚙️  Ready to execute operator: '{op_name}'")
        print(f"    Targets: Torch -> {op_paths['torch']}")
        print(f"             Infini -> {op_paths['infinicore']}")
        print(f"    Target Devices: {device_str}")
        print(f"    Config: Bench={final_args.bench}, Prerun={final_args.num_prerun}, Iterations={final_args.num_iterations}")
        print(f"    Description: {test_case.description}")
        
        results = self._execute_tests(op_name, test_case, final_args, op_funcs)

        print(f"🏁 Gateway: Process finished.")
        return results

    def _normalize_override_config(self, config):
        override_dict = {}
        if config:
            if isinstance(config, str):
                if os.path.exists(config):
                    with open(config, 'r') as f: override_dict = json.load(f)
                else: raise FileNotFoundError(f"❌ Config file not found: {config}")
            elif isinstance(config, argparse.Namespace): override_dict = vars(config)
            elif isinstance(config, dict): override_dict = config
            else: raise ValueError("❌ Config must be file path, dict, or Namespace")
        return override_dict

    def _load(
        self, 
        json_file_path: str, 
        override_config: Dict[str, Any]
        ) -> Tuple[str, TestCase, argparse.Namespace, Dict[str, Any], Dict[str, str]]:
        # --- A. Read JSON ---
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format: {json_file_path}")

        # --- B. Extract Operator Info ---
        op_name = data.get("operator")
        if not op_name:
             raise ValueError("JSON missing required 'operator' field.")
        
        # Load actual functions from strings
        torch_op_str = data.get("torch_op")
        infini_op_str = data.get("infinicore_op")

        if not torch_op_str or not infini_op_str:
            raise ValueError("JSON must specify 'torch_op' and 'infinicore_op' function paths (e.g., 'torch.add').")

        op_funcs = {
            "torch": self._load_function(torch_op_str),
            "infinicore": self._load_function(infini_op_str)
        }

        op_paths = {
            "torch": torch_op_str,
            "infinicore": infini_op_str
        }

        # --- C. Construct Args ---
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]
        args = get_args()
        sys.argv = original_argv

        json_args = data.get("args", {})
        for key, value in json_args.items():
            if hasattr(args, key):
                setattr(args, key, value)

        target_device = data.get("device", "cpu").lower()
        self._set_device_flags(args, target_device)

        # --- D. Construct TestCase ---
        test_case = self._build_test_case(data, op_name)

        # --- E. Apply Override ---
        if override_config:
            for key, value in override_config.items():
                if value is not None:
                    setattr(args, key, value)
            if 'device' in override_config and override_config['device']:
                 self._set_device_flags(args, override_config['device'])

        return op_name, test_case, args, op_funcs, op_paths

    def _execute_tests(
        self, 
        op_name: str, 
        test_case: TestCase, 
        args: argparse.Namespace, 
        op_funcs: Dict[str, Any]):
        """
        Constructs a DynamicOpTest class on the fly and runs it.
        """
        cases_to_run = [test_case]
        
        torch_func = op_funcs["torch"]
        infini_func = op_funcs["infinicore"]

        class DynamicOpTest(BaseOperatorTest):
            def __init__(self):
                super().__init__(op_name)
            
            def get_test_cases(self):
                return cases_to_run

            def torch_operator(self, *args, **kwargs):
                return torch_func(*args, **kwargs)

            def infinicore_operator(self, *args, **kwargs):
                return infini_func(*args, **kwargs)

        runner = GenericTestRunner(DynamicOpTest, args)
        try: runner.run_and_exit()
        except SystemExit: pass

        return getattr(runner, "test_results", "Done")

    def _load_function(self, func_path: str) -> Any:
        """
        Dynamically imports a module and retrieves a function object.
        
        Supports:
          - "torch.add" -> module: torch, func: add
          - "torch.nn.functional.adaptive_max_pool1d" -> module: torch.nn.functional, func: adaptive_max_pool1d
        """
        if "." not in func_path:
            raise ValueError(f"Invalid function path: '{func_path}'. Must be 'module.function'.")
        
        # Split from the right to separate module path and function name
        module_name, func_name = func_path.rsplit(".", 1)
        
        try:
            # Attempt to import the module part
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(f"❌ Could not import module '{module_name}' for function '{func_path}': {e}")
        
        try:
            # Retrieve the function from the imported module
            func = getattr(module, func_name)
        except AttributeError:
            raise AttributeError(f"❌ Module '{module_name}' has no function named '{func_name}'")
            
        return func

    def _set_device_flags(self, args, target_device_str):
        
        for flag in self.SUPPORTED_HARDWARE_FLAGS:
            if hasattr(args, flag): setattr(args, flag, False)

        d = target_device_str.lower()
        if "cpu" in d: args.cpu = True
        elif "cuda" in d or "nvidia" in d: args.nvidia = True
        elif "npu" in d or "ascend" in d: args.ascend = True
        elif "mlu" in d or "cambricon" in d: args.cambricon = True
        elif "iluvatar" in d: args.iluvatar = True
        elif "metax" in d or "maca" in d: args.metax = True
        elif "musa" in d or "moore" in d: args.moore = True
        elif "xpu" in d or "kunlun" in d: args.kunlun = True
        elif "dcu" in d or "hygon" in d: args.hygon = True
        elif "qy" in d: args.qy = True
        else:
            print(f"⚠️ Unknown device '{d}'. Fallback to CPU.")
            args.cpu = True

    def _build_test_case(self, data, op_name):
        # 1. Parse Inputs
        inputs_list = []
        raw_inputs = data.get("inputs", [])
        for idx, inp in enumerate(raw_inputs):
            spec = self._parse_spec_from_dict(inp, f"in_{idx}")
            inputs_list.append(spec)

        # 2. Parse Kwargs
        raw_kwargs = data.get("kwargs", {})
        kwargs = {}
        for k, v in raw_kwargs.items():
            if isinstance(v, dict) and "shape" in v and "dtype" in v:
                kwargs[k] = self._parse_spec_from_dict(v, default_name=k)
            else:
                kwargs[k] = v

        # 3. Parse Output Spec
        output_spec = None
        raw_out_spec = data.get("output_spec")
        if raw_out_spec:
            output_spec = self._parse_spec_from_dict(raw_out_spec, "out_0")

        # 4. Parse Output Specs (Multiple)
        output_specs = None
        raw_out_specs = data.get("output_specs")
        if raw_out_specs:
            output_specs = []
            for idx, spec_data in enumerate(raw_out_specs):
                spec = self._parse_spec_from_dict(spec_data, f"out_{idx}")
                output_specs.append(spec)

        # 5. Determine output count
        output_count = 1
        if output_specs:
            output_count = len(output_specs)
        elif "output_count" in data:
            output_count = int(data["output_count"])
            
        comparison_target = data.get("comparison_target")
        tolerance = data.get("tolerance", {"atol": 1e-3, "rtol": 1e-3})
        description = data.get("description", f"Auto-test {op_name}")

        return TestCase(
            inputs=inputs_list,
            kwargs=kwargs,
            output_spec=output_spec,
            output_specs=output_specs,
            comparison_target=comparison_target,
            tolerance=tolerance,
            description=description,
            output_count=output_count
        )

    def _parse_spec_from_dict(self, spec_dict: Dict, default_name: str):
        """Helper to create TensorSpec from a dictionary definition"""
        return TensorSpec.from_tensor(
            shape=tuple(spec_dict["shape"]),
            strides=tuple(spec_dict["strides"]) if spec_dict.get("strides") else None,
            dtype=self._parse_dtype(spec_dict.get("dtype", "float32")),
            name=spec_dict.get("name", default_name)
        )

    def _parse_dtype(self, dtype_str: str):
        dtype_map = {
            "float16": infinicore.float16, "float32": infinicore.float32,
            "bfloat16": infinicore.bfloat16, "int32": infinicore.int32,
            "int64": infinicore.int64, "bool": infinicore.bool,
        }
        return dtype_map.get(dtype_str, infinicore.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Gateway")
    parser.add_argument("file_path", type=str, help="Path to JSON config file")
    
    # Overrides
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--bench", type=str, choices=["host", "device", "both"], default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--num_prerun", type=int, default=None)
    parser.add_argument("--num_iterations", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    
    gateway_args = parser.parse_args()

    override_dict = {
        k: v for k, v in vars(gateway_args).items()
        if k != "file_path" and v is not None and v is not False
    }

    gateway = TestExecutionGateway()
    gateway.run(json_file_path=gateway_args.file_path, config=override_dict)
