import sys
import os
import json
import importlib
import argparse
from typing import Any, Optional, Tuple, Union, Dict, List
from dataclasses import is_dataclass

import infinicore
import torch

# Path adaptation
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from framework.base import BaseOperatorTest, TestCase, TensorSpec
from framework.config import get_args, get_supported_hardware_platforms
from framework.runner import GenericTestRunner
from framework.devices import InfiniDeviceEnum


class TestCaseManager:
    """
    Test Case Manager (Strict Schema Version)
    """

    def __init__(self):
        # Load supported hardware flags for CLI args (Strings)
        self.supported_hw_flags = [
            item[0].lstrip("-") for item in get_supported_hardware_platforms()
        ]

    def run(
        self,
        json_file_path: Optional[str] = None,
        config: Union[str, Dict[str, Any], argparse.Namespace, None] = None,
        save_path: str = None,
    ) -> Any:
        print(f"🚀 Test Case Manager: Start processing...")
        override_dict = self._normalize_override_config(config)

        test_configs = []

        # 1. Load Configurations
        if json_file_path and os.path.exists(json_file_path):
            print(f"📄 Source: Loading {json_file_path}")
            test_configs = self._load(json_file_path, override_config=override_dict)
        else:
            # Fallback to default hardcoded case
            (
                op_name,
                test_cases,
                final_args,
                op_funcs,
                op_paths,
            ) = self._load_default_case(override_dict)

            test_configs.append(
                {
                    "op_name": op_name,
                    "test_cases": test_cases,
                    "args": final_args,
                    "op_funcs": op_funcs,
                    "op_paths": op_paths,
                    "target_device": "cpu",
                }
            )

        total_results = []

        # 2. Execute & Collect Results
        for idx, cfg in enumerate(test_configs):
            op_name = cfg["op_name"]
            test_cases = cfg["test_cases"]
            n_cases = len(test_cases)

            print(f"\n🔹 Config {idx + 1}/{len(test_configs)}: {op_name} ({n_cases} cases)")

            # Execute
            # results_list is a list of TestResult objects
            results_list = self._execute_tests(
                op_name, test_cases, cfg["args"], cfg["op_funcs"]
            )

            # Report
            entry = self._prepare_report_entry(
                op_name,
                test_cases,
                cfg["args"],
                cfg["op_paths"],
                cfg["target_device"],
                results_list,
            )
            total_results.append(entry)

        # 3. Save
        if save_path:
            self._save_all_results(save_path, total_results)

        return total_results

    def _load(self, json_file_path: str, override_config: Dict[str, Any]) -> List[Dict]:
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON: {json_file_path}")

        data_list = data if isinstance(data, list) else [data]
        configs = []

        for case_data in data_list:
            op_name = case_data.get("operator")
            if not op_name:
                continue

            torch_op = case_data.get("torch_op") or self._discover_op_path(
                op_name, 
                ["torch", "torch.nn.functional"]
            )
            infini_op = case_data.get("infinicore_op") or self._discover_op_path(
                op_name, 
                ["infinicore", "infinicore.nn.functional"]
            )

            # Load Functions
            op_funcs = {
                "torch": self._load_function(torch_op),
                "infinicore": self._load_function(infini_op),
            }
            op_paths = {
                "torch": torch_op,
                "infinicore": infini_op,
            }

            # Setup Args & Device
            case_args = self._get_default_args()
            self._merge_args(case_args, case_data.get("args", {}))

            if override_config:
                self._merge_args(case_args, override_config)

            # Determine Target Device
            if override_config and "device" in override_config:
                target_device = override_config["device"]
            else:
                target_device = case_data.get("device", "cpu")

            self._set_device_flags(case_args, target_device)

            # Build Cases (Strict Mode)
            test_cases = self._build_test_cases(case_data, op_name)

            configs.append(
                {
                    "op_name": op_name,
                    "test_cases": test_cases,
                    "args": case_args,
                    "op_funcs": op_funcs,
                    "op_paths": op_paths,
                    "target_device": target_device,
                }
            )
        return configs

    def _build_test_cases(self, data: Dict, op_name: str) -> List[TestCase]:
        """
        Parses 'cases' list from JSON. 
        """
        cases_data = data.get("cases")
        if not cases_data or not isinstance(cases_data, list):
            raise ValueError(f"❌ Config for '{op_name}' missing required 'cases' list.")

        base_desc = data.get("description", f"Auto-test {op_name}")
        base_tol = data.get("tolerance", {"atol": 1e-3, "rtol": 1e-3})
        base_cmp = data.get("comparison_target", None)

        test_cases_list = []

        for idx, sub in enumerate(cases_data):
            full_desc = f"{base_desc} - {sub.get('description', f'Case_{idx}')}"

            # Parse inputs
            raw_inputs = sub.get("inputs", [])
            inputs = [
                self._parse_spec(inp, f"in_{i}") for i, inp in enumerate(raw_inputs)
            ]

            # Parse kwargs
            kwargs = {}
            for k, v in sub.get("kwargs", {}).items():
                if isinstance(v, dict) and "shape" in v and "dtype" in v:
                    kwargs[k] = self._parse_spec(v, k)
                else:
                    kwargs[k] = v

            # Parse outputs
            out_spec = None
            if "output_spec" in sub:
                out_spec = self._parse_spec(sub["output_spec"], "out")

            out_specs = None
            if "output_specs" in sub:
                out_specs = [
                    self._parse_spec(s, f"out_{i}")
                    for i, s in enumerate(sub["output_specs"])
                ]

            # Determine output count
            out_count = len(out_specs) if out_specs else sub.get("output_count", 1)

            tc = TestCase(
                inputs=inputs,
                kwargs=kwargs,
                output_spec=out_spec,
                output_specs=out_specs,
                comparison_target=base_cmp,
                tolerance=sub.get("tolerance", base_tol),
                description=full_desc,
                output_count=out_count,
            )
            test_cases_list.append(tc)

        return test_cases_list

    def _execute_tests(self, op_name, test_cases, args, op_funcs):
        # Define dynamic test class
        class DynamicOpTest(BaseOperatorTest):
            def __init__(self):
                super().__init__(op_name)

            def get_test_cases(self):
                return test_cases

            def torch_operator(self, *a, **k):
                return op_funcs["torch"](*a, **k)

            def infinicore_operator(self, *a, **k):
                return op_funcs["infinicore"](*a, **k)

        runner = GenericTestRunner(DynamicOpTest, args)
        _, internal_runner = runner.run()

        # Returns a list of TestResult objects
        return getattr(internal_runner, "test_results", [])

    def _prepare_report_entry(
        self, op_name, test_cases, args, op_paths, device, results_list
    ):
        """
        Separates 'cases' (static input) and 'execution_results' (dynamic output).
        """
        # Map results by index
        results_map = {}
        if isinstance(results_list, list):
            results_map = {i: res for i, res in enumerate(results_list)}
        elif isinstance(results_list, dict):
            results_map = results_list
        else:
            results_map = {0: results_list}

        processed_cases = []
        formatted_results = []

        for idx, tc in enumerate(test_cases):
            # 1. Reconstruct case dict (Static info ONLY)
            case_data = {
                "description": tc.description,
                "inputs": [self._spec_to_dict(i) for i in tc.inputs],
                "kwargs": {
                    k: (
                        self._spec_to_dict(v) if isinstance(v, TensorSpec) else v
                    )
                    for k, v in tc.kwargs.items()
                },
                "comparison_target": tc.comparison_target,
                "tolerance": tc.tolerance,
            }

            if tc.output_spec:
                case_data["output_spec"] = self._spec_to_dict(tc.output_spec)

            if hasattr(tc, "output_specs") and tc.output_specs:
                case_data["output_specs"] = [
                    self._spec_to_dict(s) for s in tc.output_specs
                ]
            
            processed_cases.append(case_data)

            # 2. Extract Result
            res = results_map.get(idx)
            if res:
                formatted_results.append(self._fmt_result(res))
            else:
                formatted_results.append({"status": {"success": False, "error": "No result"}})

        # Global Arguments
        global_args = {
            k: getattr(args, k)
            for k in ["bench", "num_prerun", "num_iterations", "verbose", "debug"]
            if hasattr(args, k)
        }

        # Use tolerance from the first case as global tolerance display
        global_tolerance = test_cases[0].tolerance if test_cases else  {"atol": 1e-3, "rtol": 1e-3}

        return {
            "operator": op_name,
            "device": device,
            "description": f"Test Report for {op_name}",
            "torch_op": op_paths["torch"],
            "infinicore_op": op_paths["infinicore"],
            "tolerance": global_tolerance,
            "args": global_args,
            "cases": processed_cases,
            "execution_results": formatted_results,
        }

    def _save_all_results(self, save_path, total_results):
        print(f"💾 Saving to: {save_path}")
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write("[\n")

                for i, entry in enumerate(total_results):
                    f.write("    {\n")
                    keys = list(entry.keys())

                    for j, key in enumerate(keys):
                        # ✅ Apply list compression to both 'cases' and 'execution_results'
                        if key in ["cases", "execution_results"] and isinstance(entry[key], list):
                            f.write(f'        "{key}": [\n')
                            sub_list = entry[key]
                            for c_idx, c_item in enumerate(sub_list):
                                # Compress each item into one line
                                c_str = json.dumps(c_item, ensure_ascii=False)
                                comma = "," if c_idx < len(sub_list) - 1 else ""
                                f.write(f"            {c_str}{comma}\n")
                            
                            list_comma = "," if j < len(keys) - 1 else ""
                            f.write(f"        ]{list_comma}\n")
                        else:
                            # Standard compact formatting for other fields
                            k_str = json.dumps(key, ensure_ascii=False)
                            v_str = json.dumps(entry[key], ensure_ascii=False)
                            
                            comma = "," if j < len(keys) - 1 else ""
                            f.write(f"        {k_str}: {v_str}{comma}\n")

                    if i < len(total_results) - 1:
                        f.write("    },\n")
                    else:
                        f.write("    }\n")

                f.write("]\n")
            print(f"   ✅ Saved (Structure Matched).")
        except Exception as e:
            print(f"   ❌ Save failed: {e}")

    # --- Helpers ---

    def _discover_op_path(self, op_name: str, candidates: List[str]) -> str:
        """
        Attempts to find a valid function path by trying imports.
        """
        for prefix in candidates:
            full_path = f"{prefix}.{op_name}"
            try:
                self._load_function(full_path)
                return full_path
            except (ImportError, AttributeError, ValueError):
                continue
        raise ValueError(
            f"❌ Could not auto-discover function for operator '{op_name}' "
            f"in candidates: {candidates}. Please specify 'torch_op'/'infinicore_op' explicitly."
        )
        
    def _parse_spec(self, d, name):
        """
        Parses dict into TensorSpec.
        """
        strides = tuple(d["strides"]) if d.get("strides") else None
        
        return TensorSpec.from_tensor(
            tuple(d["shape"]),
            strides,
            getattr(infinicore, d.get("dtype", "float32"), infinicore.float32),
            name=d.get("name", name),
        )

    def _spec_to_dict(self, s):
        return {
            "name": s.name,
            "shape": list(s.shape) if s.shape else None,
            "dtype": str(s.dtype).split(".")[-1],
            # Add strides to output if present
            "strides": list(s.strides) if s.strides else None,
        }

    def _fmt_result(self, res):
        """
        Format result with optimized Map lookup.
        """
        if not (is_dataclass(res) or hasattr(res, "success")):
            return str(res)

        get_time = lambda k: round(getattr(res, k, 0.0), 4)

        # Build Map Locally
        device_id_map = {
            v: k 
            for k, v in vars(InfiniDeviceEnum).items() 
            if not k.startswith("_")
        }

        raw_id = getattr(res, "device", None)
        dev_str = device_id_map.get(raw_id, str(raw_id))

        return {
            "status": {
                "success": getattr(res, "success", False),
                "error": getattr(res, "error_message", ""),
            },
            "perf_ms": {
                "torch": {
                    "host": get_time("torch_host_time"),
                    "device": get_time("torch_device_time"),
                },
                "infinicore": {
                    "host": get_time("infini_host_time"),
                    "device": get_time("infini_device_time"),
                },
            },
            "device": dev_str,
        }

    def _load_function(self, path):
        if not path or "." not in path:
            raise ValueError(f"Invalid path: {path}")
        module_name, func_name = path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)

    def _get_default_args(self):
        old_argv = sys.argv
        sys.argv = [sys.argv[0]]
        args = get_args()
        sys.argv = old_argv
        return args

    def _merge_args(self, args, overrides):
        if not overrides:
            return

        data = (
            vars(overrides) if isinstance(overrides, argparse.Namespace) else overrides
        )
        for k, v in data.items():
            if v is not None:
                setattr(args, k, v)

    def _set_device_flags(self, args, device_str):
        # Reset existing flags
        for flag in self.supported_hw_flags:
            if hasattr(args, flag):
                setattr(args, flag, False)

        d = str(device_str).lower()

        if hasattr(args, d):
            setattr(args, d, True)
        else:
            args.cpu = True
            print(f"⚠️ Device '{d}' -> CPU (Fallback)")

    def _normalize_override_config(self, config):
        if isinstance(config, str) and os.path.exists(config):
            with open(config) as f:
                return json.load(f)

        if isinstance(config, argparse.Namespace):
            return vars(config)

        return config or {}

    def _load_default_case(self, overrides):
        args = self._get_default_args()
        self._merge_args(args, overrides)
        self._set_device_flags(args, "cpu")

        data = {
            "description": "Default Add",
            "cases": [
                {
                    "inputs": [{"shape": [13, 4, 4]}, {"shape": [13, 4, 4]}],
                    "output_spec": {"shape": [13, 4, 4]},
                }
            ],
        }

        op_name = "add"
        test_cases = self._build_test_cases(data, op_name)

        op_funcs = {
            "torch": self._load_function("torch.add"),
            "infinicore": self._load_function("infinicore.add"),
        }
        op_paths = {
            "torch": "torch.add",
            "infinicore": "infinicore.add",
        }

        return op_name, test_cases, args, op_funcs, op_paths
