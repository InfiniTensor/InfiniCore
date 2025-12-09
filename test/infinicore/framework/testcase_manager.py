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
        overrides = self._normalize_override_config(config)

        # 1. Unified Configuration Loading
        if json_file_path and os.path.exists(json_file_path):
            print(f"📄 Source: Loading {json_file_path}")
            test_configs = self._load(json_file_path, overrides)
        else:
            test_configs = self._load_default_case(overrides)

        total_results = []

        # 2. Execute & Collect Results
        for idx, cfg in enumerate(test_configs):
            op_name = cfg["op_name"]
            n_cases = len(cfg["test_cases"])
            print(f"\n🔹 Config {idx + 1}/{len(test_configs)}: {op_name} ({n_cases} cases)")

            # Execute
            results = self._execute_tests(
                op_name, cfg["test_cases"], cfg["args"], cfg["op_funcs"]
            )

            # Report
            entry = self._prepare_report_entry(cfg, results)
            total_results.append(entry)

        # 3. Save
        if save_path:
            self._save_all_results(save_path, total_results)

        return total_results

    def _create_exec_config(self, raw_data: Dict, overrides: Dict) -> Optional[Dict]:
        """
        ✅ Core Simplification: Unified logic to build a config object from raw dict.
        """
        op_name = raw_data.get("operator")
        if not op_name:
            return None

        # 1. Resolve Paths
        t_op = raw_data.get("torch_op") or self._discover_op_path(
            op_name, ["torch", "torch.nn.functional", "torch.special", "torch.fft"]
        )
        i_op = raw_data.get("infinicore_op") or self._discover_op_path(
            op_name, ["infinicore", "infinicore.nn.functional"]
        )

        # 2. Args & Device
        args = self._get_default_args()
        self._merge_args(args, raw_data.get("args", {}))
        self._merge_args(args, overrides)

        dev_str = (
            overrides.get("device")
            if overrides and "device" in overrides
            else raw_data.get("device", "cpu")
        )
        self._set_device_flags(args, dev_str)

        # 3. Build & Return
        return {
            "op_name": op_name,
            "test_cases": self._build_test_cases(raw_data, op_name),
            "args": args,
            "op_funcs": {
                "torch": self._load_function(t_op),
                "infinicore": self._load_function(i_op),
            },
            "op_paths": {"torch": t_op, "infinicore": i_op},
            "target_device": dev_str,
        }

    def _load(self, json_file_path: str, overrides: Dict) -> List[Dict]:
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON: {json_file_path}")

        data_list = data if isinstance(data, list) else [data]
        # Use generator to filter None configs
        return [
            cfg
            for d in data_list
            if (cfg := self._create_exec_config(d, overrides)) is not None
        ]

    def _load_default_case(self, overrides: Dict) -> List[Dict]:
        # Construct raw dict and pass to unified creator
        raw_data = {
            "operator": "add",
            "description": "Default Add",
            "testcases": [
                {
                    "inputs": [{"shape": [13, 4, 4]}, {"shape": [13, 4, 4]}],
                    "output_spec": {"shape": [13, 4, 4]},
                }
            ],
        }
        return [self._create_exec_config(raw_data, overrides)]

    def _build_test_cases(self, data: Dict, op_name: str) -> List[TestCase]:
        cases_data = data.get("testcases")
        if not cases_data or not isinstance(cases_data, list):
            raise ValueError(f"❌ Config for '{op_name}' missing 'testcases' list.")

        base_desc = data.get("description", f"Auto-test {op_name}")

        test_cases_list = []
        for idx, sub in enumerate(cases_data):
            # Compact list/dict comprehensions
            inputs = [
                self._parse_spec(inp, f"in_{i}")
                for i, inp in enumerate(sub.get("inputs", []))
            ]
            
            kwargs = {
                k: (
                    self._parse_spec(v, k)
                    if isinstance(v, dict) and "shape" in v
                    else v
                )
                for k, v in sub.get("kwargs", {}).items()
            }

            out_spec = (
                self._parse_spec(sub["output_spec"], "out")
                if "output_spec" in sub
                else None
            )
            
            out_specs = (
                [self._parse_spec(s, f"out_{i}") for i, s in enumerate(sub["output_specs"])]
                if "output_specs" in sub
                else None
            )

            tol = sub.get("tolerance", {"atol": 1e-3, "rtol": 1e-3})
            cmp = sub.get("comparison_target", None)

            tc = TestCase(
                inputs=inputs,
                kwargs=kwargs,
                output_spec=out_spec,
                output_specs=out_specs,
                comparison_target=cmp,
                tolerance=tol,
                description=f"{base_desc} - {sub.get('description', f'Case_{idx}')}",
                output_count=len(out_specs) if out_specs else sub.get("output_count", 1),
            )
            test_cases_list.append(tc)

        return test_cases_list

    def _execute_tests(self, op_name, test_cases, args, op_funcs):
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
        return getattr(internal_runner, "test_results", [])

    def _prepare_report_entry(self, cfg, results_list):
        # Map results by index
        res_map = (
            {i: r for i, r in enumerate(results_list)}
            if isinstance(results_list, list)
            else {0: results_list}
        )

        cases, results = [], []
        for idx, tc in enumerate(cfg["test_cases"]):
            # 1. Static Info
            cases.append({
                "description": tc.description,
                "inputs": [self._spec_to_dict(i) for i in tc.inputs],
                "kwargs": {
                    k: (self._spec_to_dict(v) if isinstance(v, TensorSpec) else v)
                    for k, v in tc.kwargs.items()
                },
                "comparison_target": tc.comparison_target,
                "tolerance": tc.tolerance,
                **({"output_spec": self._spec_to_dict(tc.output_spec)} if tc.output_spec else {}),
                **({"output_specs": [self._spec_to_dict(s) for s in tc.output_specs]} if tc.output_specs else {}),
                **({"output_count": tc.output_count} if tc.output_count > 1 and not tc.output_specs else {})
            })

            # 2. Dynamic Result
            res = res_map.get(idx)
            results.append(
                self._fmt_result(res) if res else {"status": {"success": False, "error": "No result"}}
            )

        # Global Args
        g_args = {
            k: getattr(cfg["args"], k)
            for k in ["bench", "num_prerun", "num_iterations", "verbose", "debug"]
            if hasattr(cfg["args"], k)
        }

        return {
            "operator": cfg["op_name"],
            "device": cfg["target_device"],
            "description": f"Test Report for {cfg['op_name']}",
            "torch_op": cfg["op_paths"]["torch"],
            "infinicore_op": cfg["op_paths"]["infinicore"],
            "args": g_args,
            "testcases": cases,
            "execution_results": results,
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
                        # Special handling for lists (cases/results)
                        if key in ["testcases", "execution_results"] and isinstance(entry[key], list):
                            f.write(f'        "{key}": [\n')
                            for k_idx, item in enumerate(entry[key]):
                                c_str = json.dumps(item, ensure_ascii=False)
                                comma = "," if k_idx < len(entry[key]) - 1 else ""
                                f.write(f"            {c_str}{comma}\n")
                            f.write(f"        ]{',' if j < len(keys) - 1 else ''}\n")
                        else:
                            k_str = json.dumps(key, ensure_ascii=False)
                            v_str = json.dumps(entry[key], ensure_ascii=False)
                            f.write(f"        {k_str}: {v_str}{',' if j < len(keys) - 1 else ''}\n")
                    f.write(f"    }}{',' if i < len(total_results) - 1 else ''}\n")
                f.write("]\n")
            print(f"   ✅ Saved.")
        except Exception as e:
            print(f"   ❌ Save failed: {e}")

    # --- Helpers ---

    def _discover_op_path(self, op_name: str, candidates: List[str]) -> str:
        for prefix in candidates:
            path = f"{prefix}.{op_name}"
            try:
                self._load_function(path)
                return path
            except (ImportError, AttributeError, ValueError):
                continue
        raise ValueError(f"❌ Cannot find op '{op_name}' in {candidates}")

    def _parse_spec(self, d, name):
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
            "strides": list(s.strides) if s.strides else None,
        }

    def _fmt_result(self, res):
        if not (is_dataclass(res) or hasattr(res, "success")):
            return str(res)
        
        get_time = lambda k: round(getattr(res, k, 0.0), 4)
        
        # Build Map
        dev_map = {v: k for k, v in vars(InfiniDeviceEnum).items() if not k.startswith("_")}
        dev_str = dev_map.get(getattr(res, "device", None), str(getattr(res, "device", None)))

        return {
            "status": {
                "success": getattr(res, "success", False),
                "error": getattr(res, "error_message", ""),
            },
            "perf_ms": {
                "torch": {"host": get_time("torch_host_time"), "device": get_time("torch_device_time")},
                "infinicore": {"host": get_time("infini_host_time"), "device": get_time("infini_device_time")},
            },
            "dev": dev_str,
        }

    def _load_function(self, path):
        if not path or "." not in path: raise ValueError(f"Invalid path: {path}")
        m, f = path.rsplit(".", 1)
        return getattr(importlib.import_module(m), f)

    def _get_default_args(self):
        old_argv = sys.argv; sys.argv = [sys.argv[0]]; args = get_args(); sys.argv = old_argv
        return args

    def _merge_args(self, args, overrides):
        if not overrides: return
        data = vars(overrides) if isinstance(overrides, argparse.Namespace) else overrides
        for k, v in data.items():
            if v is not None: setattr(args, k, v)

    def _set_device_flags(self, args, dev_str):
        for flag in self.supported_hw_flags: setattr(args, flag, False)
        d = str(dev_str).lower()
        if hasattr(args, d): setattr(args, d, True)
        else: args.cpu = True; print(f"⚠️ Device '{d}' -> CPU")

    def _normalize_override_config(self, config):
        if isinstance(config, str) and os.path.exists(config):
            with open(config) as f: return json.load(f)
        return vars(config) if isinstance(config, argparse.Namespace) else (config or {})
