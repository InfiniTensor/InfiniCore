import sys
import os
import json
import importlib
import argparse
from typing import Any, Optional, Tuple, Union, Dict, List
from dataclasses import is_dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
from infinicore import float32
import torch

from framework.base import BaseOperatorTest, TestCase, TensorSpec
from framework.config import get_args, get_supported_hardware_platforms
from framework.runner import GenericTestRunner
from framework.devices import InfiniDeviceEnum
from .reporter import TestReporter

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
        if json_file_path:
            if os.path.exists(json_file_path):
                print(f"📄 Source: Loading {json_file_path}")
                test_configs = self._load(json_file_path, overrides)
            else:
                print(f"⚠️ Warning: File not found: '{json_file_path}'. Falling back to default case.")
                test_configs = self._load_default_case(overrides)
        else:
            print(f"ℹ️ No file provided. Using default built-in case.")
            test_configs = self._load_default_case(overrides)

        total_results = []

        # 2. Execute & Collect Results
        for idx, cfg in enumerate(test_configs):
            op_name = cfg["op_name"]
            n_cases = len(cfg["test_cases"])
            print(f"\n🔹 Config {idx + 1}/{len(test_configs)}: {op_name} ({n_cases} cases)")

            # Execute
            results = self._execute_tests(
                op_name, cfg["test_cases"], cfg["args"], cfg["op_funcs"], cfg["op_paths"]
            )
            
            total_results.append(results)

        return total_results

    def _create_exec_config(self, raw_data: Dict, overrides: Dict) -> Optional[Dict]:
        """
        Unified logic to build a config object from raw dict.
        """
        op_name = raw_data.get("operator")
        if not op_name:
            return None

        # 1. Resolve Paths
        t_op = raw_data.get("torch_op") or self._discover_op_path(
            op_name, ["torch", "torch.nn.functional"]
        )
        i_op = raw_data.get("infinicore_op") or self._discover_op_path(
            op_name, ["infinicore", "infinicore.nn.functional"]
        )

        # 2. Args 
        args = self._get_default_args()
        self._merge_args(args, raw_data.get("args", {}))
        self._merge_args(args, overrides)

        # 3. Resolve Device (String resolution with Priority)
        # Priority 1: Explicit 'device' string in overrides
        dev_str = overrides.get("device") if overrides else None
        
        # Priority 2: Boolean hardware flags in overrides (CLI args like --nvidia)
        if not dev_str and overrides:
            active_flags = [
                flag for flag in self.supported_hw_flags 
                if overrides.get(flag)
            ]
            if active_flags:
                dev_str = ",".join(active_flags)
        
        # Priority 3: JSON config or default
        if not dev_str:
            dev_str = raw_data.get("device", "cpu")
        self._set_device_flags(args, dev_str)

        # 4. Build & Return
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

        test_cases_list = []
        for idx, sub in enumerate(cases_data):
            # 1. Parse inputs (list of TensorSpecs)
            inputs = [
                self._parse_spec(inp, f"in_{i}")
                for i, inp in enumerate(sub.get("inputs", []))
            ]
            
            # 2. Parse kwargs
            kwargs = {}
            for k, v in (sub.get("kwargs") or {}).items():
                if isinstance(v, dict) and "shape" in v:
                    kwargs[k] = self._parse_spec(v, k)
                elif k == "out" and isinstance(v, str):
                    # Find the index of this name within the inputs list
                    index = None
                    for i, spec in enumerate(inputs):
                        if spec.name == v:
                            index = i
                            break
                    if index is None:
                        raise ValueError(
                            f"❌ In test '{op_name}' case {idx}: kwargs['out'] references an unknown input name '{v}'"
                        )
                    # ✅ Replace the string with the index, e.g., "a" -> 0
                    kwargs[k] = index
                else:
                    kwargs[k] = v

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
                description=f"{sub.get('description', f'Case_{idx}')}",
                output_count=len(out_specs) if out_specs else sub.get("output_count", 1),
            )
            test_cases_list.append(tc)

        return test_cases_list

    def _execute_tests(self, op_name, test_cases, args, op_funcs, op_paths):
        class DynamicOpTest(BaseOperatorTest):
            def __init__(self):
                super().__init__(op_name)
                self._op_paths = op_paths

            def get_test_cases(self):
                return test_cases

            def torch_operator(self, *a, **k):
                return op_funcs["torch"](*a, **k)

            def infinicore_operator(self, *a, **k):
                return op_funcs["infinicore"](*a, **k)

            @property
            def op_paths(self):
                return self._op_paths

        runner = GenericTestRunner(DynamicOpTest, args)
        _, internal_runner = runner.run()
        return getattr(internal_runner, "test_results", [])

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
        # 1. Reset all hardware flags first
        for flag in self.supported_hw_flags:
            setattr(args, flag, False)
        
        # 2. Parse string (split by comma)
        d_str = str(dev_str).lower()
        devices = [d.strip() for d in d_str.split(",") if d.strip()]
        
        activated = False
        for d in devices:
            if hasattr(args, d):
                setattr(args, d, True)
                activated = True
            else:
                if d != "cpu":
                    print(f"⚠️ Warning: Unknown device flag '{d}' ignored.")
        
        # 3. Fallback
        if not activated:
            args.cpu = True
            if dev_str != "cpu":
                print(f"⚠️ Device '{dev_str}' invalid/unsupported -> CPU (Fallback)")

    def _normalize_override_config(self, config):
        if isinstance(config, str) and os.path.exists(config):
            with open(config) as f: return json.load(f)
        return vars(config) if isinstance(config, argparse.Namespace) else (config or {})
