import json
import time
import os
from typing import List, Dict, Any
from dataclasses import is_dataclass
from .base import TensorSpec
from .devices import InfiniDeviceEnum

class TestReporter:
    """
    Handles report generation and file saving for test results.
    """

    @staticmethod
    def prepare_report_entry(
        op_name: str, 
        test_cases: List[Any], 
        args: Any, 
        op_paths: Dict[str, str], 
        device: str, 
        results_list: List[Any]
    ) -> Dict[str, Any]:
        """
        Combines static test case info with dynamic execution results.
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
        
        for idx, tc in enumerate(test_cases):
            # 1. Reconstruct case dict (Static info)
            case_data = {
                "description": tc.description,
                "inputs": [TestReporter._spec_to_dict(i) for i in tc.inputs],
                "kwargs": {
                    k: (
                        TestReporter._spec_to_dict(v) if isinstance(v, TensorSpec) else v
                    )
                    for k, v in tc.kwargs.items()
                },
                "comparison_target": tc.comparison_target,
                "tolerance": tc.tolerance,
            }

            if tc.output_spec:
                case_data["output_spec"] = TestReporter._spec_to_dict(tc.output_spec)

            if hasattr(tc, "output_specs") and tc.output_specs:
                case_data["output_specs"] = [
                    TestReporter._spec_to_dict(s) for s in tc.output_specs
                ]
            
            # 2. Inject Result (Dynamic info) directly into the case
            res = results_map.get(idx)
            if res:
                case_data["result"] = TestReporter._fmt_result(res)
            else:
                case_data["result"] = {"status": {"success": False, "error": "No result"}}

            processed_cases.append(case_data)

        # Global Arguments
        global_args = {
            k: getattr(args, k)
            for k in ["bench", "num_prerun", "num_iterations", "verbose", "debug"]
            if hasattr(args, k)
        }

        return {
            "operator": op_name,
            "device": device,
            "torch_op": op_paths.get("torch") or "unknown",
            "infinicore_op": op_paths.get("infinicore") or "unknown",
            "args": global_args,
            "testcases": processed_cases
        }

    @staticmethod
    def save_all_results(save_path: str, total_results: List[Dict[str, Any]]):
        """
        Saves the report list to a JSON file with compact formatting.
        """
        directory, filename = os.path.split(save_path)
        name, ext = os.path.splitext(filename)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        final_path = os.path.join(directory, f"{name}_{timestamp}{ext}")

        print(f"💾 Saving to: {final_path}")
        try:
            with open(final_path, "w", encoding="utf-8") as f:
                f.write("[\n")

                for i, entry in enumerate(total_results):
                    f.write("    {\n")
                    keys = list(entry.keys())

                    for j, key in enumerate(keys):
                        # Special Handling for list fields: vertical expansion
                        if key in ["testcases"] and isinstance(entry[key], list):
                            f.write(f'        "{key}": [\n')
                            sub_list = entry[key]
                            for c_idx, c_item in enumerate(sub_list):
                                c_str = json.dumps(c_item, ensure_ascii=False)
                                comma = "," if c_idx < len(sub_list) - 1 else ""
                                f.write(f"            {c_str}{comma}\n")
                            
                            list_comma = "," if j < len(keys) - 1 else ""
                            f.write(f"        ]{list_comma}\n")
                        else:
                            # Standard compact formatting
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

    # --- Internal Helpers ---

    @staticmethod
    def _spec_to_dict(s):
        return {
            "name": getattr(s, "name", "unknown"),
            "shape": list(s.shape) if s.shape else None,
            "dtype": str(s.dtype).split(".")[-1],
            "strides": list(s.strides) if s.strides else None,
        }

    @staticmethod
    def _fmt_result(res):
        if not (is_dataclass(res) or hasattr(res, "success")):
            return str(res)

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
                    "host": round(getattr(res, "torch_host_time", 0.0), 4),
                    "device": round(getattr(res, "torch_device_time", 0.0), 4),
                },
                "infinicore": {
                    "host": round(getattr(res, "infini_host_time", 0.0), 4),
                    "device": round(getattr(res, "infini_device_time", 0.0), 4),
                },
            },
            "device": dev_str,
        }
