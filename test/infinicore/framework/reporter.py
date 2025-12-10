import json
import os
from datetime import datetime
from typing import List, Dict, Any, Union
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
        results_list: List[Any]
    ) -> List[Dict[str, Any]]:
        """
        Combines static test case info with dynamic execution results.
        """
        # 1. Normalize results
        results_map = {}
        if isinstance(results_list, list):
            results_map = {i: res for i, res in enumerate(results_list)}
        elif isinstance(results_list, dict):
            results_map = results_list
        else:
            results_map = {0: results_list} if results_list else {}

        # 2. Global Args
        global_args = {
            k: getattr(args, k)
            for k in ["bench", "num_prerun", "num_iterations", "verbose", "debug"]
            if hasattr(args, k)
        }

        grouped_entries: Dict[int, Dict[str, Any]] = {}

        # 3. Iterate Test Cases
        for idx, tc in enumerate(test_cases):
            res = results_map.get(idx)
            dev_id = getattr(res, "device", 0) if res else 0

            # --- A. Initialize Group ---
            if dev_id not in grouped_entries:
                device_id_map = {v: k for k, v in vars(InfiniDeviceEnum).items() if not k.startswith("_")}
                dev_str = device_id_map.get(dev_id, str(dev_id))
                
                grouped_entries[dev_id] = {
                    "operator": op_name,
                    "device": dev_str,
                    "torch_op": op_paths.get("torch") or "unknown",
                    "infinicore_op": op_paths.get("infinicore") or "unknown",
                    "args": global_args,
                    "testcases": []
                }

            # --- B. Build Kwargs ---
            display_kwargs = {}
            
            # B1. Process existing kwargs
            for k, v in tc.kwargs.items():
                # Handle Inplace: "out": index -> "out": "input_name"
                if k == "out" and isinstance(v, int):
                    if 0 <= v < len(tc.inputs):
                        display_kwargs[k] = tc.inputs[v].name
                    else:
                        display_kwargs[k] = f"Invalid_Index_{v}"
                else:
                    display_kwargs[k] = (TestReporter._spec_to_dict(v) if isinstance(v, TensorSpec) else v)

            # B2. Inject Outputs into Kwargs
            if hasattr(tc, "output_specs") and tc.output_specs:
                for i, spec in enumerate(tc.output_specs):
                    display_kwargs[f"out_{i}"] = TestReporter._spec_to_dict(spec)
            elif tc.output_spec:
                if "out" not in display_kwargs:
                    display_kwargs["out"] = TestReporter._spec_to_dict(tc.output_spec)

            # --- C. Build Test Case Dictionary ---
            case_data = {
                "description": tc.description,
                "inputs": [TestReporter._spec_to_dict(i) for i in tc.inputs],
                "kwargs": display_kwargs, 
                "comparison_target": tc.comparison_target,
                "tolerance": tc.tolerance,
            }

            # --- D. Inject Result ---
            if res:
                case_data["result"] = TestReporter._fmt_result(res)
            else:
                case_data["result"] = {"status": {"success": False, "error": "No result"}}

            grouped_entries[dev_id]["testcases"].append(case_data)

        return list(grouped_entries.values())

    @staticmethod
    def save_all_results(save_path: str, total_results: List[Dict[str, Any]]):
        """
        Saves the report list to a JSON file with specific custom formatting
        """
        directory, filename = os.path.split(save_path)
        name, ext = os.path.splitext(filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        final_path = os.path.join(directory, f"{name}_{timestamp}{ext}")

        # Define indentation levels for cleaner code
        indent_4  = ' ' * 4
        indent_8  = ' ' * 8
        indent_12 = ' ' * 12
        indent_16 = ' ' * 16
        indent_20 = ' ' * 20

        print(f"💾 Saving to: {final_path}")
        try:
            with open(final_path, "w", encoding="utf-8") as f:
                f.write("[\n")

                for i, entry in enumerate(total_results):
                    f.write(f"{indent_4}{{\n")
                    keys = list(entry.keys())

                    for j, key in enumerate(keys):
                        val = entry[key]
                        comma = "," if j < len(keys) - 1 else ""

                        # -------------------------------------------------
                        # Special Handling for 'testcases' list formatting
                        # -------------------------------------------------
                        if key == "testcases" and isinstance(val, list):
                            f.write(f'{indent_8}"{key}": [\n')
                            
                            for c_idx, case_item in enumerate(val):
                                f.write(f"{indent_12}{{\n")
                                case_keys = list(case_item.keys())
                                
                                for k_idx, c_key in enumerate(case_keys):
                                    c_val = case_item[c_key]
                                    
                                    # [Logic A] Skip fields we merged manually after 'kwargs'
                                    if c_key in ["comparison_target", "tolerance"]:
                                        continue
                                    
                                    # Check comma for standard logic (might be overridden below)
                                    c_comma = "," if k_idx < len(case_keys) - 1 else ""

                                    # [Logic B] Handle 'kwargs' + Grouped Fields
                                    if c_key == "kwargs":
                                        # 1. Use Helper for kwargs (Fill/Flow logic)
                                        TestReporter._write_smart_field(
                                            f, c_key, c_val, indent_16, indent_20, close_comma=","
                                        )

                                        # 2. Write subsequent comparison_target and tolerance (on a new line)
                                        cmp_v = json.dumps(case_item.get("comparison_target"), ensure_ascii=False)
                                        tol_v = json.dumps(case_item.get("tolerance"), ensure_ascii=False)
                                        
                                        remaining_keys = [k for k in case_keys[k_idx+1:] if k not in ("comparison_target", "tolerance")]
                                        line_comma = "," if remaining_keys else ""
                                        
                                        f.write(f'{indent_16}"comparison_target": {cmp_v}, "tolerance": {tol_v}{line_comma}\n')
                                        continue

                                    # [Logic C] Handle 'inputs' (Smart Wrap)
                                    if c_key == "inputs" and isinstance(c_val, list):
                                        TestReporter._write_smart_field(
                                            f, c_key, c_val, indent_16, indent_20, close_comma=c_comma
                                        )
                                        continue
                                    
                                    # [Logic D] Standard fields (description, result, output_spec, etc.)
                                    else:
                                        c_val_str = json.dumps(c_val, ensure_ascii=False)
                                        f.write(f'{indent_16}"{c_key}": {c_val_str}{c_comma}\n')
                                
                                close_comma = "," if c_idx < len(val) - 1 else ""
                                f.write(f"{indent_12}}}{close_comma}\n")
                            
                            f.write(f"{indent_8}]{comma}\n")

                        # -------------------------------------------------
                        # Standard top-level fields (operator, args, etc.)
                        # -------------------------------------------------
                        else:
                            k_str = json.dumps(key, ensure_ascii=False)
                            v_str = json.dumps(val, ensure_ascii=False)
                            f.write(f"{indent_8}{k_str}: {v_str}{comma}\n")

                    if i < len(total_results) - 1:
                        f.write(f"{indent_4}}},\n")
                    else:
                        f.write(f"{indent_4}}}\n")

                f.write("]\n")
            print(f"   ✅ Saved (Structure Matched).")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"   ❌ Save failed: {e}")

    # --- Internal Helpers ---
    @staticmethod
    def _write_smart_field(f, key, value, indent, sub_indent, close_comma=""):
        """
        Helper to write a JSON field (List or Dict) with smart wrapping.
        - If compact length <= 180: Write on one line.
        - If > 180: Use 'Fill/Flow' mode (multiple items per line, wrap when line is full).
        """
        # 1. Try Compact Mode
        compact_json = json.dumps(value, ensure_ascii=False)
        if len(compact_json) <= 180:
            f.write(f'{indent}"{key}": {compact_json}{close_comma}\n')
            return

        # 2. Fill/Flow Mode
        is_dict = isinstance(value, dict)
        open_char = '{' if is_dict else '['
        close_char = '}' if is_dict else ']'
        
        f.write(f'{indent}"{key}": {open_char}')
        
        # Normalize items for iteration
        if is_dict:
            items = list(value.items())
        else:
            items = value # List

        # Initialize current line length tracking
        # Length includes indent + "key": [
        current_len = len(indent) + len(f'"{key}": {open_char}')
        
        for i, item in enumerate(items):
            # Format individual item string
            if is_dict:
                k, v = item
                val_str = json.dumps(v, ensure_ascii=False)
                item_str = f'"{k}": {val_str}'
            else:
                item_str = json.dumps(item, ensure_ascii=False)
            
            is_last = (i == len(items) - 1)
            item_comma = "" if is_last else ", "
            
            # Predict new length: current + item + comma
            if current_len + len(item_str) + len(item_comma) > 180:
                # Wrap to new line
                f.write(f'\n{sub_indent}') 
                current_len = len(sub_indent)
            
            f.write(f'{item_str}{item_comma}')
            current_len += len(item_str) + len(item_comma)
            
        f.write(f'{close_char}{close_comma}\n')

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

        get_time = lambda k: round(getattr(res, k, 0.0), 4)

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
        }
