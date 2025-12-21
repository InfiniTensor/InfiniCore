from typing import List, Dict, Any
from dataclasses import is_dataclass
from .devices import InfiniDeviceEnum
from .base import TensorSpec
from .utils.json_utils import save_json_report


class TestSummary:
    """
    Test summary:
    1. Aggregates results (Timing & Status calculation).
    2. Handles Console Output (Live & Summary).
    3. Handles File Reporting (Data Preparation).
    """

    def __init__(self, verbose=False, bench_mode=None):
        self.verbose = verbose
        self.bench_mode = bench_mode
        self.report_entries = []  # Cache for JSON report

    # =========================================================
    #  Part 1: Result Aggregation
    # =========================================================

    def process_operator_result(self, op_result, sub_results: List):
        """
        Updates the OperatorResult object in-place.
        """
        if not sub_results:
            op_result.return_code = -1
            return

        # 1. Analyze Return Code (Status)
        if op_result.success:
            op_result.return_code = 0
        else:
            has_failures = any(r.return_code == -1 for r in sub_results)
            has_partial = any(r.return_code == -3 for r in sub_results)
            has_skipped = any(r.return_code == -2 for r in sub_results)

            if has_failures:
                op_result.return_code = -1
            elif has_partial:
                op_result.return_code = -3
            elif has_skipped:
                op_result.return_code = -2
            else:
                op_result.return_code = -1

        # 2. Extract Timing (Aggregation)
        t = op_result.timing
        t.torch_host = sum(r.torch_host_time for r in sub_results)
        t.torch_device = sum(r.torch_device_time for r in sub_results)
        t.infini_host = sum(r.infini_host_time for r in sub_results)
        t.infini_device = sum(r.infini_device_time for r in sub_results)
        t.operators_tested = len(sub_results)

    # =========================================================
    #  Part 2: Console Output (View)
    # =========================================================

    def list_tests(self, discoverer):
        ops_dir = discoverer.ops_dir
        operators = discoverer.get_available_operators()

        if operators:
            print(f"Available operator test files in {ops_dir}:")
            for operator in operators:
                print(f"  - {operator}")
            print(f"\nTotal: {len(operators)} operators")
        else:
            print(f"No valid operator tests found in {ops_dir}")
            raw_files = discoverer.get_raw_python_files()
            if raw_files:
                print(
                    f"\n💡 Debug Hint: Found Python files but they are not valid tests:"
                )
                print(f"   {raw_files}")

    def print_header(self, ops_dir, count):
        print(f"InfiniCore Operator Test Runner")
        print(f"Directory: {ops_dir}")
        print(f"Tests found: {count}\n")

    def print_live_result(self, result):
        print(
            f"{result.status_icon}  {result.name}: {result.status_text} (code: {result.return_code})"
        )

        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print("\nSTDERR:", result.stderr.rstrip())
        if result.error_message:
            print(f"💥 Error: {result.error_message}")

        if result.stdout or result.stderr or self.verbose:
            print("-" * 40)

    def print_summary(self, results, cumulative_timing, ops_dir, total_expected=0):
        print(f"\n{'='*80}\nCUMULATIVE TEST SUMMARY\n{'='*80}")

        passed = [r for r in results if r.return_code == 0]
        failed = [r for r in results if r.return_code == -1]
        skipped = [r for r in results if r.return_code == -2]
        partial = [r for r in results if r.return_code == -3]

        total = len(results)
        print(f"Total tests run: {total}")
        if total_expected > 0 and total < total_expected:
            print(f"Total tests expected: {total_expected}")
            print(f"Tests not executed: {total_expected - total}")

        print(f"Passed: {len(passed)}")
        print(f"Failed: {len(failed)}")
        if skipped:
            print(f"Skipped: {len(skipped)}")
        if partial:
            print(f"Partial: {len(partial)}")

        # 1. Benchmark
        if cumulative_timing:
            self._print_timing(cumulative_timing)

        # 2. Lists
        if passed:
            self._print_op_list("✅ PASSED OPERATORS", passed)
        else:
            print(f"\n✅ PASSED OPERATORS: None")

        if failed:
            self._print_op_list("❌ FAILED OPERATORS", failed)
        if skipped:
            self._print_op_list("⏭️  SKIPPED OPERATORS", skipped)
        if partial:
            self._print_op_list("⚠️  PARTIAL IMPLEMENTATIONS", partial)

        # 3. Verdict
        if total > 0:
            executed_tests = total - len(skipped)
            if executed_tests > 0:
                success_rate = len(passed) / executed_tests * 100
                print(f"\nSuccess rate: {success_rate:.1f}%")

        if not failed:
            if skipped or partial:
                print(f"\n⚠️  Tests completed with some operators not fully implemented")
            else:
                print(f"\n🎉 All tests passed!")
        else:
            print(f"\n❌ {len(failed)} tests failed")

        if not failed and (skipped or partial):
            print(f"\n⚠️  Note: Some operators are not fully implemented")
            print(f"   Run individual tests for details on missing implementations")

        if self.verbose and failed:
            print(
                f"\n💡 Verbose mode tip: Use individual test commands for detailed debugging:"
            )
            for r in failed[:3]:
                file_path = ops_dir / (r.name + ".py")
                print(f"   python {file_path} --verbose")
            if len(failed) > 3:
                print(f"   ... (and {len(failed) - 3} others)")

        return len(failed) == 0

    def _print_timing(self, t):
        print(f"{'-'*40}")
        if hasattr(t, "operators_tested") and t.operators_tested > 0:
            print(f"BENCHMARK SUMMARY ({t.operators_tested} cases):")

        if self.bench_mode in ["host", "both"]:
            print(f"  [Host] PyTorch:    {t.torch_host:10.3f} ms")
            print(f"  [Host] InfiniCore: {t.infini_host:10.3f} ms")

        if self.bench_mode in ["device", "both"]:
            print(f"  [Device] PyTorch:    {t.torch_device:10.3f} ms")
            print(f"  [Device] InfiniCore: {t.infini_device:10.3f} ms")
        print(f"{'-'*40}")

    def _print_op_list(self, title, result_list):
        print(f"\n{title} ({len(result_list)}):")
        names = [r.name for r in result_list]
        for i in range(0, len(names), 10):
            print("  " + ", ".join(names[i : i + 10]))

    # =========================================================
    #  Part 3: Report Generation
    # =========================================================

    def collect_report_entry(self, op_name, test_cases, args, op_paths, results_list):
        """
        Prepares the data and adds it to the internal list.
        """
        entry = self._prepare_entry_logic(
            op_name, test_cases, args, op_paths, results_list
        )
        self.report_entries.extend(entry)

    def save_report(self, save_path):
        """
        Delegates the actual writing to save_json_report.
        """
        if not self.report_entries:
            return
        # Call the external utility
        save_json_report(save_path, self.report_entries)

    def _prepare_entry_logic(self, op_name, test_cases, args, op_paths, results_list):
        """
        Combines static test case info with dynamic execution results.
        Refactored to reduce duplication.
        """
        # 1. Normalize results
        results_map = (
            results_list
            if isinstance(results_list, dict)
            else {i: res for i, res in enumerate(results_list or [])}
        )

        # 2. Global Args
        global_args = {
            k: getattr(args, k)
            for k in ["bench", "num_prerun", "num_iterations", "verbose", "debug"]
            if hasattr(args, k)
        }

        grouped_entries = {}

        # Cache device enum map
        device_id_map = {
            v: k for k, v in vars(InfiniDeviceEnum).items() if not k.startswith("_")
        }

        for idx, tc in enumerate(test_cases):
            res = results_map.get(idx)
            dev_id = getattr(res, "device", 0) if res else 0

            # --- A. Initialize Group ---
            if dev_id not in grouped_entries:
                grouped_entries[dev_id] = {
                    "operator": op_name,
                    "device": device_id_map.get(dev_id, str(dev_id)),
                    "torch_op": op_paths.get("torch", "unknown"),
                    "infinicore_op": op_paths.get("infinicore", "unknown"),
                    "args": global_args,
                    "testcases": [],
                }

            # --- B. Helpers for Spec Processing ---
            def process_spec(spec, default_name):
                final_name = self._resolve_name(spec, default_name)
                # Call internal method (no need for external converters file)
                return self._spec_to_dict(spec, name=final_name)

            # --- C. Build Inputs ---
            processed_inputs = [
                process_spec(inp, f"in_{i}") for i, inp in enumerate(tc.inputs)
            ]

            # --- D. Build Kwargs ---
            display_kwargs = {}
            for k, v in tc.kwargs.items():
                if k == "out" and isinstance(v, int):
                    # Handle Inplace Index
                    if 0 <= v < len(tc.inputs):
                        display_kwargs[k] = self._resolve_name(tc.inputs[v], f"in_{v}")
                    else:
                        display_kwargs[k] = f"Invalid_Index_{v}"
                elif isinstance(v, TensorSpec):
                    display_kwargs[k] = process_spec(v, v.name)
                else:
                    display_kwargs[k] = v

            # --- E. Inject Outputs ---
            if getattr(tc, "output_specs", None):
                for i, spec in enumerate(tc.output_specs):
                    display_kwargs[f"out_{i}"] = process_spec(spec, f"out_{i}")
            elif tc.output_spec and "out" not in display_kwargs:
                display_kwargs["out"] = process_spec(tc.output_spec, "out")

            # --- F. Assemble Case Data ---
            case_data = {
                "description": tc.description,
                "inputs": processed_inputs,
                "kwargs": display_kwargs,
                "comparison_target": tc.comparison_target,
                "tolerance": tc.tolerance,
                "result": (
                    self._fmt_result(res)
                    if res
                    else {"status": {"success": False, "error": "No result"}}
                ),
            }

            grouped_entries[dev_id]["testcases"].append(case_data)

        return list(grouped_entries.values())

    # --- Internal Helpers ---

    def _resolve_name(self, obj, default_name):
        return getattr(obj, "name", None) or default_name

    def _spec_to_dict(self, s, name=None):
        return {
            "name": name if name else getattr(s, "name", "unknown"),
            "shape": list(s.shape) if s.shape else None,
            "dtype": str(s.dtype).split(".")[-1],
            "strides": list(s.strides) if s.strides else None,
        }

    def _fmt_result(self, res):
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
