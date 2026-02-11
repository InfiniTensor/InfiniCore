import sys
import argparse
import tempfile
from pathlib import Path
from .executor import TestExecutor
from .results import TestSummary, TestTiming
from .utils.load_utils import TestGenerator


class TestCollector:
    """
    Responsible for scanning and verifying operator test files.
    """

    def __init__(self, ops_dir_path=None):
        self.ops_dir = self._resolve_dir(ops_dir_path)

    def _resolve_dir(self, path):
        if path:
            p = Path(path)
            if p.exists():
                return p
        # Fallback: 'ops' directory relative to the project root
        fallback = Path(__file__).parent.parent / "ops"
        return fallback if fallback.exists() else None

    def get_available_operators(self):
        if not self.ops_dir:
            return []
        files = self.scan()
        return sorted([f.stem for f in files])

    def get_raw_python_files(self):
        if not self.ops_dir or not self.ops_dir.exists():
            return []
        files = list(self.ops_dir.glob("*.py"))
        return [
            f.name for f in files if f.name != "run.py" and not f.name.startswith("__")
        ]

    def scan(self, specific_ops=None):
        if not self.ops_dir or not self.ops_dir.exists():
            return []

        files = list(self.ops_dir.glob("*.py"))
        target_ops_set = set(specific_ops) if specific_ops else None
        valid_files = []

        for f in files:
            if f.name.startswith("_") or f.name == "run.py":
                continue
            if target_ops_set and f.stem not in target_ops_set:
                continue
            if self._is_operator_test(f):
                valid_files.append(f)
        return valid_files

    def _is_operator_test(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                return "infinicore" in content and (
                    "BaseOperatorTest" in content or "GenericTestRunner" in content
                )
        except:
            return False


class TestManager:
    """
    High-level API to execute operator tests.
    Encapsulates the test loop, timing aggregation, and reporting.
    """

    def __init__(self, ops_dir=None, verbose=False, bench_mode=None):
        self.collector = TestCollector(ops_dir)
        self.verbose = verbose
        self.bench_mode = bench_mode

        # Initialize components
        self.executor = TestExecutor()
        self.summary = TestSummary(verbose, bench_mode)
        self.cumulative_timing = TestTiming()
        self.results = []

    def test(self, target_ops=None, json_cases_list=None, global_exec_args=None):
        """
        Args:
            target_ops: List of target operators for local scan
            json_cases_list: List of test cases in JSON mode
            global_exec_args (argparse.Namespace): Unified argument object passed to Executor in local scan mode
        """
        with tempfile.TemporaryDirectory() as temp_dir_str:

            test_files = []
            test_configs = []  # Stores args for each file
            display_location = ""

            # =================================================
            # 1. Mode Selection
            # =================================================
            if json_cases_list:
                # [Mode A] Dynamic Execution (JSON)
                print(f"🚀 Mode: Dynamic Execution")
                project_root = getattr(
                    self, "project_root", Path(__file__).resolve().parent.parent
                )
                generator = TestGenerator(project_root=str(project_root))

                # Generate files
                dynamic_paths = generator.generate(json_cases_list, temp_dir_str)
                test_files = [Path(p) for p in dynamic_paths]

                # Convert JSON dict to Namespace
                for case_data in json_cases_list:
                    # run.py has sanitized the data, convert directly to Namespace
                    ns = argparse.Namespace(**case_data.get("args", {}))
                    test_configs.append(ns)

                display_location = f"Dynamic ({len(test_files)} cases)"

            else:
                # [Mode B] Local File Scan
                # print(f"📂 Mode: Local File Scan")
                test_files = self.collector.scan(target_ops)
                display_location = str(self.collector.ops_dir)

                # ✅ Key Logic: Apply global_exec_args passed from run.py to all files
                # If global_exec_args is None (run.py should theoretically fill this), executor falls back to default behavior
                test_configs = [global_exec_args] * len(test_files)

            # =================================================
            # 2. Execution Loop
            # =================================================
            if not test_files:
                print(f"No valid tests found in {display_location}")
                return True

            self.summary.print_header(display_location, len(test_files))

            saved_files = []
            for f, run_args in zip(test_files, test_configs):

                # Inject prepared args (whether from JSON or Local global) into Executor
                result = self.executor.execute(f, test_args=run_args)

                self.results.append(result)
                self.summary.print_live_result(result)

                # Collect saved report files
                if hasattr(result, "saved_file") and result.saved_file:
                    saved_files.append(result.saved_file)

                if result.success:
                    self._accumulate_timing(result.timing)

                if self.verbose and not result.success:
                    print("\nStopping due to failure in verbose mode.")
                    break

            # Summary
            all_passed = self.summary.print_summary(
                self.results,
                self.cumulative_timing if self.bench_mode else None,
                ops_dir=display_location,
                total_expected=len(test_files),
            )

            return all_passed, saved_files

    def _accumulate_timing(self, timing):
        self.cumulative_timing.torch_host += timing.torch_host
        self.cumulative_timing.infini_host += timing.infini_host
        self.cumulative_timing.torch_device += timing.torch_device
        self.cumulative_timing.infini_device += timing.infini_device
        self.cumulative_timing.operators_tested += 1
