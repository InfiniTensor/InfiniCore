import sys
from pathlib import Path
from .driver import TestDriver
from .summary import TestSummary
from .structs import TestTiming

class TestDiscoverer:
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
        # Fallback: 'ops' directory relative to the project root (assuming framework/.. is root)
        # Adjust logic depending on where 'framework' package sits relative to 'ops'
        # Here assuming standard structure: root/ops and root/framework/runner.py
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
        return [f.name for f in files if f.name != "run.py" and not f.name.startswith("__")]

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


class TestRunner:
    """
    High-level API to execute operator tests.
    Encapsulates the test loop, timing aggregation, and reporting.
    """
    def __init__(self, ops_dir=None, verbose=False, bench_mode=None):
        self.discoverer = TestDiscoverer(ops_dir)
        self.verbose = verbose
        self.bench_mode = bench_mode
        
        # Initialize components
        self.driver = TestDriver()
        self.summary = TestSummary(verbose, bench_mode)
        self.cumulative_timing = TestTiming()
        self.results = []

    def list_operators(self):
        """Helper to list available operators without running."""
        return self.discoverer.get_available_operators()

    def run(self, target_ops=None):
        """
        Main execution entry point.
        Args:
            target_ops (list[str]): Optional list of operator names to run.
        Returns:
            bool: True if all tests passed, False otherwise.
        """
        # 1. Discovery
        test_files = self.discoverer.scan(target_ops)
        if not test_files:
            print(f"No valid tests found in {self.discoverer.ops_dir}")
            return True # Or False, depending on strictness

        # 2. Print Header
        self.summary.print_header(self.discoverer.ops_dir, len(test_files))

        # 3. Execution Loop
        for f in test_files:
            # Drive single test
            result = self.driver.drive(f)
            self.results.append(result)

            # Live Reporting
            self.summary.print_live_result(result)

            # Accumulate Timing (only if successful)
            if result.success:
                self._accumulate_timing(result.timing)

            # Fail Fast
            if self.verbose and not result.success:
                print("\nStopping due to failure in verbose mode.")
                break

        # 4. Final Summary Report
        all_passed = self.summary.print_summary(
            self.results,
            self.cumulative_timing if self.bench_mode else None,
            ops_dir=self.discoverer.ops_dir,
            total_expected=len(test_files),
        )
        
        # Save Report if needed (TestSummary handles internal buffering)
        # Note: If your TestSummary has a save method, call it here.
        # self.summary.save_report(save_path) 

        return all_passed

    def _accumulate_timing(self, timing):
        """Helper to aggregate timing stats."""
        self.cumulative_timing.torch_host += timing.torch_host
        self.cumulative_timing.infini_host += timing.infini_host
        self.cumulative_timing.torch_device += timing.torch_device
        self.cumulative_timing.infini_device += timing.infini_device
        self.cumulative_timing.operators_tested += 1
