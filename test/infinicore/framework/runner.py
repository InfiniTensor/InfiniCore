"""
Generic test runner that handles the common execution flow for all operators
"""

import sys
import os
import inspect
import re
from . import TestConfig, TestRunner, get_args, get_test_devices
from .results import TestSummary


class GenericTestRunner:
    """Generic test runner that handles the common execution flow"""

    def __init__(self, operator_test_class, args=None):
        """
        Args:
            operator_test_class: A class that implements BaseOperatorTest interface
        """
        self.operator_test = operator_test_class()
        self.args = args or get_args()
        self.saved_file = None  # Store the path of saved report

    def run(self):
        """Execute the complete test suite

        Returns:
            tuple: (success, test_runner) where:
                success: bool indicating if all tests passed or were skipped/partial
                test_runner: TestRunner instance with test results
        """
        config = TestConfig(
            debug=self.args.debug,
            bench=self.args.bench,
            num_prerun=self.args.num_prerun,
            num_iterations=self.args.num_iterations,
            verbose=self.args.verbose,
            equal_nan=self.args.eq_nan,
        )

        runner = TestRunner(self.operator_test.test_cases, config)

        # Pass the test runner instance to config for benchmark timing accumulation
        config._test_runner = runner

        devices = get_test_devices(self.args)

        # Run unified tests - returns True if no tests failed
        # (skipped/partial tests don't count as failures)
        has_no_failures = runner.run_tests(
            devices, self.operator_test.run_test, self.operator_test.operator_name
        )

        # Print summary and get final result
        # summary_passed returns True if no tests failed (skipped/partial are OK)
        summary_passed = runner.print_summary()

        if getattr(self.args, "save", None):
            self.saved_file = self._save_report(runner)

        # Both conditions must be True for overall success
        # - has_no_failures: no test failures during execution
        # - summary_passed: summary confirms no failures
        return (has_no_failures and summary_passed), runner

    def run_and_exit(self):
        """Run tests and exit with appropriate status code

        Exit codes:
            0: All tests passed or were skipped/partial (no failures)
            1: One or more tests failed
        """
        success, runner = self.run()

        sys.exit(0 if success else 1)

    def _save_report(self, runner):
        """
        Helper method to collect metadata and trigger report saving.
        """
        try:

            # 1. Prepare metadata (Paths)
            t_path = self._infer_op_path(self.operator_test.torch_operator, "torch")
            i_path = self._infer_op_path(
                self.operator_test.infinicore_operator, "infinicore"
            )

            op_paths = {"torch": t_path, "infinicore": i_path}

            # 2. Generate Report Entries
            test_summary = TestSummary()
            entries = test_summary.collect_report_entry(
                op_name=self.operator_test.operator_name,
                test_cases=self.operator_test.test_cases,
                args=self.args,
                op_paths=op_paths,
                results_list=runner.test_results,
            )

            # 3. Save to File and return the file name
            return test_summary.save_report(self.args.save)

        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"⚠️ Failed to save report: {e}")
            return None

    def _infer_op_path(self, method, lib_prefix):
        """
        Introspects the method source code to find calls like 'torch.add' or 'infinicore.mul'.
        Returns the full path string (e.g., 'torch.add') or None if not found.
        """
        try:
            source = inspect.getsource(method)

            # Regex to find 'lib.func' or 'lib.submodule.func'
            # Matches: 'torch.add', 'torch.nn.functional.relu'
            pattern = re.compile(
                rf"\b{lib_prefix}\.([a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*)", re.IGNORECASE
            )
            match = pattern.search(source)
            if match:
                # Return the matched string exactly as found in source code
                # or normalize it (e.g. lowercase lib_prefix + match)
                return f"{lib_prefix}.{match.group(1)}"
        except Exception:
            # Handle cases where source is not available (e.g. compiled modules)
            pass
        return None
