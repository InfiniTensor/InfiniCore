"""
Generic test runner that handles the common execution flow for all operators
"""

import sys
from . import TestConfig, TestRunner, get_args, get_test_devices


class GenericTestRunner:
    """Generic test runner that handles the common execution flow"""

    def __init__(self, operator_test_class):
        """
        Args:
            operator_test_class: A class that implements BaseOperatorTest interface
        """
        self.operator_test = operator_test_class()
        self.args = get_args()

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
            verbose=self.args.verbose,  # Pass verbose flag to TestConfig
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
