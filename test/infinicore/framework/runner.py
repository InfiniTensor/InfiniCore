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
        """Execute the complete test suite"""
        config = TestConfig(
            tensor_dtypes=self.operator_test.tensor_dtypes,
            tolerance_map=self.operator_test.tolerance_map,
            debug=self.args.debug,
            bench=self.args.bench,
            num_prerun=self.args.num_prerun,
            num_iterations=self.args.num_iterations,
            dtype_combinations=self.operator_test.dtype_combinations,
        )

        runner = TestRunner(self.operator_test.test_cases, config)
        devices = get_test_devices(self.args)

        # Run unified tests
        all_passed = runner.run_tests(
            devices, self.operator_test.run_test, self.operator_test.operator_name
        )

        # Print summary
        summary_passed = runner.print_summary()
        all_passed = all_passed and summary_passed

        return all_passed

    def run_and_exit(self):
        """Run tests and exit with appropriate status code"""
        success = self.run()
        sys.exit(0 if success else 1)
