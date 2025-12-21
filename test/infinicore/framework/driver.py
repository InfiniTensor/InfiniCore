import sys
import importlib.util
from io import StringIO
from contextlib import contextmanager
from .structs import OperatorResult
from .summary import TestSummary


@contextmanager
def capture_output():
    """Context manager: captures stdout and stderr."""
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield new_out, new_err
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class TestDriver:
    def drive(self, file_path) -> OperatorResult:
        result = OperatorResult(name=file_path.stem)

        try:
            # 1. Dynamically import the module
            module = self._import_module(file_path)

            # 2. Look for TestRunner
            if not hasattr(module, "GenericTestRunner"):
                raise ImportError("No GenericTestRunner found in module")

            # 3. Look for TestClass (subclass of BaseOperatorTest)
            test_class = self._find_test_class(module)
            if not test_class:
                raise ImportError("No BaseOperatorTest subclass found")

            test_instance = test_class()
            runner_class = module.GenericTestRunner
            runner = runner_class(test_instance.__class__)

            # 4. Execute and capture output
            with capture_output() as (out, err):
                success, internal_runner = runner.run()

            # 5. Populate results
            result.success = success
            result.stdout = out.getvalue()
            result.stderr = err.getvalue()

            # Extract detailed results from internal_runner
            test_results = internal_runner.get_test_results() if internal_runner else []

            test_summary = TestSummary()
            test_summary.process_operator_result(result, test_results)
            # test_summary._extract_timing(result, test_results)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.stderr += f"\nExecutor Error: {str(e)}"
            result.return_code = -1

        return result

    def _import_module(self, path):
        module_name = f"op_test_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if not spec or not spec.loader:
            raise ImportError(f"Could not load spec from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _find_test_class(self, module):
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "__bases__"):
                # Simple check for base class name
                if any("BaseOperatorTest" in str(b) for b in attr.__bases__):
                    return attr
        return None
