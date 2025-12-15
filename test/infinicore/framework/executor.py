import sys
import importlib.util
from io import StringIO
from contextlib import contextmanager
from .datatypes import SingleTestResult, TestTiming

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

class SingleTestExecutor:
    def run(self, file_path) -> SingleTestResult:
        result = SingleTestResult(name=file_path.stem)
        
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
            self._analyze_return_code(result, test_results)
            self._extract_timing(result, test_results)

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

    def _analyze_return_code(self, result, test_results):
        # Logic consistent with original code: determine if all passed, partially passed, or skipped
        if not result.success:
            result.return_code = -1
            return
            
        codes = [r.return_code for r in test_results]
        if -1 in codes: result.return_code = -1
        elif -3 in codes: result.return_code = -3
        elif -2 in codes: result.return_code = -2
        else: result.return_code = 0

    def _extract_timing(self, result, test_results):
        # Accumulate timing
        t = result.timing
        t.torch_host = sum(r.torch_host_time for r in test_results)
        t.torch_device = sum(r.torch_device_time for r in test_results)
        t.infini_host = sum(r.infini_host_time for r in test_results)
        t.infini_device = sum(r.infini_device_time for r in test_results)
