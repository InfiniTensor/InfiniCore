import sys
import os
import importlib
import inspect
import infinicore

# Adapt path to ensure framework can be referenced
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from framework.base import BaseOperatorTest, TestCase, TensorSpec
from framework.runner import GenericTestRunner

# -------------------------------------------------------
# Core Utility Methods
# -------------------------------------------------------
def run_case_with_generic_runner(op_name, test_cases_input):
    """
    Reuse GenericTestRunner to run externally injected test cases.
    Supports both single TestCase object and list of TestCase objects.
    """
    
    # 1. Input Normalization: Ensure we always work with a list
    if isinstance(test_cases_input, list):
        cases_to_run = test_cases_input
    else:
        cases_to_run = [test_cases_input]

    # 2. Dynamically import target operator module (e.g., ops.add)
    module_path = f"ops.{op_name}"
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        print(f"❌ Module not found: {module_path}")
        return

    # 3. Find the original OpTest class
    OriginalOpTest = None
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, BaseOperatorTest) and obj is not BaseOperatorTest:
            OriginalOpTest = obj
            break
    
    if not OriginalOpTest:
        print("❌ OpTest class not found")
        return

    # 4. Dynamically define a subclass (Proxy Class)
    class ProxyOpTest(OriginalOpTest):
        def __init__(self):
            super().__init__() 

        def get_test_cases(self):
            # 🔥 Core: Return the list we prepared externally
            return cases_to_run

    print(f"🚀 Running {len(cases_to_run)} externally injected Case(s) for '{op_name}'...")
    
    # 5. Pass this proxy class to the Runner and run test
    runner = GenericTestRunner(ProxyOpTest)
    
    # Wrap in try-except because run_and_exit usually calls sys.exit()
    try:
        runner.run_and_exit()
    except SystemExit:
        pass


# -------------------------------------------------------
# Usage Example
# -------------------------------------------------------
if __name__ == "__main__":

    # Define common specs
    dtype = infinicore.float32
    
    # Case 1: Small Shape
    spec_small = TensorSpec.from_tensor((2, 4), None, dtype)
    case_1 = TestCase(
        inputs=[spec_small, spec_small],
        kwargs={},
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-5, "rtol": 1e-3},
        description="Custom Small Case"
    )

    # Case 2: Large Shape
    spec_large = TensorSpec.from_tensor((16, 16), None, dtype)
    case_2 = TestCase(
        inputs=[spec_large, spec_large],
        kwargs={},
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 1e-5, "rtol": 1e-3},
        description="Custom Large Case"
    )

    # -------------------------------------------------------
    # Scenario A: Run a single case
    # -------------------------------------------------------
    print("\n--- Scenario A: Single Case ---")
    run_case_with_generic_runner("add", case_1)

    # -------------------------------------------------------
    # Scenario B: Run multiple cases
    # -------------------------------------------------------
    print("\n--- Scenario B: Multiple Cases List ---")
    run_case_with_generic_runner("add", [case_1, case_2])
