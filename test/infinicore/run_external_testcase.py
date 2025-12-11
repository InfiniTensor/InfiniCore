import sys
import os
import argparse
import time

from framework.testcase_manager import TestCaseManager

def main():
    parser = argparse.ArgumentParser(description="External Test Case Runner for InfiniCore")
    
    # Optional file path (if None, uses default add case)
    parser.add_argument("file_path", type=str, nargs="?", help="Path to JSON config file")
    
    # Overrides
    parser.add_argument("--device", type=str, default=None, help="Override target device (e.g. nvidia, cpu)")
    parser.add_argument("--bench", type=str, choices=["host", "device", "both"], default=None, help="Override benchmark mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--num_prerun", type=int, default=None, help="Override warmup iterations")
    parser.add_argument("--num_iterations", type=int, default=None, help="Override measured iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    parser.add_argument(
        "--save", 
        nargs="?", 
        const="test_report.json", 
        default=None, 
        help="Save test results to JSON. Default file: test_report.json"
    )
    
    args = parser.parse_args()

    final_save_path = args.save

    # Construct override dictionary
    # Filter out None/False values and specific keys not meant for override config
    override_dict = {
        k: v for k, v in vars(args).items()
        if k not in ["file_path"] and v is not None and v is not False
    }

    if override_dict:
        print(f"⚡ CLI Overrides detected: {override_dict}")

    # Run Manager
    manager = TestCaseManager()
    try:
        results = manager.run(
            json_file_path=args.file_path, 
            config=override_dict, 
            save_path=final_save_path
        )
        
        # Simple exit code logic based on results
        success = True
        failure_count = 0
        
        # Ensure results is a non-empty list
        if isinstance(results, list) and results:
            # results is a list of sub-lists (e.g., [[TestResult,...], [...]])
            for op_results_list in results:
                
                # op_results_list contains all TestResult objects for a specific Operator
                for res_obj in op_results_list:
                    # Check if it is a TestResult object (safety check)
                    if not hasattr(res_obj, 'success'):
                        # Skip if not a TestResult object
                        continue

                    # Access TestResult object attributes directly
                    if not res_obj.success:
                        success = False
                        failure_count += 1
                        
                        # Attempt to retrieve Operator name and Test Case description
                        case_desc = getattr(res_obj.test_case, 'description', 'No description')
                        error_msg = res_obj.error_message
                        
                        # Print clear error logs
                        print(f"❌ Failure detected:")
                        print(f"   - Case:    {case_desc}")
                        # Raw result object usually has device ID (no string name), fetching ID
                        print(f"   - Device:  {res_obj.device}") 
                        print(f"   - Error:   {error_msg if error_msg else 'Test failed but no specific error message provided.'}")

        if not success:
            print(f"\n⚠️  Test Suite Failed: {failure_count} case(s) failed.")
            sys.exit(1)
        
        print("\n✅ All tests passed successfully.")

    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n❌ Execution Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
