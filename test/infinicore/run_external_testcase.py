import sys
import os
import argparse
import time

# ==============================================================================
# 🛠️ Path Adaptation
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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
        if k not in ["file_path", "save"] and v is not None and v is not False
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
        if isinstance(results, list):
            for entry in results:
                
                cases = entry.get("testcases", [])
                for case in cases:
                    res = case.get("result", {})
                    status = res.get("status", {})
                    if not status.get("success", False):
                        success = False
                        print(f"❌ Failure detected: {status.get('error', 'Unknown error')}")
                        break
                
                if not success:
                    break
            
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n❌ Execution Error: {e}")
        sys.exit(1)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
