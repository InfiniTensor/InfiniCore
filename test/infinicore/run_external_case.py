import sys
import os
import argparse
import time

# ==============================================================================
# 🛠️ Path Adaptation
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from framework.testcase_manager import TestCaseManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="External Test Case Runner for InfiniCore")
    
    # Optional file path (if None, uses default add case)
    parser.add_argument("file_path", type=str, nargs="?", help="Path to JSON config file")
    
    # Overrides
    parser.add_argument("--device", type=str, default=None, help="Override target device (e.g. cuda, cpu)")
    parser.add_argument("--bench", type=str, choices=["host", "device", "both"], default=None, help="Override benchmark mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--num_prerun", type=int, default=None, help="Override warmup iterations")
    parser.add_argument("--num_iterations", type=int, default=None, help="Override measured iterations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # Save option
    parser.add_argument(
        "--save", 
        nargs="?", 
        const="AUTO", 
        default=None, 
        help="Path to save effective config JSON with results. If flag is used without value, generates 'test_case_<timestamp>.json'"
    )
    
    args = parser.parse_args()

    # Handle automatic save path generation
    final_save_path = args.save
    if final_save_path == "AUTO":
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        final_save_path = f"result_{timestamp}.json"

    # Construct override dictionary
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
                
                exec_results = entry.get("execution_results", [])
                for res in exec_results:
                    
                    status = res.get("status", {})
                    if not status.get("success", False):
                        success = False
                        print(f"❌ Failure detected: {status.get('error', 'Unknown error')}")
                        break
                
                if not success:
                    break
            
    except Exception as e:
        print(f"\n❌ Execution Error: {e}")
        sys.exit(1)
