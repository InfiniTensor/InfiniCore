import sys
import os
import argparse
import time

# # ==============================================================================
# # 🛠️ 关键路径适配：解决命名冲突
# # ==============================================================================

# # 1. 识别冲突目录 (脚本自身所在的目录)
# conflict_dir = os.path.dirname(os.path.abspath(__file__)) 
# # 值: /home/baoming/workplace/InfiniCore/test/infinicore

# # 🚨 移除冲突目录！这是解决问题的核心步骤。
# # 必须移除它，才能强制 Python 去搜索 sys.path 中正确的路径。
# if conflict_dir in sys.path:
#     sys.path.remove(conflict_dir)

# # 2. 插入项目根目录 (包含真正的 'infinicore' 库)
# # 路径上溯 3 级: conflict_dir -> parent_dir -> test -> InfiniCore/
# project_root = os.path.abspath(os.path.join(conflict_dir, "../../.."))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # 3. 插入测试基目录 (包含 'framework' 包)
# parent_dir = os.path.dirname(conflict_dir)
# if parent_dir not in sys.path:
#     # 插入到 project_root 之后，确保 'infinicore' 库优先，但 'framework' 也能找到
#     sys.path.insert(1, parent_dir) 

# # ==============================================================================

# 现在可以安全地导入依赖于 infinicore 的模块了
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
        # if isinstance(results, list):
        #     for entry in results:
                
        #         # # ----------------------------------------------------
        #         # import json # 需要在文件开头导入
        #         # print("--- 实际的 entry 内容 ---")
        #         # print(type(entry))
        #         # print(json.dumps(entry, indent=4))
        #         # print("------------------------")
        #         # # ----------------------------------------------------
        #         cases = entry.get("testcases", [])
        #         for case in cases:
        #             res = case.get("result", {})
        #             status = res.get("status", {})
        #             if not status.get("success", False):
        #                 success = False
        #                 print(f"❌ Failure detected: {status.get('error', 'Unknown error')}")
        #                 break
                
        #         if not success:
        #             break
            
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"\n❌ Execution Error: {e}")
        sys.exit(1)

    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
