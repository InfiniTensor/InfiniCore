import os
import subprocess
import platform
import sys
from set_env import set_env

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)
print(f"Changed directory to: {PROJECT_DIR}")

def run_cmd(cmd):
    """执行命令并添加打印和错误处理"""
    print(f"\nExecuting: {cmd}")
    try:
        subprocess.run(cmd, text=True, encoding="utf-8", check=True, shell=True)
        print(f"Successfully executed: {cmd}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with return code {e.returncode}: {cmd}")
        raise e
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred while running: {cmd}")
        print(f"Error details: {e}")
        raise e

def install(user_config_flags_str=""):
    """配置、编译和安装 InfiniCore"""

    base_flags = ["-c", "-v"]
    user_flags_list = user_config_flags_str.split()
    platform_flags = []

    if platform.system() == "Darwin":
        print("[Info] Detected macOS platform.")

        # 1. 禁用 OpenMP
        # 检查用户是否明确要求开启 omp
        user_wants_omp = any(flag == "--omp=y" for flag in user_flags_list)
        if user_wants_omp:
            print("[Info] User explicitly requested '--omp=y'. Enabling OpenMP (compiler support required).")
        else:
            # 检查用户是否已明确要求关闭 omp
            user_disabled_omp = any(flag == "--omp=n" for flag in user_flags_list)
            if not user_disabled_omp:
                print("[Info] Defaulting to '--omp=n' on macOS to avoid compiler errors.")
                # 避免重复添加
                if "--omp=n" not in platform_flags and "--omp=n" not in user_flags_list:
                    platform_flags.append("--omp=n")
            else:
                 print("[Info] User explicitly requested '--omp=n'. Disabling OpenMP.")

    # 合并标志
    final_config_flags_list = platform_flags + user_flags_list + base_flags
    # 简单的去重
    final_config_flags_list = sorted(list(set(final_config_flags_list)), key=final_config_flags_list.index)
    xmake_config_command = ["xmake", "f"] + final_config_flags_list

    try:
        run_cmd(" ".join(xmake_config_command)) # 组合回字符串以适应 shell=True
        run_cmd("xmake")
        run_cmd("xmake install")
        print("\nBuilding test target 'infiniop-test'...")
        run_cmd("xmake build infiniop-test")
        print("\nInstalling test target 'infiniop-test'...")
        run_cmd("xmake install infiniop-test")
        print("\n[SUCCESS] InfiniCore installation process completed successfully.")
    except Exception as e:
        print(f"\n[FAILED] Installation process encountered an error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("--- Starting InfiniCore Installation Script ---")
    print("Setting up environment variables...")
    set_env()
    print("Environment setup complete.")

    user_provided_flags = " ".join(sys.argv[1:])
    print(f"User provided flags: '{user_provided_flags}'")
    install(user_provided_flags)
    print("--- InfiniCore Installation Script Finished ---")
