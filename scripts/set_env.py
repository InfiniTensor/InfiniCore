import os
import platform

def set_env():
    """
    Sets up environment variables for Infini software, supporting Windows, Linux, and macOS.
    """

    infini_root_env = "INFINI_ROOT"
    if os.environ.get(infini_root_env) is None:
        default_infini_root = os.path.expanduser("~/.infini")
        os.environ[infini_root_env] = default_infini_root
        print(f"[Info] '{infini_root_env}' was not set. Using default: {default_infini_root}")

    infini_root = os.environ.get(infini_root_env)
    if not infini_root:
        raise RuntimeError(f"Could not determine or set {infini_root_env}")

    bin_path = os.path.join(infini_root, "bin")
    lib_path = os.path.join(infini_root, "lib")

    # Update PATH environment variable
    path_env_var = "PATH"
    current_path = os.environ.get(path_env_var, "")
    path_list = [p for p in current_path.split(os.pathsep) if p]
    if bin_path not in path_list:
        new_path_value = f"{bin_path}{os.pathsep}{current_path}"
        os.environ[path_env_var] = new_path_value
        print(f"[Info] Added '{bin_path}' to {path_env_var}")

    # Handle platform-specific library path environment variables
    system = platform.system()

    # Linux platform
    if system == "Linux":
        lib_path_env_var = "LD_LIBRARY_PATH"
        current_lib_path = os.environ.get(lib_path_env_var, "")
        lib_path_list = [p for p in current_lib_path.split(os.pathsep) if p]
        if lib_path not in lib_path_list:
            new_lib_path_value = f"{lib_path}{os.pathsep}{current_lib_path}"
            os.environ[lib_path_env_var] = new_lib_path_value
            print(f"[Info] Added '{lib_path}' to {lib_path_env_var}")

    # macOS platform
    elif system == "Darwin":
        lib_path_env_var = "DYLD_LIBRARY_PATH"
        current_lib_path = os.environ.get(lib_path_env_var, "")
        lib_path_list = [p for p in current_lib_path.split(os.pathsep) if p]
        if lib_path not in lib_path_list:
            new_lib_path_value = f"{lib_path}{os.pathsep}{current_lib_path}"
            os.environ[lib_path_env_var] = new_lib_path_value
            print(f"[Info] Added '{lib_path}' to {lib_path_env_var}")

    # Windows platform
    elif system == "Windows":
        print(f"[Info] On Windows, ensure '{lib_path}' is accessible (e.g., via PATH).")

    # Unsupported platforms
    else:
        raise RuntimeError(f"Unsupported platform: {system}. Cannot set library path.")

