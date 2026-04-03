import os
import platform


def _hpcc_toolkit_root() -> str:
    """HPCC/MACA install root (cu-bridge, headers). Env vars first; else common container path."""
    for key in ("MACA_PATH", "MACA_HOME", "MACA_ROOT"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    if os.path.isdir("/opt/hpcc"):
        return "/opt/hpcc"
    return ""


def _prepend_path_var(name, prefixes):
    """Prepend colon-separated *prefixes* to env var *name* (POSIX)."""
    if not prefixes:
        return
    chunk = ":".join(prefixes)
    cur = os.environ.get(name, "")
    os.environ[name] = f"{chunk}:{cur}" if cur else chunk


def ensure_aten_torch_compiler_includes() -> None:
    """If HPCC root is known, prepend cu-bridge + HPCC headers for g++ compiling ATen .cc (c10/cuda)."""
    root = _hpcc_toolkit_root()
    if not root:
        return
    dirs = [
        os.path.join(root, "tools", "cu-bridge", "include"),
        os.path.join(root, "include", "hcr"),
        os.path.join(root, "include"),
    ]
    for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        _prepend_path_var(var, dirs)


def _parse_xmake_cli_flag_values(flags: str):
    """Parse a string like '--metax-gpu=y --aten=y' into {key: value}."""
    parts = flags.replace("=", " ").split()
    d = {}
    i = 0
    n = len(parts)
    while i < n:
        p = parts[i]
        if p.startswith("--") and len(p) > 2:
            key = p[2:].lower()
            i += 1
            if i < n and not parts[i].startswith("--"):
                d[key] = parts[i].lower()
                i += 1
            else:
                d[key] = "y"
        else:
            i += 1
    return d


def _truthy_flag_value(v: str) -> bool:
    return v in ("y", "yes", "true", "1", "on")


# xmake.lua GPU / accelerator backends (any of these + aten may compile C++ against torch+cuda-style headers).
_XMAKE_GPU_BACKEND_KEYS = frozenset(
    {
        "metax-gpu",
    }
)


def xmake_flags_need_aten_torch_compiler_includes(flags: str) -> bool:
    """True when ATen is enabled with any GPU/accelerator backend (install.py / xmake f ...)."""
    d = _parse_xmake_cli_flag_values(flags)
    if not _truthy_flag_value(d.get("aten", "n")):
        return False
    return any(_truthy_flag_value(d.get(k, "n")) for k in _XMAKE_GPU_BACKEND_KEYS)


def set_env():
    if os.environ.get("INFINI_ROOT") == None:
        os.environ["INFINI_ROOT"] = os.path.expanduser("~/.infini")

    if platform.system() == "Windows":
        new_path = os.path.expanduser(os.environ.get("INFINI_ROOT") + "/bin")
        if new_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{new_path};{os.environ.get('PATH', '')}"

    elif platform.system() == "Linux":
        new_path = os.path.expanduser(os.environ.get("INFINI_ROOT") + "/bin")
        if new_path not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{new_path}:{os.environ.get('PATH', '')}"

        new_lib_path = os.path.expanduser(os.environ.get("INFINI_ROOT") + "/lib")
        if new_lib_path not in os.environ.get("LD_LIBRARY_PATH", ""):
            os.environ["LD_LIBRARY_PATH"] = (
                f"{new_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
            )
    else:
        raise RuntimeError("Unsupported platform.")
