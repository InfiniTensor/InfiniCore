import os
import platform


def _maca_root_from_env():
    return (
        os.environ.get("MACA_PATH")
        or os.environ.get("MACA_HOME")
        or os.environ.get("MACA_ROOT")
        or ""
    ).strip()


def metax_hpc_compiler_include_dirs():
    """Directories needed so g++ finds cuda_runtime_api.h (cu-bridge) when compiling against PyTorch c10/cuda headers on MetaX/HPCC."""
    maca = _maca_root_from_env()
    if not maca:
        return []
    return [
        os.path.join(maca, "tools", "cu-bridge", "include"),
        os.path.join(maca, "include", "hcr"),
        os.path.join(maca, "include"),
    ]


def _prepend_path_var(name, prefixes):
    """Prepend colon-separated *prefixes* to env var *name* (POSIX)."""
    if not prefixes:
        return
    chunk = ":".join(prefixes)
    cur = os.environ.get(name, "")
    os.environ[name] = f"{chunk}:{cur}" if cur else chunk


def ensure_metax_hpc_compiler_includes():
    """
    Prepend HPCC/cu-bridge includes to CPATH, CPLUS_INCLUDE_PATH, and C_INCLUDE_PATH.
    g++ uses CPLUS_INCLUDE_PATH for .cc files; C_INCLUDE_PATH alone is not enough.
    """
    dirs = metax_hpc_compiler_include_dirs()
    if not dirs:
        return
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


def xmake_flags_need_metax_aten_torch_includes(flags: str) -> bool:
    """True when install.py-style args enable MetaX GPU and ATen (PyTorch) together."""
    d = _parse_xmake_cli_flag_values(flags)
    return _truthy_flag_value(d.get("metax-gpu", "n")) and _truthy_flag_value(
        d.get("aten", "n")
    )


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
