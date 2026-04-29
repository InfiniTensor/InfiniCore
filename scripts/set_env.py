import os
import platform

from metax_env import set_env_for_metax_gpu


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


def set_env_by_config(flags: str) -> None:
    """Set environment variables for InfiniCore builds with xmake config flags."""
    d = _parse_xmake_cli_flag_values(flags)
    if _truthy_flag_value(d.get("metax-gpu", "n")):
        set_env_for_metax_gpu(
            flags,
            parse_xmake_cli_flag_values=_parse_xmake_cli_flag_values,
            truthy_flag_value=_truthy_flag_value,
        )
    else:
        pass


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
