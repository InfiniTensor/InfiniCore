import ctypes
import glob
import os
from typing import Iterable, List


def _candidate_prefixes(path: str) -> List[str]:
    """
    Return HPCC install prefixes to search for libs.
    Prefer HPCC_PATH; if absent and explicitly opted-in, fall back to /opt/hpcc.
    """
    prefixes: List[str] = []
    if path:
        prefixes.append(path)

    seen = set()
    unique: List[str] = []
    for p in prefixes:
        if p and p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def _try_load(paths: Iterable[str], name: str) -> bool:
    """Try to load a shared library from given paths or system search path."""
    for path in paths:
        full = os.path.join(path, "lib", name)
        if os.path.exists(full):
            try:
                ctypes.CDLL(full, mode=ctypes.RTLD_GLOBAL)
                return True
            except OSError:
                # Try next candidate
                continue
    # Last resort: rely on loader search path
    try:
        ctypes.CDLL(name, mode=ctypes.RTLD_GLOBAL)
        return True
    except OSError:
        return False


def preload_hpcc() -> None:
    """
    Best-effort preload of key HPCC runtime libs with RTLD_GLOBAL.

    This mirrors the behavior of torch's HPCC build that loads libtorch_global_deps.so,
    but avoids introducing a hard torch dependency. All failures are swallowed.
    """
    hpcc_path = os.getenv("HPCC_PATH")
    if not hpcc_path:
        return

    prefixes = _candidate_prefixes(hpcc_path)
    libs = [
        "libhcruntime.so",
        "libhcToolsExt.so",
        "libruntime_cu.so",
        "libhccompiler.so",
    ]

    for lib in libs:
        _try_load(prefixes, lib)


def _should_preload_device(device_type: str) -> bool:
    """
    Check if preload is needed for a specific device type.
    """
    device_env_map = {
        "METAX": ["HPCC_PATH", "INFINICORE_PRELOAD_HPCC"],  # HPCC/METAX
        # Add other device types here as needed:
        # "ASCEND": ["ASCEND_PATH"],
        # "CAMBRICON": ["NEUWARE_HOME"],
    }

    env_vars = device_env_map.get(device_type, [])
    for env_var in env_vars:
        if os.getenv(env_var):
            return True
    return False


def preload_device(device_type: str) -> None:
    """
    Preload runtime libraries for a specific device type if needed.

    Args:
        device_type: Device type name (e.g., "METAX", "ASCEND", etc.)
    """
    if device_type == "METAX":
        preload_hpcc()
    # Add other device preload functions here as needed:
    # elif device_type == "ASCEND":
    #     preload_ascend()
    # etc.


def preload_flash_attn_for_cpp_api() -> None:
    """
    Best-effort load of flash-attn's CUDA extension with RTLD_GLOBAL.

    ``libinfinicore_cpp_api.so`` (aten builds) may reference flash-attn symbols;
    loading via ``LD_PRELOAD`` breaks unrelated subprocesses (e.g. vLLM/Triton).
    """
    if os.environ.get("INFINICORE_DISABLE_FLASH_ATTN_RTLD_GLOBAL", "") == "1":
        return
    if os.environ.get("INFINILM_DISABLE_FLASH_ATTN_RTLD_GLOBAL", "") == "1":
        return

    candidates: List[str] = []
    try:
        import flash_attn

        base = os.path.dirname(flash_attn.__file__)
        parent = os.path.dirname(base)
        candidates.extend(glob.glob(os.path.join(parent, "flash_attn_2_cuda*.so")))
        candidates.extend(glob.glob(os.path.join(base, "flash_attn_2_cuda*.so")))
    except ImportError:
        pass

    # Typical wheel layout (image / CI); last resort after package discovery
    candidates.append(
        "/usr/local/lib/python3.12/dist-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so"
    )

    for fa in candidates:
        if fa and os.path.isfile(fa):
            try:
                ctypes.CDLL(fa, mode=ctypes.RTLD_GLOBAL)
                return
            except OSError:
                continue


def preload() -> None:
    """
    Universal preload function that loops through device types and preloads when required.

    This function detects available device types and preloads their runtime libraries
    if the environment indicates they are needed.
    """
    preload_flash_attn_for_cpp_api()

    # Device types that may require preload
    device_types = [
        "METAX",  # HPCC/METAX
        # Add other device types here as they are implemented:
        # "ASCEND",
        # "CAMBRICON",
        # etc.
    ]

    for device_type in device_types:
        if _should_preload_device(device_type):
            try:
                preload_device(device_type)
            except Exception:
                # Swallow all errors - preload is best-effort
                pass
