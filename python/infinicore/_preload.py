import ctypes
import glob
import importlib
import importlib.util
import os
import sys
from typing import Iterable, List


_PRELOADED_HANDLES: List[ctypes.CDLL] = []


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


def preload_torch_hip() -> None:
    """
    Best-effort import of the torch HIP runtime.

    Loading torch's shared libraries individually with ctypes bypasses
    Python's extension initialization order and can corrupt HIP teardown.
    """
    try:
        importlib.import_module("torch")
    except (ImportError, OSError):
        pass


def _import_extension_global(module_name: str, path: str) -> None:
    module = sys.modules.get(module_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load extension module from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

    module_path = getattr(module, "__file__", None) or path
    _PRELOADED_HANDLES.append(
        ctypes.CDLL(module_path, mode=ctypes.RTLD_GLOBAL)
    )


def preload_flash_attn() -> None:
    """
    Best-effort preload of flash_attn_2_cuda extension with RTLD_GLOBAL.

    InfiniCore hygon wrapper resolves C symbols like `mha_varlen_fwd` from the
    flash-attn extension at runtime via dlsym(RTLD_DEFAULT, ...). The symbols
    only need to be available when the operator is actually called, not at
    library load time. So this preload is a convenience — if it fails, the
    symbols will be resolved later when torch + flash_attn are imported by
    the application (e.g. InfiniLM).
    """
    candidates: List[str] = []
    from_env = os.getenv("FLASH_ATTN_PREBUILT")
    if from_env:
        if os.path.isfile(from_env):
            candidates.append(from_env)
        elif os.path.isdir(from_env):
            candidates.extend(
                glob.glob(os.path.join(from_env, "flash_attn_2_cuda*.so"))
            )

    # Try resolving via Python import metadata.
    spec = importlib.util.find_spec("flash_attn_2_cuda")
    if spec and spec.origin and os.path.exists(spec.origin):
        candidates.append(spec.origin)

    # Fallback: scan python paths for extension module.
    for p in sys.path:
        if not p:
            continue
        candidates.extend(glob.glob(os.path.join(p, "flash_attn_2_cuda*.so")))

    # Common installation locations.
    candidates.extend(
        glob.glob("/usr/local/lib/python*/dist-packages/flash_attn_2_cuda*.so")
    )
    candidates.extend(glob.glob("/root/.infini/lib/flash_attn_2_cuda*.so"))

    seen = set()
    for so_path in candidates:
        if not so_path or so_path in seen:
            continue
        seen.add(so_path)
        if not os.path.exists(so_path):
            continue
        try:
            _import_extension_global("flash_attn_2_cuda", so_path)
            return
        except (ImportError, OSError):
            continue


def _should_preload_device(device_type: str) -> bool:
    """
    Check if preload is needed for a specific device type.
    """
    device_env_map = {
        "METAX": ["HPCC_PATH", "INFINICORE_PRELOAD_HPCC"],  # HPCC/METAX
        "HYGON": ["DTK_ROOT", "INFINICORE_PRELOAD_TORCH_HIP"],
        # Add other device types here as needed:
        # "ASCEND": ["ASCEND_PATH"],
        # "CAMBRICON": ["NEUWARE_HOME"],
    }

    env_vars = device_env_map.get(device_type, [])
    for env_var in env_vars:
        if os.getenv(env_var):
            return True
    if device_type == "HYGON":
        dtk_root = os.getenv("DTK_ROOT") or "/opt/dtk"
        if os.path.isdir(dtk_root):
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
    elif device_type == "HYGON":
        preload_torch_hip()
        preload_flash_attn()
    # Add other device preload functions here as needed:
    # elif device_type == "ASCEND":
    #     preload_ascend()
    # etc.


def preload() -> None:
    """
    Universal preload function that loops through device types and preloads when required.

    This function detects available device types and preloads their runtime libraries
    if the environment indicates they are needed.
    """
    # Device types that may require preload. Keep Hygon-only preloads gated by
    # Hygon environment markers so other CUDA-compatible platforms do not load
    # unrelated torch/flash-attn libraries during package import.
    device_types = [
        "METAX",  # HPCC/METAX
        "HYGON",
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
