import os


def _first_existing_dir(paths: list[str]) -> str:
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return ""


def _toolkit_root(env_names: tuple[str, ...], fallback: str) -> str:
    for key in env_names:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return _first_existing_dir([fallback])


def _prepend_path_var(name: str, prefixes: list[str]) -> None:
    """Prepend colon-separated *prefixes* to env var *name* (POSIX)."""
    if not prefixes:
        return
    chunk = ":".join(prefixes)
    cur = os.environ.get(name, "")
    os.environ[name] = f"{chunk}:{cur}" if cur else chunk


def set_env_for_metax_gpu(
    flags: str,
    *,
    parse_xmake_cli_flag_values,
    truthy_flag_value,
) -> None:
    """
    Prepend compiler include paths needed when building ATen-enabled C++ against torch headers.

    MetaX always uses the MACA SDK. Mars/HPCC is configured separately.
    """
    d = parse_xmake_cli_flag_values(flags)
    if not truthy_flag_value(d.get("aten", "n")):
        return

    if truthy_flag_value(d.get("metax-gpu", "n")):
        root = _toolkit_root(("MACA_PATH", "MACA_HOME", "MACA_ROOT"), "/opt/maca")
        if not root:
            return
        dirs = [
            os.path.join(root, "tools", "cu-bridge", "include"),
            os.path.join(root, "include", "hcr"),
            # cu-bridge cuComplex.h includes "hcComplex.h" from HPCC include/common
            os.path.join(root, "include", "common"),
            # cu-bridge cusparse wrapper includes "hcsparse.h" under include/hcsparse
            os.path.join(root, "include", "hcsparse"),
            # cu-bridge cublasLt wrapper includes "hcblasLt.h" under include/hcblas
            os.path.join(root, "include", "hcblas"),
            # cu-bridge cusolver wrapper includes "hcsolver_common.h" under include/hcsolver
            os.path.join(root, "include", "hcsolver"),
            os.path.join(root, "include"),
        ]
        for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
            _prepend_path_var(var, dirs)


def set_env_for_mars_gpu(
    flags: str,
    *,
    parse_xmake_cli_flag_values,
    truthy_flag_value,
) -> None:
    """Prepend HPCC compatibility headers for ATen-enabled Mars builds."""
    d = parse_xmake_cli_flag_values(flags)
    if not truthy_flag_value(d.get("aten", "n")):
        return
    if not truthy_flag_value(d.get("mars-gpu", "n")):
        return

    root = _toolkit_root(("HPCC_PATH", "HPCC_HOME"), "/opt/hpcc")
    if not root:
        return
    dirs = [
        os.path.join(root, "tools", "cu-bridge", "include"),
        os.path.join(root, "include", "hcr"),
        os.path.join(root, "include", "common"),
        os.path.join(root, "include", "hcsparse"),
        os.path.join(root, "include", "hcblas"),
        os.path.join(root, "include", "hcsolver"),
        os.path.join(root, "include"),
    ]
    for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
        _prepend_path_var(var, dirs)
