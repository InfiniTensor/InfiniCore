import os


def _first_existing_dir(paths: list[str]) -> str:
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return ""


def _metax_toolkit_root(use_mc: bool) -> str:
    """Return toolkit root for MetaX builds (MACA when use-mc; otherwise HPCC)."""
    if use_mc:
        for key in ("MACA_PATH", "MACA_HOME", "MACA_ROOT"):
            v = os.environ.get(key, "").strip()
            if v:
                return v
        return _first_existing_dir(["/opt/maca"])
    return _first_existing_dir(["/opt/hpcc"])


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

    This chooses paths based on xmake backend flags (e.g. --metax-gpu) and toolkit selection
    (e.g. MetaX HPCC vs MACA when --use-mc=y).
    """
    d = parse_xmake_cli_flag_values(flags)
    if not truthy_flag_value(d.get("aten", "n")):
        return

    if truthy_flag_value(d.get("metax-gpu", "n")):
        use_mc = truthy_flag_value(d.get("use-mc", "n"))
        root = _metax_toolkit_root(use_mc=use_mc)
        if not root:
            return
        dirs = [
            os.path.join(root, "tools", "cu-bridge", "include"),
            os.path.join(root, "include", "hcr"),
            os.path.join(root, "include"),
        ]
        for var in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH"):
            _prepend_path_var(var, dirs)
        return

