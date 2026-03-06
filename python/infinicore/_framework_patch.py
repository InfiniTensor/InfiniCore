def _normalize_op_name(name: str) -> str:
    # "Log10" -> "log10", "AvgPool3d" -> "avgpool3d", "HistC" -> "histc"
    s = "".join(ch.lower() for ch in str(name) if ch.isalnum())
    return s


def _patch_framework_base(mod) -> bool:
    cls = getattr(mod, "BaseOperatorTest", None)
    if cls is None:
        return False

    if getattr(cls, "_infinicore_default_dispatch_installed", False):
        return True

    original = cls.infinicore_operator

    def dispatched(self, *args, **kwargs):
        import infinicore

        op = _normalize_op_name(getattr(self, "operator_name", ""))
        if op == "log10":
            return infinicore.log10(*args, **kwargs)
        if op == "log1p":
            return infinicore.log1p(*args, **kwargs)
        if op == "histc":
            return infinicore.histc(*args, **kwargs)
        if op == "dot":
            return infinicore.dot(*args, **kwargs)
        if op == "avgpool3d":
            return infinicore.nn.functional.avg_pool3d(*args, **kwargs)
        return original(self, *args, **kwargs)

    cls.infinicore_operator = dispatched
    cls._infinicore_default_dispatch_installed = True
    return True


def install_default_operator_dispatch() -> None:
    """
    The official benchmark runner under test/infinicore uses BaseOperatorTest.infinicore_operator.
    Many operator test files intentionally do not override infinicore_operator; they rely on a
    default implementation being available. We provide a default dispatcher by patching the
    framework class at runtime when it is present.
    """
    import importlib.abc
    import contextlib
    import sys
    import threading
    import time
    from importlib.machinery import PathFinder

    mod = sys.modules.get("framework.base")
    if mod is not None:
        if _patch_framework_base(mod):
            return

        # Handle circular-import timing: framework.base may already be importing and present
        # in sys.modules, but BaseOperatorTest isn't defined yet. In that case, schedule a
        # short-lived retry thread (no busy-poll) to patch once the class is available.
        if not getattr(mod, "__dict__", {}).get("BaseOperatorTest"):
            if getattr(mod, "_infinicore_dispatch_patch_scheduled", False):
                return
            setattr(mod, "_infinicore_dispatch_patch_scheduled", True)

            def _retry_patch() -> None:
                delay = 0.001
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    m = sys.modules.get("framework.base")
                    if m is not None and _patch_framework_base(m):
                        return
                    time.sleep(delay)
                    delay = min(delay * 2.0, 0.05)

            threading.Thread(
                target=_retry_patch,
                name="infinicore-framework-dispatch-patch",
                daemon=True,
            ).start()
            return

    class _FrameworkBaseHook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        def __init__(self) -> None:
            self._wrapped = None

        def find_spec(self, fullname, path, target=None):
            if fullname != "framework.base":
                return None
            spec = PathFinder.find_spec(fullname, path)
            if spec is None:
                return None
            self._wrapped = spec.loader
            spec.loader = self
            return spec

        def create_module(self, spec):
            if self._wrapped is not None and hasattr(self._wrapped, "create_module"):
                return self._wrapped.create_module(spec)
            return None

        def exec_module(self, module):
            if self._wrapped is None:
                raise ImportError("framework.base loader not available")
            self._wrapped.exec_module(module)
            _patch_framework_base(module)
            # Remove the hook once it has fired to avoid persistent import overhead.
            with contextlib.suppress(ValueError):
                sys.meta_path.remove(self)

    # Avoid installing duplicate hooks.
    for finder in sys.meta_path:
        if finder.__class__.__name__ == "_FrameworkBaseHook":
            return
    sys.meta_path.insert(0, _FrameworkBaseHook())
