"""Test bootstrap.

On a normal 3.10+ install `import kovio.*` just works. This robot only has
Python 3.8 (the package targets 3.10+ and its top-level __init__ uses APIs newer
than 3.8), so for verifying the *pure-logic* modules here we load them directly
by path with lightweight package stubs — no hardware, no heavy __init__.
"""
import importlib.util
import sys
import types
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_by_path(dotted: str) -> None:
    rel = dotted.split(".", 1)[1].replace(".", "/")
    path = SRC / "kovio" / (rel + ".py")
    spec = importlib.util.spec_from_file_location(dotted, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)


try:  # the happy path on a real 3.10+ environment
    import kovio.adapters.tracker  # noqa: F401
    import kovio.types  # noqa: F401
except Exception:
    # Stub the package namespaces so relative-free pure modules import cleanly.
    for name in ("kovio", "kovio.adapters"):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [str(SRC / name.replace(".", "/"))]
            sys.modules[name] = stub
    for dotted in (
        "kovio.types",
        "kovio.adapters.tracker",
        "kovio.adapters.gestures",
        "kovio.adapters.detectors",
        "kovio.adapters.lidar",
    ):
        try:
            _load_by_path(dotted)
        except FileNotFoundError:
            pass
