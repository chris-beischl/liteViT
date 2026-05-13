import importlib
from typing import Any


def resolve(full_path: str) -> Any:
    module_path, attr = full_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr)
