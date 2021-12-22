from typing import Any

class RemovableHandle:
    id: int
    next_id: int

    def __init__(self, hooks_dict: Any) -> None: ...
    def remove(self) -> None: ...
    def __enter__(self): ...
    def __exit__(self, type: Any, value: Any, tb: Any) -> None: ...
