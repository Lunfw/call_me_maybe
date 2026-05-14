from torch import cuda
from typing import List, Optional, Any


class Models:
    def __init__(self) -> None:
        self.allowed: List[str] = Models.get_models()

    @staticmethod
    def get_models() -> List[str]:
        pass
