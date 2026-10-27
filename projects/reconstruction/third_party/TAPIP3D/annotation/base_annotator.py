import numpy as np
from datasets.datatypes import RawSliceData
from typing import Dict

class BaseAnnotator:
    def __call__(self, data: RawSliceData) -> Dict[str, np.ndarray]:
        ...

    def to(self, device: str) -> None:
        ...
