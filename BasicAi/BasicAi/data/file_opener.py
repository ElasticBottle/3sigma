from pathlib import Path
from typing import Callable, Dict, List, Union

import pandas as pd
import PIL
import os

print(os.path)

from BasicAi.BasicAi.data.transforms import Transforms
from BasicAi.BasicAi.utils import make_list


class FileOpener:
    """
    Args:
        - **Kwargs: used to provide additional info needed to be passed onto the [open_function] that reads in the files from disk
    """

    def __init__(
        self, transforms=[], **kwargs,
    ):
        super().__init__()
        self.transforms = transforms
        self.kwargs = kwargs

    def open(self, path: Path, open_function: Callable):
        try:
            item = open_function(path, **self.kwargs)
            for transform in self.transforms:
                item = transform(item)
            return item
        except:
            print(path)
            raise IOError("something went wrong")


class CSVOpener(FileOpener):
    def __init__(
        self, transforms=[], dtype: Dict = None, **kwargs,
    ):
        transforms = make_list(transforms)
        super().__init__(transforms, dtype=dtype, **kwargs)

    def __call__(self, path: Path):
        return super().open(path, pd.read_csv)


class ImgOpener(FileOpener):
    def __init__(
        self, transforms=[], **kwargs,
    ):
        transforms = make_list(transforms)
        super().__init__(transforms, **kwargs)

    def __call__(self, path: Path):
        return super().open(path, PIL.Image.open)
