#%%
import collections
import os

from pathlib import Path
from typing import Any, Callable, List, Set, Union

import numpy as np

from BasicAi.BasicAi.data.extension_types import Extension, TextExtension
from BasicAi.BasicAi.data.file_opener import CSVOpener, FileOpener
from BasicAi.BasicAi.utils import make_path


#%%
class ItemList(collections.UserList):
    def __init__(
        self, extension: Extension, file_opener: FileOpener, files: List[Path] = [],
    ):
        super().__init__(files)
        self.file_paths = np.array(files)
        self.extension_cls = extension
        self.extension = extension()
        self.open = file_opener

    def from_files(self, files: List[Path]):
        return ItemList(self.extension_cls, self.open, files)

    def _get_files(
        self,
        path: Union[str, Path],
        file_names: Union[str, List[str]],
        extension: Set[str],
    ) -> List[Path]:
        path = make_path(path)
        files = [
            path / file_name
            for file_name in file_names
            if file_name[file_name.rfind(".") :] in extension
        ]
        return np.array(files)

    def get_files(
        self, path: Path, recurse: bool = True, exclude: Union[None, List[str]] = None,
    ):
        assert isinstance(path, Path)
        if recurse:
            file_paths = np.array([])
            for path, _, files in os.walk(path):
                if (
                    exclude is not None
                    and not any(path.endswith(excluded_dir) for excluded_dir in exclude)
                ) or (exclude == None):
                    file_paths = np.append(
                        file_paths, self._get_files(path, files, self.extension)
                    )

            self.file_paths = file_paths
        else:
            files = [item.name for item in os.scandir(path) if item.is_file()]
            self.file_paths = self._get_files(path, files, self.extension)
        super().__init__(self.file_paths)

    # def compose(self, x, functions: List[Callable], order_key="_order"):
    #     key = lambda o: getattr(o, order_key, 0)
    #     for f in sorted(functions, key=key):
    #         x = f(x)
    #     return x

    def __getitem__(self, idx):
        res = self.file_paths[idx]
        if isinstance(res, np.ndarray):
            return [self.open(o) for o in res]
        return self.open(res)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"""{self.__class__.__name__} {len(self)} items:
{self.file_paths[:3]} {'...' if len(self) > 3 else ''}"""


#%%
import time

start = time.time()
path = Path(r"D:\Datasets\stock_data\1d")

items = ItemList(TextExtension(), file_opener=CSVOpener(dtype=None, index_col=0))
items.get_files(path / "train", recurse=True)
# length = []
# for i in range(len(items)):
#     try:
#         length.append(len(items[i]))
#     except:
#         length.append(float("inf"))
# print(min(length))
print(len(items))
end = time.time()
print(f"{end - start: .2f}")
