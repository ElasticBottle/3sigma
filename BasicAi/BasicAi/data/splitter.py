from typing import Tuple
from BasicAi.BasicAi.data.item_list import ItemList
from BasicAi.BasicAi.data.extension_types import TextExtension
from BasicAi.BasicAi.data.file_opener import CSVOpener
from pathlib import Path


class Splitter:
    pass


class SplitByGrandParentFolder(Splitter):
    def __init__(self, train: str = "train", valid: str = "valid", test: str = "test"):
        super().__init__()
        self.train, self.valid, self.test = train, valid, test
        self.train_mask, self.valid_mask, self.test_mask = [], [], []

    def _generate_mask(self, mask: str, files: ItemList):
        mask_list = list(
            map(
                lambda file: True if file.parent.parent.name == mask else False,
                files.file_paths,
            )
        )
        return mask_list

    def __call__(self, files: ItemList) -> Tuple[ItemList, ItemList, ItemList]:
        self.train_mask = self._generate_mask(self.train, files)
        self.valid_mask = self._generate_mask(self.valid, files)
        self.test_mask = self._generate_mask(self.test, files)

        return tuple(
            map(
                files.from_files,
                [
                    files.file_paths[self.train_mask],
                    files.file_paths[self.valid_mask],
                    files.file_paths[self.test_mask],
                ],
            )
        )


# Todo(Eb): Add split by Rand percentage
# Todo(Eb): Add split by filename
# Todo(Eb): Add split by CSV
# Todo(Eb): Add split by parent folder (?)
