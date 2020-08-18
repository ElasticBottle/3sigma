from item_list import ItemList
from extension_types import TextExtension
from file_opener import CSVOpener
from pathlib import Path


class Splitter:
    pass


class SplitByGrandParentFolder(Splitter):
    def __init__(self, train: str = "train", valid: str = "valid", test: str = "test"):
        super().__init__()
        self.train, self.valid, self.test = train, valid, test

    def _generate_mask(self, mask: str, files: ItemList):
        mask_list = list(
            map(
                lambda file: True if file.parent.parent.name == mask else False,
                files.file_paths,
            )
        )
        print(mask_list[:10])
        return mask_list

    def __call__(self, files: ItemList):
        train = self._generate_mask(self.train, files)
        valid = self._generate_mask(self.valid, files)
        test = self._generate_mask(self.test, files)

        return tuple(
            map(
                files.from_files,
                [
                    files.file_paths[train],
                    files.file_paths[valid],
                    files.file_paths[test],
                ],
            )
        )


test = ItemList(TextExtension(), CSVOpener())
path = Path(r"D:\Datasets\stock_data\1d")
test.get_files(path)
splitter = SplitByGrandParentFolder()
items = splitter(test)
print(items)

