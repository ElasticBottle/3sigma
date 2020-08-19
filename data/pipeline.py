from pathlib import Path
from typing import Callable, List, Tuple, Union, Any

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from data.dataset import BasicDataset, DataBunch
from data.item_list import ItemList
from data.extension_types import Extension
from data.file_opener import FileOpener
from data.splitter import Splitter
from data.labeller import Labeller
from utils import make_list, make_path


class Pipeline:
    def __init__(
        self,
        path: Union[Path, str],
        file_type: Extension,
        file_opener: FileOpener,
        splitter: Splitter,
        labeller: Labeller,
        to_exclude: Union[str, List[str]] = None,
        batch_size: int = 64,
    ):
        super().__init__()
        self.path = make_path(path)
        self.to_exclude = make_list(to_exclude) if to_exclude is not None else None
        (
            self.file_type,
            self.file_opener,
            self.splitter,
            self.labeller,
            self.batch_size,
        ) = (
            file_type,
            file_opener,
            splitter,
            labeller,
            batch_size,
        )

    def create_dataloaders(self):
        self.train_dl = DataLoader(
            self.train, sampler=RandomSampler(self.train), batch_size=self.batch_size,
        )
        self.valid_dl = DataLoader(
            self.valid,
            sampler=SequentialSampler(self.valid),
            batch_size=self.batch_size,
        )
        self.test_dl = DataLoader(
            self.test, sampler=SequentialSampler(self.test), batch_size=self.batch_size,
        )

    def run(self) -> DataBunch:
        self.files = ItemList(self.file_type, file_opener=self.file_opener)
        self.files.get_files(self.path, exclude=self.to_exclude)

        # Creating input and output
        self.train_x, self.valid_x, self.test_x = self.splitter(self.files)
        self.train_y, self.valid_y, self.test_y = self.labeller(
            self.train_x, self.valid_x, self.test_x,
        )

        # Creating datasets
        self.train = BasicDataset(self.train_x, self.train_y)
        self.valid = BasicDataset(self.valid_x, self.valid_y)
        self.test = BasicDataset(self.test_x, self.test_y)

        # Create DataLoaders
        self.create_dataloaders()

        # Tying everything together into a group
        return DataBunch(self.train_dl, self.valid_dl, c=len(set(self.train_dl.dataset.y)))
