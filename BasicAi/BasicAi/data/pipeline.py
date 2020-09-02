from pathlib import Path
from typing import Callable, List, Tuple, Union, Any
from functools import partial

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

from BasicAi.BasicAi.data.dataset import BasicDataset, DataBunch
from BasicAi.BasicAi.data.item_list import ItemList
from BasicAi.BasicAi.data.extension_types import Extension
from BasicAi.BasicAi.data.file_opener import FileOpener
from BasicAi.BasicAi.data.splitter import Splitter
from BasicAi.BasicAi.data.labeller import Labeller
from BasicAi.BasicAi.utils import make_list, make_path


class Pipeline:
    def __init__(
        self,
        path: Union[Path, str],
        file_type: Extension,
        file_opener: FileOpener,
        splitter: Splitter,
        labeller: Labeller,
        to_exclude: Union[str, List[str]] = None,
        dataLoader: Callable[..., DataLoader] = partial(DataLoader, batch_size=64),
    ):
        super().__init__()
        self.path = make_path(path)
        self.to_exclude = make_list(to_exclude) if to_exclude is not None else None
        (
            self.file_type,
            self.file_opener,
            self.splitter,
            self.labeller,
            self.dataLoader,
        ) = (
            file_type,
            file_opener,
            splitter,
            labeller,
            dataLoader,
        )

    def create_dataloaders(self):
        self.train_dl = self.dataLoader(self.train, sampler=RandomSampler(self.train))
        self.valid_dl = self.dataLoader(
            self.valid, sampler=SequentialSampler(self.valid)
        )
        self.test_dl = self.dataLoader(self.test, sampler=SequentialSampler(self.test))

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
        return DataBunch(
            self.train_dl, self.valid_dl, c=len(set(self.train_dl.dataset.y))
        )
