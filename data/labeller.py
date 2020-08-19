from typing import Tuple, List
from item_list import ItemList


class Labeller:
    pass


class LabelByParentFolder(Labeller):
    """
    Generate label for the inputs based of the parent name of the folders
    For use with categorical tasks
    """

    def __init__(self):
        super().__init__()

    def label_by_parent_folder(self, data: ItemList):
        result = []
        for file in data.file_path:
            label = file.parent.name
            result.append(label)
        return result

    def label_train(self, train: ItemList):
        return self.label_by_parent_folder(train)

    def label_valid(self, valid: ItemList):
        return self.label_by_parent_folder(valid)

    def label_test(self, test: ItemList):
        return self.label_by_parent_folder(test)

    def __call__(
        self, train: ItemList, valid: ItemList, test: ItemList
    ) -> Tuple[List, List, List]:
        train_label = self.label_train(train)
        valid_label = self.label_valid(valid)
        test_label = self.label_test(test)
        return (train_label, valid_label, test_label)


# Todo(eb): label by Filename
# Todo(eb): label by csv file
# Todo(eb): label by function (?)

from typing import List


class Encoder:
    pass


class OrdinalEncoder(Encoder):
    """
    Maps a list of categorical vairables to numbers.
    """

    def __init__(self, zero_index: bool = True):
        super().__init__()
        self.zero_index = zero_index

    def build_vocab(self, labels: List):
        categories = set(labels)
        category_mapping = {}
        for index, category in enumerate(categories):
            category_mapping[category] = index if self.zero_index else index + 1
        return category_mapping

    def __call__(self, labels):
        self.labels = labels
        self.mapping = self.build_vocab(labels)
        self.encoded_labels = []
        for label in labels:
            self.encoded_labels.append(self.mapping[label])
        return self.encoded_labels


class OneHotEncoder(Encoder):
    pass


class DummyEncoder(Encoder):
    pass
