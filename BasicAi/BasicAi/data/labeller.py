from typing import Tuple, List
from BasicAi.BasicAi.data.item_list import ItemList
from BasicAi.BasicAi.data.label_list import LabelList
from BasicAi.BasicAi.data.encoder import Encoder


class Labeller:
    pass


class LabelByParentFolder(Labeller):
    """
    Generate label for the inputs based of the parent name of the folders
    For use with categorical tasks
    """

    def __init__(self, encoder: Encoder):
        super().__init__()
        self.encoder = encoder

    def label_by_parent_folder(self, data: ItemList):
        result = []
        for file in data.file_paths:
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
        en_train_label, mappings = self.encoder(train_label, is_valid_test=False)
        train_ll = LabelList(train_label, en_train_label, encoding=mappings)

        valid_label = self.label_valid(valid)
        en_valid_label, _ = self.encoder(valid_label)
        valid_ll = LabelList(valid_label, en_valid_label, encoding=mappings)

        test_label = self.label_test(test)
        en_test_label, _ = self.encoder(test_label)
        test_ll = LabelList(test_label, en_test_label, encoding=mappings)

        return (train_ll, valid_ll, test_ll)


# Todo(eb): label by Filename
# Todo(eb): label by csv file
# Todo(eb): label by function (?)
