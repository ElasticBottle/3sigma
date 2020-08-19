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
        self.mapping = {}

    def build_vocab(self, labels: List):
        categories = set(labels)
        category_mapping = {}
        for index, category in enumerate(categories):
            category_mapping[category] = index if self.zero_index else index + 1
        return category_mapping

    def __call__(self, labels: List, is_valid_test=True):
        self.labels = labels
        if is_valid_test:
            assert len(self.mapping.keys()) != 0
        else:
            self.mapping = self.build_vocab(labels)
        self.encoded_labels = []
        for label in labels:
            self.encoded_labels.append(self.mapping[label])
        return self.encoded_labels


class OneHotEncoder(Encoder):
    pass


class DummyEncoder(Encoder):
    pass
