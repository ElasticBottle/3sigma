from typing import Callable, List, Dict
import collections


class LabelList(collections.UserList):
    """
    Keeps track of both the original Labels and the encoded labels for a classification task
    """

    def __init__(self, labels: List, encoded_labels: List, encoding: Dict = {}):
        super().__init__(labels)
        assert len(labels) == len(encoded_labels)
        self.labels = labels
        self.en_labels = encoded_labels
        self.encoding = encoding

    @property
    def encoding(self):
        return self.__encoding

    @encoding.setter
    def encoding(self, value: Dict):
        self.__encoding = value

    def __getitem__(self, index):
        return self.en_labels[index]

    def __repr__(self):
        length = len(self.labels)
        labels = set(self.labels)
        if len(labels) > 30:
            res = []
            for i, v in enumerate(labels):
                if i > 30:
                    break
                res.append(v)
            labels = res
        return f"""{self.__class__.__name__}: {length}
{self.labels if length < 5 else self.labels[:5]}...
{len(labels)} unique labels
Labels: {labels}.
Mapping: {self.encoding}"""

