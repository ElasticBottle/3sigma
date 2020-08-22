import re
from pathlib import Path
from typing import Union, Any, Iterable


def make_list(to_convert: Any) -> list:
    if to_convert is None:
        return []
    elif isinstance(to_convert, list):
        return to_convert
    elif isinstance(to_convert, Iterable):
        return list(to_convert)
    return [to_convert]


# Code from https://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case

_camel_re1 = re.compile("(.)([A-Z][a-z]+)")
_camel_re2 = re.compile("([a-z0-9])([A-Z])")


def camel_to_snake(name):
    s1 = re.sub(_camel_re1, r"\1_\2", name)
    return re.sub(_camel_re2, r"\1_\2", s1).lower()


def make_path(path: Union[str, Path]) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)
