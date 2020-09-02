import mimetypes
from typing import Set


class Extension:
    def __init__(self, extensions: Set[str]):
        super().__init__()
        self.extension = extensions

    def __call__(self):
        return self.extension


class ImageExtension(Extension):
    def __init__(self):
        ex = set(k for k, v in mimetypes.types_map.items() if v.startswith("image/"))
        super().__init__(ex)


class TextExtension(Extension):
    def __init__(self):
        ex = set(k for k, v in mimetypes.types_map.items() if v.startswith("text/"))
        super().__init__(ex)


class AudioExtension(Extension):
    def __init__(self):
        ex = set(k for k, v in mimetypes.types_map.items() if v.startswith("audio/"))
        super().__init__(ex)


class VideoExtension(Extension):
    def __init__(self):
        ex = set(k for k, v in mimetypes.types_map.items() if v.startswith("video/"))
        super().__init__(ex)
