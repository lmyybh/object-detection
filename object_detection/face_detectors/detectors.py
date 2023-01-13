from .registry import FACE_DETECTORS
from .dlib import DLIB


def build_face_detector(name, *args, **kwargs):
    return FACE_DETECTORS[name](*args, **kwargs)

@FACE_DETECTORS.register("DLIB")
def build_DLIB(*args, **kwargs):
    return DLIB(*args, **kwargs)


