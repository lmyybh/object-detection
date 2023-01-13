from .DETR import DETR
from .registry import OBJECT_DETECTORS


@OBJECT_DETECTORS.register("DETR")
def build_DETR(*args, **kwargs):
    return DETR(*args, **kwargs)


def build_object_detector(name, *args, **kwargs):
    return OBJECT_DETECTORS[name](*args, **kwargs)
