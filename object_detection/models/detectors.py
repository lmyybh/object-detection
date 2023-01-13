from .DETR import DETR
from .registry import DETECTORS


@DETECTORS.register("DETR")
def build_DETR():
    return DETR()


def build_detector(name):
    return DETECTORS[name]()
