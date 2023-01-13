from .DETR import DETRProcessor
from .registry import PROCESSORS


@PROCESSORS.register("DETR")
def build_DETR_processor():
    return DETRProcessor()


def build_processor(name):
    return PROCESSORS[name]()
