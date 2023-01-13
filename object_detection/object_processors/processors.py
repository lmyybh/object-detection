from .DETR import DETRProcessor
from .registry import OBJECT_PROCESSORS

def build_object_processor(name, *args, **kwargs):
    return OBJECT_PROCESSORS[name](*args, **kwargs)

@OBJECT_PROCESSORS.register("DETR")
def build_DETR_processor(*args, **kwargs):
    return DETRProcessor(*args, **kwargs)

