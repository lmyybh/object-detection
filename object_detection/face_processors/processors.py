from .dlib import DLIBProcessor
from .registry import FACE_PROCESSORS

def build_face_processor(name, *args, **kwargs):
    return FACE_PROCESSORS[name](*args, **kwargs)

@FACE_PROCESSORS.register("DLIB")
def build_DETR_processor(*args, **kwargs):
    return DLIBProcessor(*args, **kwargs)

