import numpy as np

def convert_to_array(data):
    if not isinstance(data, np.ndarray):
        return np.array(data)
    return data

class Face:
    def __init__(self, bbox=[], landmarks=[]):
        self.bbox = convert_to_array(bbox)
        self.landmarks = convert_to_array(landmarks)
    
    def items(self):
        return self.bbox, self.landmarks
