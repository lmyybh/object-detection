import cv2
import numpy as np

class DLIBProcessor:
    def __init__(self):
        pass

    def process(self, img):
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img
