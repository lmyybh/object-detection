import os
import dlib
import numpy as np

from .face import Face

def convert_dlib_rect_to_bbox(rect):
    return np.array([rect.left(), rect.top(), rect.right(), rect.bottom()])

def convert_dlib_detection_to_landmarks(shape):
    points = []
    for point in shape.parts():
        points.append([point.x, point.y])
    return np.array(points)

class DLIB:
    def __init__(self, predictor_path):
        assert os.path.exists(predictor_path), f"{predictor_path} doesn't exist."
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def detect_faces(self, img):
        rects = self.detector(img, 1)

        faces = []
        for rect in rects:
            bbox = convert_dlib_rect_to_bbox(rect)
            faces.append(Face(bbox))
        return faces
    
    def detect_landmarks(self, img, rects=None):
        if rects is None:
            rects = self.detector(img, 1)
        
        shapes = [self.predictor(img, rect) for rect in rects]

        faces = []
        for rect, shape in zip(rects, shapes):
            bbox = convert_dlib_rect_to_bbox(rect)
            landmarks = convert_dlib_detection_to_landmarks(shape)
            faces.append(Face(bbox, landmarks))
        
        return faces

