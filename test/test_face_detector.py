import sys
sys.path.append('..')

from PIL import Image

from object_detection.face_processors import build_face_processor
from object_detection.face_detectors import build_face_detector
from object_detection.utils.visual import plot_face_results

processor = build_face_processor("DLIB")
detector = build_face_detector("DLIB", predictor_path='../models/shape_predictor_68_face_landmarks.dat')

img = Image.open("../data/group_photo.jpg")

inputs = processor.process(img)
faces = detector.detect_landmarks(inputs)

plot_face_results(img, faces)
