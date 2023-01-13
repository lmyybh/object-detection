import sys
sys.path.append('..')

from PIL import Image

from object_detection.object_detectors import build_object_detector
from object_detection.object_processors import build_object_processor
from object_detection.utils.visual import plot_results

processor = build_object_processor("DETR")
detector = build_object_detector("DETR")

img = Image.open("../data/pet.jpg")

inputs = processor.process(img)
outputs = detector.predict(inputs, img_size=img.size, threshold=0.9)

plot_results(img, outputs["probs"], outputs["boxes"], outputs["classes"])
