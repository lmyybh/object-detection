from PIL import Image

from object_detection.models.detectors import build_detector
from object_detection.processor.processors import build_processor
from object_detection.utils.visual import plot_results
from object_detection.utils.classes import COCO_CLASSES

processor = build_processor("DETR")
model = build_detector("DETR")

img = Image.open("./data/zebra.jpg")

inputs = processor.process(img)
outputs = model.predict(inputs, img_size=img.size, threshold=0.9)

plot_results(img, outputs["probs"], outputs["boxes"], outputs["classes"])
