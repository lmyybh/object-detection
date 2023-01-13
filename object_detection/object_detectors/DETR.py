import torch

from object_detection.utils.classes import COCO_CLASSES

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


class DETR:
    def __init__(self):
        self.model = torch.hub.load(
            "facebookresearch/detr", "detr_resnet50", pretrained=True
        )
        self.model.eval()

    def __call__(self, img: torch.Tensor):
        return self.model(img)

    def predict(self, img: torch.Tensor, img_size, threshold=0.9):
        outputs = self.model(img)
        probs = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        keep = probs.max(-1).values >= threshold

        probs = probs[keep]
        # convert boxes from [0; 1] to image scales
        bboxes = rescale_bboxes(outputs["pred_boxes"][0, keep], img_size).tolist()

        indexs = probs.argmax(dim=1)
        probs = probs[range(probs.shape[0]), indexs].tolist()
        classes = [COCO_CLASSES[idx] for idx in indexs]

        return dict(probs=probs, classes=classes, boxes=bboxes)
