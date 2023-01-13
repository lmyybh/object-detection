import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_results(pil_img, probs, boxes, classes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    colors = {cls: np.random.rand(3) for cls in set(classes)}

    for p, (xmin, ymin, xmax, ymax), cls in zip(probs, boxes, classes):        
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=colors[cls], linewidth=3
            )
        )
        
        text = f"{cls}: {p:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()

def plot_face_results(pil_img, faces):
    bbox_color = [0.000, 0.447, 0.741]
    landmarks_color = [0.933, 0.205, 0.301]

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    for face in faces:
        bbox, landmarks = face.items()
        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            ax.add_patch(
                plt.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=bbox_color, linewidth=2.5
                )
            )
        
        if landmarks.shape[0] > 0:
            plt.plot(landmarks[:, 0], landmarks[:, 1], 'o', color=landmarks_color, alpha=0.8, ms=1.5)

    plt.axis("off")
    plt.show()

def mark_image_faces(img, faces):
    bbox_color = [189, 114, 0]
    landmarks_color = [77, 52, 238]

    img = img.copy()

    for face in faces:
        bbox, landmarks = face.items()
        if len(bbox) == 4:
            cv2.rectangle(img, bbox[:2], bbox[2:], color=bbox_color, thickness=3)
        if landmarks.shape[0] > 0:
            for point in landmarks:
                cv2.circle(img, center=point, radius=2, color=landmarks_color, thickness=-1)

    return img