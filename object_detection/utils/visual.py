import numpy as np
import matplotlib.pyplot as plt


# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def plot_results(pil_img, prob, boxes, classes):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    
    colors = {cls: np.random.rand(3) for cls in set(classes)}

    for p, (xmin, ymin, xmax, ymax), cls in zip(prob, boxes, classes):        
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=colors[cls], linewidth=3
            )
        )
        
        text = f"{cls}: {p:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.show()
