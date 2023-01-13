import torchvision.transforms as T


class DETRProcessor:
    def __init__(self):
        pass

    def process(self, img):
        # standard PyTorch mean-std input image normalization
        transform = T.Compose(
            [
                T.Resize(800),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        img = transform(img).unsqueeze(0)

        return img
