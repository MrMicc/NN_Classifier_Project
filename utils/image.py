from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt


def process(image: str):
    # getting image from a file path
    img = Image.open(image)

    # resizing the image to 256px
    img = img.resize((256, 256))

    # crop image - left, upper, right, and lower pixel coordinate
    img = img.crop((0, 0, 224, 224))

    # geting color channels
    img = np.array(img) / 255

    # normalize with the data that was passed in the instructor note
    mean = np.array([0.485, 0.456, 0.406])
    deviation = np.array([0.229, 0.224, 0.225])
    np_image = (img - mean) / deviation

    # transpose image
    np_image = np_image.transpose((2, 0, 1))
    transpose_image = torch.from_numpy(np_image)

    return transpose_image


def show(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
