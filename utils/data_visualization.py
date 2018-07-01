# Copyright      2018  Yiwen Shao

# Apache 2.0
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from scipy import ndimage
from utils.data_types import validate_image_with_mask
matplotlib.use('Agg')


def visualize_mask(x, c, transparency=0.7, show_labels=True):
    """
    This function accepts an object x that should represent an image with a
    mask, a config class c, and a float 0 < transparency < 1.
    It changes the image in-place by overlaying the mask with transparency
    described by the parameter.
    x['img_with_mask'] = image with transparent mask overlay
    """
    validate_image_with_mask(x, c)
    im = x['img']
    mask = x['mask']
    plt.clf()
    plt.imshow(im)
    for i in range(1, mask.max() + 1):
        b_mask = (mask == i)
        base_img = np.ones((b_mask.shape[0], b_mask.shape[1], 3))
        color = np.random.random((1, 3)).tolist()[0]
        for k in range(3):
            base_img[:, :, k] = color[k]
        plt.imshow(np.dstack((base_img, b_mask * transparency)))
        if show_labels:
            center = np.round(ndimage.measurements.center_of_mass(b_mask))
            plt.text(center[1] - 2, center[0] + 2, '{}'.format(i), fontsize=7,
                     color=color, bbox=dict(facecolor='white',
                                            edgecolor='none', pad=0))

    plt.subplots_adjust(0, 0, 1, 1)
    buffer_ = BytesIO()
    plt.savefig(buffer_, format="png")
    buffer_.seek(0)
    image = Image.open(buffer_)
    buffer_.close()
    return image
