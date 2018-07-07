# Copyright      2018  Yiwen Shao

# Apache 2.0
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from scipy import ndimage
matplotlib.use('Agg')


def visualize_mask(img, mask, transparency=0.7, show_labels=True):
    """
    This function accepts a image, its mask and a float 0 < transparency < 1.
    It overlays the mask on image with transparency described by the parameter.
    """
    im = np.moveaxis(img, 0, -1)
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
    masked_img = Image.open(buffer_)
    buffer_.close()
    return masked_img
