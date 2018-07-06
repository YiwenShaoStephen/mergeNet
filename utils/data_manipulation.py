# Copyright      Yiwen Shao

# Apache 2.0

import numpy as np
from utils.data_types import validate_config, validate_image_with_mask, validate_combined_image


def convert_to_combined_image(x, c):
    """ This function processes an 'image-with-mask' x into a 'combined' image,
    containing both input and supervision information in a single numpy array.
    see 'validate_combined_image' in data_types.py for a description of what
    a combined image is.

    This function returns the 'combined' image; it does not modify x.

    The width of the resulting image will be the same as the image in x:
    this function doesn't do padding, you need to call pad_combined_image.
    """
    validate_config(c)
    validate_image_with_mask(x, c)
    # x['img'] is of size (height, width, color), switch it to (color, height, width)
    im = np.moveaxis(x['img'], -1, 0)
    im = im.astype('float32') / 256.0
    mask = x['mask']
    _, height, width = im.shape
    object_class = x['object_class']
    num_outputs = c.num_colors + c.num_classes + len(c.offsets)
    y = np.ndarray(
        shape=(num_outputs, height, width), dtype='float32')

    y[:c.num_colors, :, :] = im

    # map object_id to class_id
    def obj_to_class(x):
        return object_class[x]
    class_mask = np.array([[obj_to_class(pixel)
                            for pixel in row] for row in mask])
    for n in range(c.num_classes):
        class_feature = (class_mask == n).astype('float32')
        y[c.num_colors + n, :, :] = class_feature

    for k, (i, j) in enumerate(c.offsets):
        rolled_mask = np.roll(np.roll(mask, -i, axis=0), -j, axis=1)
        offset_feature = (rolled_mask == mask).astype('float32')
        y[c.num_colors + c.num_classes + k, :, :] = offset_feature

    validate_combined_image(y, c)
    return y


def prepare_for_combined_image(x, c):
    validate_config(c)
    validate_image_with_mask(x, c)
    # x['img'] is of size (height, width, color), switch it to (color, height, width)
    im = np.moveaxis(x['img'], -1, 0)
    mask = x['mask']
    _, height, width = im.shape
    object_class = x['object_class']
    num_outputs = c.num_classes + len(c.offsets)
    y = np.ndarray(
        shape=(num_outputs, height, width), dtype='bool')

    # map object_id to class_id
    def obj_to_class(x):
        return object_class[x]
    class_mask = np.array([[obj_to_class(pixel)
                            for pixel in row] for row in mask])
    for n in range(c.num_classes):
        class_feature = (class_mask == n)
        y[n, :, :] = class_feature

    for k, (i, j) in enumerate(c.offsets):
        rolled_mask = np.roll(np.roll(mask, -i, axis=0), -j, axis=1)
        offset_feature = (rolled_mask == mask)
        y[c.num_classes + k, :, :] = offset_feature

    return im, y


def combine_img_and_feature(img, feature, c):
    validate_config(c)
    _, height, width = img.shape
    num_outputs = c.num_colors + c.num_classes + len(c.offsets)
    combined_img = np.ndarray(
        shape=(num_outputs, height, width), dtype='float32')
    combined_img[:c.num_colors, :, :] = img.astype('float32') / 256

    combined_img[c.num_colors:, :, :] = feature.astype('float32')
    validate_combined_image(combined_img, c)
    return combined_img
