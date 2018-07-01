# Copyright      2018  Johns Hopkins University (author: Daniel Povey)


# Apache 2.0
from utils.data_types import validate_combined_image, validate_image_with_mask
import numpy as np
import cv2


def randomly_crop_combined_image(combined_image, config,
                                 image_height, image_width):
    """
    This function randomly crops a 'combined image' combined_image
    as would be validated by validate_combined_image(), to an image size
    of 'image_height' by 'image_width'.  It zero-pads if the current
    image is smaller than that.
    It returns the randomly cropped image; it doesn't modify
    'combined_image'.
    """
    validate_combined_image(combined_image, config)

    n_channels, height, width = combined_image.shape

    # it has been made a square image and we only consider one side
    if height <= image_height:
        cropped_image = np.pad(
            combined_image, ((0, 0), (0, image_height - height), (0, image_width - width)), 'constant')
    else:
        top = np.random.randint(0, height - image_height)
        left = np.random.randint(0, width - image_width)
        cropped_image = combined_image[:, top:top +
                                       image_height, left:left + image_width]

    validate_combined_image(cropped_image, config)

    return cropped_image


def randomly_crop_image_with_mask(image_with_mask, config,
                                  image_height, image_width):
    """
    This function randomly crops a 'image_with_mask' to an image size
    of 'image_height' by 'image_width'.  It zero-pads if the current
    image is smaller than that.
    It returns the randomly cropped image; it doesn't modify
    'combined_image'.
    """
    validate_image_with_mask(image_with_mask, config)

    cropped_image_with_mask = image_with_mask
    img = image_with_mask['img']
    mask = image_with_mask['mask']
    height, width, channels = img.shape

    if height < image_height:
        diff = image_height - height
        top_pad = int(diff / 2)
        bot_pad = diff - top_pad
        img = np.pad(img, ((top_pad, bot_pad), (0, 0), (0, 0)), 'constant')
        mask = np.pad(mask, ((top_pad, bot_pad), (0, 0)), 'constant')
    if width < image_width:
        diff = image_width - width
        left_pad = int(diff / 2)
        right_pad = diff - left_pad
        img = np.pad(img, ((0, 0), (left_pad, right_pad), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, 0), (left_pad, right_pad)), 'constant')
    else:
        top = np.random.randint(0, height - image_height)
        left = np.random.randint(0, width - image_width)
        cropped_image_with_mask['img'] = img[top:top +
                                             image_height, left:left + image_width, :]
        cropped_image_with_mask['mask'] = mask[top:top +
                                               image_height, left:left + image_width]

    return cropped_image_with_mask


def resize_to_square_image(image, image_size, preserve_ar=True, order=1):
    """ 
    This function resizes an image (1d or 3d numpy array) to a square image (height=width)
    with given size. 
    If 'preserve_ar' is True, the aspect ratio of the original image will be kept
    by zero-padding pixels equally at top and down (or left and right).
    'order' is the order of the spline interpolation, default is 1.
    """
    if preserve_ar:
        image = make_square_image_with_equal_padding(image)
    image = cv2.resize(image, (image_size, image_size))
    return image


def make_square_image_with_equal_padding(im_arr, pad_value=0):
    """
    This function pads an image to make it squre, if both height and width are
    different, (Otherwise it leaves it the same size).
    It returns the padded image; but note, if it does not have
    to pad the image, it just returns the input variable
    'image', it does not make a deep copy. Note: it pads equally on both sides.
    """

    dims = len(im_arr.shape)
    height = int(im_arr.shape[0])
    width = int(im_arr.shape[1])

    if width == height:
        return im_arr

    if width > height:
        diff = width - height
        top = int(diff / 2)
        bottom = diff - top
        if dims == 2:
            im_arr_pad = np.pad(
                im_arr, [(top, bottom), (0, 0)], mode='constant', constant_values=pad_value)
        else:
            im_arr_pad = np.pad(
                im_arr, [(top, bottom), (0, 0), (0, 0)], mode='constant', constant_values=pad_value)
    else:
        diff = height - width
        left = int(diff / 2)
        right = diff - left
        if dims == 2:
            im_arr_pad = np.pad(
                im_arr, [(0, 0), (left, right)], mode='constant', constant_values=pad_value)
        else:
            im_arr_pad = np.pad(
                im_arr, [(0, 0), (left, right), (0, 0)], mode='constant', constant_values=pad_value)

    return im_arr_pad
