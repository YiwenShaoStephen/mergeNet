#!/usr/bin/env python3

# Copyright      2018  Yiwen Shao

# Apache 2.0

import torch
import torch.utils.data
import torch.nn.functional as F
import argparse
import os
import pickle
import random
import numpy as np
import scipy.misc
from models import fcn_resnet, fcn_vgg16
from utils.segmenter import ObjectSegmenter, SegmenterOptions
from skimage.transform import resize
from utils.core_config import CoreConfig
from utils.data_visualization import visualize_mask
from utils.dataset import COCODataset, COCOTestset
from pycocotools import mask as maskUtils

parser = argparse.ArgumentParser(
    description='Pytorch COCO instance segmentation setup')
parser.add_argument('--img', type=str, required=True,
                    help='Directory of test images')
parser.add_argument('--ann', type=str, required=True,
                    help='Path to test annotation or info file')
parser.add_argument('--dir', type=str, required=True,
                    help='Experiment directory which contains config, model'
                    'and the output result of this script')
parser.add_argument('--segment', type=str, default='segment',
                    help='sub dir name under <args.dir> that segmentation results'
                    'will be stored to')
parser.add_argument('--arch', default='fcn16s', type=str,
                    help='model architecture')
parser.add_argument('--model', type=str, default='model_best.pth.tar',
                    help='Name of the model file to use for segmenting.')
parser.add_argument('--mode', type=str, default='val', choices=['val', 'oracle'],
                    help='Input type of segmenter. val mode will use the output of model'
                    'as the input of segmenter; oracle mode will use the ground'
                    'truth target as the input of segmenter.')
parser.add_argument('--limits', default=None, type=int,
                    help="If given, is the size of subset we do segmenting on")
parser.add_argument('--train-image-size', default=128, type=int,
                    help='The size of the parts of training images that we'
                    'train on (in order to form a fixed minibatch size).')
parser.add_argument('--seg-size', default=128, type=int,
                    help='test images will be resized to and then do segmentation on.'
                    'Note: segmentation results will be resized to its original size for'
                    'future evaluation.')
parser.add_argument('--object-merge-factor', type=float, default=None,
                    help='Scale for object merge scores in the segmentaion '
                    'algorithm. If not set, it will be set to '
                    '1.0 / num_offsets by default.')
parser.add_argument('--same-different-bias', type=float, default=0.0,
                    help='Bias for same/different probs in the segmentation '
                    'algorithm.')
parser.add_argument('--merge-logprob-bias', type=float, default=0.0,
                    help='A bias that is added to merge logprobs in the '
                    'segmentation algorithm.')
parser.add_argument('--prune-threshold', type=float, default=0.0,
                    help='Threshold used in the pruning step of the '
                    'segmentation algorithm. Higher values --> more pruning.')
parser.add_argument('--visualize', type=bool, default=True,
                    help='Whether to store segmentation results as images to disk')
parser.add_argument('--job', type=int, default=0, help='job id')
parser.add_argument('--num-jobs', type=int, default=1,
                    help='number of parallel jobs')
random.seed(0)
np.random.seed(0)


def main():
    global args
    args = parser.parse_args()
    args.batch_size = 1  # only segment one image for experiment

    core_config_path = os.path.join(args.dir, 'configs/core.config')

    core_config = CoreConfig()
    core_config.read(core_config_path)
    print('Using core configuration from {}'.format(core_config_path))

    offset_list = core_config.offsets
    print("offsets are: {}".format(offset_list))

    # model configurations from core config
    num_classes = core_config.num_classes
    num_colors = core_config.num_colors
    num_offsets = len(core_config.offsets)

    # if exists, class_nms is the list of class names we are going to detect
    class_nms_file = os.path.join(args.dir, 'configs/subclass.txt')
    if os.path.exists(class_nms_file):
        with open(class_nms_file, 'r') as fh:
            class_nms = fh.readline().split()
            print('Segmenting on {} classes: {}'.format(
                len(class_nms), class_nms))
    else:
        class_nms = None
        print('Segmenting on all classes.')

    # dataset
    if 'train' or 'val' in args.ann_data:
        testset = COCODataset(args.img, args.ann, core_config,
                              mode=args.mode,
                              class_nms=class_nms, limits=args.limits,
                              job=args.job, num_jobs=args.num_jobs)
    else:
        testset = COCOTestset(args.img, args.ann, core_config)
    print('Total samples in the dataset to be segmented: {0}'.format(
        len(testset)))

    catIds = testset.catIds

    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

    # load model
    if args.mode == 'val':
        # model
        valid_archs = ['fcn{}_resnet{}'.format(x, y)
                       for x in [8, 16, 32] for y in [18, 34, 50, 101, 152]]
        valid_archs += ['fcn{}_vgg16'.format(x) for x in [8, 16, 32]]
        valid_archs += ['unet']
        if args.arch not in valid_archs:
            raise ValueError('Supported models are: {} \n'
                             'but given {}'.format(valid_archs, args.arch))
        if args.arch == 'unet':
            model = UNet(num_classes, num_offsets)
        elif 'vgg16' in args.arch:
            names = args.arch.split('_')
            scale = int(names[0][3:])
            model = fcn_vgg16(num_classes + num_offsets,
                              scale=scale)
        elif 'resnet' in args.arch:
            names = args.arch.split('_')
            scale = int(names[0][3:])
            layer = int(names[1][6:])
            model = fcn_resnet(num_classes + num_offsets,
                               scale=scale, layer=layer)

        model_path = os.path.join(args.dir, args.model)
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            print("loaded.")
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(model_path))
    else:
        model = None  # we don't need model in oracle mode

    segment_dir = os.path.join(args.dir, args.segment)
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)
    segment(dataloader, segment_dir, model, core_config, catIds)


def segment(dataloader, segment_dir, model, core_config, catIds):
    if model:
        model.eval()  # convert the model into evaluation mode
    img_dir = os.path.join(segment_dir, 'img')
    pkl_dir = os.path.join(segment_dir, 'pkl')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    exist_ids = next(os.walk(pkl_dir))[2]

    num_classes = core_config.num_classes
    offset_list = core_config.offsets

    for i, vals in enumerate(dataloader):
        image_id = vals[0][0].item()
        img = vals[1]
        size = vals[2]
        original_height, original_width = size[0].item(), size[1].item()
        if str(image_id) + '.pkl' in exist_ids:
            continue
        if len(vals) == 3:  # 'val' mode, using model output to do segmentation
            with torch.no_grad():
                img_input = F.upsample(img,
                                       size=(args.train_image_size,
                                             args.train_image_size * 2),
                                       mode='bilinear', align_corners=True)
                output = model(img_input)
                output = F.upsample(output, size=(args.seg_size, args.seg_size * 2),
                                    mode='bilinear', align_corners=True)
                class_pred = output[:, :num_classes, :, :]
                adj_pred = output[:, num_classes:, :, :]
        elif len(vals) == 5:  # 'oracle' mode, using ground truth label to do segmentation
            class_pred = vals[3]
            adj_pred = vals[4]
            class_pred = F.upsample(class_pred, size=(args.seg_size, args.seg_size * 2),
                                    mode='bilinear', align_corners=True)
            adj_pred = F.upsample(adj_pred, size=(args.seg_size, args.seg_size * 2),
                                  mode='bilinear', align_corners=True)
            import torchvision
            for i in range(10):
                torchvision.utils.save_image(
                    adj_pred[:, i:i + 1, :, :], '{}/bound_{}.png'.format(segment_dir, i))

        if args.object_merge_factor is None:
            args.object_merge_factor = 1.0 / len(offset_list)
            segmenter_opts = SegmenterOptions(same_different_bias=args.same_different_bias,
                                              object_merge_factor=args.object_merge_factor,
                                              merge_logprob_bias=args.merge_logprob_bias)
        seg = ObjectSegmenter(class_pred[0].detach().numpy(),
                              adj_pred[0].detach().numpy(),
                              num_classes, offset_list,
                              segmenter_opts)
        mask_pred, object_class = seg.run_segmentation()

        # resize the mask back to the original image size
        mask_pred = resize(mask_pred, (original_height, original_width),
                           order=0, preserve_range=True).astype(int)
        image_with_mask = {}
        image_with_mask['img'] = np.moveaxis(img[0].detach().numpy(), 0, -1)
        image_with_mask['mask'] = mask_pred
        image_with_mask['object_class'] = object_class

        # store segmentation result as image
        if args.visualize:
            visual_mask = visualize_mask(image_with_mask, core_config)
            scipy.misc.imsave(
                '{}/{}.png'.format(img_dir, image_id), visual_mask)

        # store in coco format as pickle
        result = convert_to_coco_result(image_with_mask, image_id, catIds)
        with open('{}/{}.pkl'.format(pkl_dir, image_id), 'wb') as fh:
            pickle.dump(result, fh)


def convert_to_coco_result(image_with_mask, image_id, catIds):
    """ This function accepts image_with_mask and convert it to coco results
        image_with_mask: a dict, defined in data_types.ty
        image_id: image id in COCO dataset
        catIds: a list that the index is the class_id, value is the category id in COCO
        return:
        results: a list of dictionaries that each dict represents an object instance
    """
    results = []
    mask = image_with_mask['mask']
    object_class = image_with_mask['object_class']
    num_objects = mask.max()
    for i in range(1, num_objects + 1):
        b_mask = (mask == i).astype('uint8')
        class_id = object_class[i - 1]  # class_id in our dataset
        category_id = catIds[class_id]  # category_id in official coco dataset
        result = {
            "image_id": image_id,
            "score": 1,  # TODO. for now set it as 1
            "category_id": category_id,
            "segmentation": maskUtils.encode(np.asfortranarray(b_mask))
        }
        results.append(result)
    return results


if __name__ == '__main__':
    main()
