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
import cv2
from models import get_model
from utils.segmenter import ObjectSegmenter, SegmenterOptions
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
parser.add_argument('--num-classes', default=9, type=int,
                    help='number of classes')
parser.add_argument('--num-offsets', default=10, type=int,
                    help='number of offsets')
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

    num_classes = args.num_classes
    num_offsets = args.num_offsets

    if args.mode == 'val':
        model = get_model(num_classes, num_offsets, args.arch, pretrain=False)
        model_path = os.path.join(args.dir, args.model)
        if os.path.isfile(model_path):
            print("=> loading checkpoint '{}'".format(model_path))
            checkpoint = torch.load(model_path,
                                    map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint['state_dict'])
            offset_list = checkpoint['offset']
            print("loaded.")
            print("offsets are: {}".format(offset_list))
        else:
            raise ValueError(
                "=> no checkpoint found at '{}'".format(model_path))
    elif args.mode == 'oracle':
        model = None  # we don't need model in oracle mode

    # dataset
    if args.mode == 'val' or args.mode == 'oracle':
        testset = COCODataset(args.img, args.ann, num_classes, offset_list,
                              scale=args.scale,
                              mode=args.mode,
                              limits=args.limits,
                              job=args.job, num_jobs=args.num_jobs)
    elif args.mode == 'test':
        testset = COCOTestset(args.img, args.ann)
    print('Total samples in the dataset to be segmented: {0}'.format(
        len(testset)))
    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=1, batch_size=args.batch_size)

    catIds = testset.catIds
    segment_dir = os.path.join(args.dir, args.segment)
    segment(dataloader, segment_dir, model,
            num_classes, offset_list, catIds, args.mode)


def segment(dataloader, segment_dir, model, num_classes, offset_list, catIds, mode):
    if model:
        model.eval()  # convert the model into evaluation mode
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)
    img_dir = os.path.join(segment_dir, 'img')
    pkl_dir = os.path.join(segment_dir, 'pkl')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(pkl_dir):
        os.makedirs(pkl_dir)
    exist_ids = next(os.walk(pkl_dir))[2]

    for i, vals in enumerate(dataloader):
        image_id = vals[0][0].item()
        img = vals[1]
        size = vals[2]
        original_height, original_width = size[0].item(), size[1].item()
        if str(image_id) + '.pkl' in exist_ids:
            continue
        if mode == 'val':  # 'val' mode, using model output to do segmentation
            with torch.no_grad():
                img_input = F.upsample(img,
                                       size=(args.train_image_size,
                                             args.train_image_size * 2),
                                       mode='bilinear', align_corners=True)
                output = model(img_input)
                class_mask = output[:, :num_classes, :, :]
                bound_mask = output[:, num_classes:, :, :]
        elif mode == 'oracle':  # 'oracle' mode, using ground truth label to do segmentation
            class_mask = vals[3][:, :num_classes, :, :]
            bound_mask = vals[3][:, num_classes:, :, :]

        if args.object_merge_factor is None:
            args.object_merge_factor = 1.0 / len(offset_list)
            segmenter_opts = SegmenterOptions(same_different_bias=args.same_different_bias,
                                              object_merge_factor=args.object_merge_factor,
                                              merge_logprob_bias=args.merge_logprob_bias)
        seg = ObjectSegmenter(class_mask[0].detach().numpy(),
                              bound_mask[0].detach().numpy(),
                              num_classes, offset_list,
                              segmenter_opts)
        mask, object_class = seg.run_segmentation()

        # resize the mask back to the original image size
        mask = cv2.resize(mask, (original_width, original_height),
                          interpolation=cv2.INTER_NEAREST)
        img = img[0].detach().numpy()

        # store segmentation result as image
        if args.visualize:
            masked_img = visualize_mask(img, mask, object_class)
            cv2.imwrite('{}/{}.png'.format(img_dir, image_id),
                        cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

        # store in coco format as pickle
        result = convert_to_coco_result(mask, object_class, image_id, catIds)
        with open('{}/{}.pkl'.format(pkl_dir, image_id), 'wb') as fh:
            pickle.dump(result, fh)


def convert_to_coco_result(mask, object_class, image_id, catIds):
    """ This function accepts mask and object_class, and convert it to coco results
        image_id: image id in COCO dataset
        catIds: a list that the index is the class_id, value is the category id in the
        original dataset.
        return:
        results: a list of dictionaries that each dict represents an object instance
    """
    results = []
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
