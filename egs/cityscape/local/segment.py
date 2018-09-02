#!/usr/bin/env python3

# Copyright      2018  Yiwen Shao

# Apache 2.0

import argparse
import os
import pickle
import random
import numpy as np
import torch
import torch.utils.data
import cv2
from utils.data_visualization import visualize_mask
from utils.dataset import AllDataset
from utils.train_utils import generate_offsets
from pycocotools import mask as maskUtils
import utils.csegment.c_segment as cseg


parser = argparse.ArgumentParser(
    description='Pytorch cityscapes instance segmentation setup')
parser.add_argument('--dir', type=str, required=True,
                    help='Experiment directory')
parser.add_argument('--class-dir', type=str, required=True,
                    help='directory of class output numpy arrays')
parser.add_argument('--offset-dir', type=str, required=True,
                    help='directory of offset output numpy arrays')
parser.add_argument('--img', type=str, default='data/val',
                    help='test images directory')
parser.add_argument('--ann', type=str,
                    default='data/annotations/instancesonly_filtered_gtFine_val.json',
                    help='path to test annotation or info file')
parser.add_argument('--segment', type=str, default='segment',
                    help='sub dir name under <args.dir> that segmentation results'
                    'will be stored to')
parser.add_argument('--num-classes', default=9, type=int,
                    help='number of classes')
parser.add_argument('--num-offsets', default=10, type=int,
                    help='number of offsets')
parser.add_argument('--limits', default=None, type=int,
                    help="If given, is the size of subset we do segmenting on")
parser.add_argument('--seg-size', default=128, type=int,
                    help='network output will be resized to and then do segmentation on.')
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
parser.add_argument('--job', type=int, default=0, help='job id')
parser.add_argument('--num-jobs', type=int, default=1,
                    help='number of parallel jobs')
parser.add_argument('--visualize',
                    help='Whether to store segmentation results as images to disk',
                    action='store_true')
random.seed(0)
np.random.seed(0)


def main():
    global args
    args = parser.parse_args()
    args.batch_size = 1  # only segment one image for experiment

    num_classes = args.num_classes
    num_offsets = args.num_offsets

    offset_list = generate_offsets(40, num_offsets)
    print("offsets are: {}".format(offset_list))

    # dataset
    testset = AllDataset(args.img, args.ann, num_classes, offset_list,
                         mode='test',
                         limits=args.limits,
                         job=args.job, num_jobs=args.num_jobs)
    print('Total samples in the dataset to be segmented: {0}'.format(
        len(testset)))
    dataloader = torch.utils.data.DataLoader(
        testset, num_workers=0, batch_size=1)

    catIds = testset.catIds
    # coco_obj = testset.coco
    segment_dir = os.path.join(args.dir, args.segment)
    seg_size = (1024, 512)
    segment(dataloader, segment_dir, num_classes,
            offset_list, seg_size, catIds)


def segment(dataloader, segment_dir, num_classes, offset_list, seg_size, catIds):
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)
    img_dir = os.path.join(segment_dir, 'img')
    pkl_dir = os.path.join(segment_dir, 'pkl')
    # result_dir = os.path.join(segment_dir, 'result')
    exist_ids = next(os.walk(pkl_dir))[2]

    for i, (image_id, img, size) in enumerate(dataloader):
        image_id = image_id.item()
        if str(image_id) + '.pkl' in exist_ids:
            continue
        class_filename = '{}/npy/{}.class.npy'.format(args.class_dir, image_id)
        offset_filename = '{}/npy/{}.offset.npy'.format(
            args.offset_dir, image_id)
        class_mask = np.load(class_filename)
        bound_mask = np.load(offset_filename)
        if seg_size:
            class_mask = np.moveaxis(class_mask, 0, -1)
            bound_mask = np.moveaxis(bound_mask, 0, -1)
            class_mask = cv2.resize(class_mask, seg_size)
            bound_mask = cv2.resize(bound_mask, seg_size)
            class_mask = np.moveaxis(class_mask, -1, 0)
            bound_mask = np.moveaxis(bound_mask, -1, 0)
            class_mask = class_mask.copy(order='C')
            bound_mask = bound_mask.copy(order='C')

        # if args.object_merge_factor is None:
        #     args.object_merge_factor = 1.0 / len(offset_list)
        #     segmenter_opts = SegmenterOptions(same_different_bias=args.same_different_bias,
        #                                       object_merge_factor=args.object_merge_factor,
        #                                       merge_logprob_bias=args.merge_logprob_bias)
        # seg = ObjectSegmenter(class_mask, bound_mask, num_classes, offset_list,
        #                       segmenter_opts)
        # mask, object_class = seg.run_segmentation()

        if args.object_merge_factor is None:
            args.object_merge_factor = 1  # / len(offset_list)
        args.merge_logprob_bias = 0.03

        mask, object_class = cseg.run_segmentation(class_mask, bound_mask,
                                                   num_classes,
                                                   offset_list,
                                                   args.same_different_bias,
                                                   args.object_merge_factor,
                                                   args.merge_logprob_bias)

        # resize the mask back to the original image size
        if seg_size:
            original_height, original_width = size[0].item(), size[1].item()
            mask = cv2.resize(mask, (original_width, original_height),
                              interpolation=cv2.INTER_NEAREST)
        img = img[0].detach().numpy()
        # store segmentation result as image
        if args.visualize:
            masked_img = visualize_mask(img, mask, transparency=0.3)
            cv2.imwrite('{}/{}.png'.format(img_dir, image_id),
                        cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

        # store in coco format as pickle
        result = convert_to_coco_result(mask, object_class, image_id, catIds)
        with open('{}/{}.pkl'.format(pkl_dir, image_id), 'wb') as fh:
            pickle.dump(result, fh)
        # convert_to_cityscapes_result(
        #     mask, object_class, image_id, result_dir, coco)


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


def convert_to_cityscapes_result(mask, object_class, image_id, result_dir, coco,
                                 labelID=[0, 24, 25, 26, 27, 28, 31, 32, 33]):
    img_name = coco.loadImgs(image_id)[0]['file_name'].split('.')[0]
    txt_filename = img_name + '.txt'
    txt_path = os.path.join(result_dir, txt_filename)
    with open(txt_path, 'w') as fh:
        num_objects = mask.max()
        for i in range(1, num_objects + 1):
            b_mask = (mask == i).astype('uint8') * 255
            b_mask_filename = '{}_{}.png'.format(img_name, i)
            b_mask_path = os.path.join(result_dir, b_mask_filename)
            cv2.imwrite(b_mask_path, b_mask)
            label_id = labelID[object_class[i - 1]]
            confidence = 1.0
            result_string = "{} {} {}\n".format(
                b_mask_filename, label_id, confidence)
            fh.write(result_string)


if __name__ == '__main__':
    main()
