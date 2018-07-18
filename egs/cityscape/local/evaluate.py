#!/usr/bin/env python3

# Copyright      2018  Yiwen Shao

# Apache 2.0

import os
import pickle
import argparse
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

parser = argparse.ArgumentParser(description='scoring script for COCO dataset')
parser.add_argument('--segment-dir', type=str, required=True,
                    help='Directory of segmentation results')
parser.add_argument('--val-ann', type=str,
                    default='data/download/annotations/instances_val2017.json',
                    help='Path to validation annotations file')
parser.add_argument('--imgid', type=int, default=None,
                    help='If given, only do evaluation on that image.'
                    'This is for analysis reason.')


def main():
    global args
    args = parser.parse_args()

    cocoGt = COCO(args.val_ann)
    class_nms = None
    class_nms_file = os.path.join(args.segment_dir, '../configs/subclass.txt')
    if os.path.exists(class_nms_file):
        with open(class_nms_file, 'r') as fh:
            class_nms = fh.readline().split()
            print('Evaluating on {} classes: {}'.format(
                len(class_nms), class_nms))
    else:
        print('Evaluating on all classes.')
    evaluate(cocoGt, args.segment_dir, class_nms, args.imgid)


def evaluate(coco, segment_dir, class_nms, imgid=None):
    pkl_dir = os.path.join(segment_dir, 'pkl')
    results = []
    if imgid:
        imgIds = [imgid]
    else:
        imgIds = []
    pkl_files = next(os.walk(pkl_dir))[2]
    for pkl_file in pkl_files:
        with open('{}/{}'.format(pkl_dir, pkl_file), 'rb') as fh:
            result = pickle.load(fh)
            for r in result:
                area = maskUtils.area(r['segmentation'])
                if area > 0:
                    results.append(r)
            if not imgid:
                # imgId should be int instead of str
                imgId = int(pkl_file.split('.')[0])
                imgIds.append(imgId)

    print('Evaluating on {} images: {}'.format(len(imgIds), imgIds))
    coco_results = coco.loadRes(results)
    # annIds = coco_results.getAnnIds(imgIds=imgIds, iscrowd=None)
    # print(annIds)
    # anns = coco_results.loadAnns(annIds)
    # for i, ann in enumerate(anns):
    #     print('{} \t{} \t {}'.format(i, ann['category_id'], ann['area']))
    cocoEval = COCOeval(coco, coco_results, 'segm')
    cocoEval.params.imgIds = imgIds
    if class_nms:
        cocoEval.params.catIds = coco.getCatIds(catNms=class_nms)
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
