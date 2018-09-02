# Copyright      2018  Yiwen Shao

# Apache 2.0

import os
import pickle
import argparse
import cv2
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

parser = argparse.ArgumentParser(description='scoring script for COCO dataset')
parser.add_argument('--segment-dir', type=str, required=True,
                    help='Directory of segmentation results')
parser.add_argument('--val-ann', type=str,
                    default='data/annotations/instancesonly_filtered_gtFine_val.json',
                    help='Path to validation annotations file')


def main():
    global args
    args = parser.parse_args()
    coco_obj = COCO(args.val_ann)
    convert_pkl_to_cityscapes_result(coco_obj, args.segment_dir)


def convert_pkl_to_cityscapes_result(coco, segment_dir):
    pkl_dir = os.path.join(segment_dir, 'pkl')
    result_outdir = os.path.join(segment_dir, 'result')
    pkl_files = next(os.walk(pkl_dir))[2]
    for pkl_file in pkl_files:
        with open('{}/{}'.format(pkl_dir, pkl_file), 'rb') as fh:
            result = pickle.load(fh)
            # if len(result) > 0:
            #     convert_coco_to_cityscapes_result(coco, result, result_outdir)
            if len(result) <= 0:
                print(result)


def convert_coco_to_cityscapes_result(coco_obj, coco_result, result_outdir,
                                      labelID=[0, 24, 25, 26, 27, 28, 31, 32, 33]):
    image_id = coco_result[0]['image_id']
    img_name = coco_obj.loadImgs(image_id)[0]['file_name'].split('.')[0]
    txt_filename = img_name + '.txt'
    txt_path = os.path.join(result_outdir, txt_filename)
    with open(txt_path, 'w') as fh:
        for i, r in enumerate(coco_result):
            rle = r['segmentation']
            category_id = r['category_id']
            b_mask = maskUtils.decode(rle) * 255
            b_mask_filename = '{}_{}.png'.format(img_name, i)
            b_mask_path = os.path.join(result_outdir, b_mask_filename)
            cv2.imwrite(b_mask_path, b_mask)
            label_id = labelID[category_id]
            confidence = r['score']
            result_string = "{} {} {}\n".format(
                b_mask_filename, label_id, confidence)
            fh.write(result_string)


if __name__ == '__main__':
    main()
