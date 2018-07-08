#!/bin/bash

# Copyright 2018 Yiwen Shao
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh;

dl_dir=data/download
username=
password=
ann=s3

. ../../utils/parse_options.sh

# use the below command to get a cookie to download data from cityscape.
wget --keep-session-cookies --save-cookies=cookies.txt --post-data username=${username}\&password=${password}\&submit=Login https://www.cityscapes-dataset.com/login/

### Train/Val/Test images ###
img_dir=$dl_dir/leftImg8bit_trainvaltest
img_zip=leftImg8bit_trainvaltest.zip
if [ -d $img_dir ]; then
  echo Not downloading cityscape $img_zip as it is already there
else
  if [ ! -f $dl_dir/$img_zip ]; then
    img_url=https://www.cityscapes-dataset.com/file-handling/?packageID=3
    wget --load-cookies cookies.txt --content-disposition -P $dl_dir $img_url
  fi
  echo Unzipping $dl_dir/$img_zip
  unzip -qq $dl_dir/$img_zip -d $img_dir || exit 1;
  echo Done downloading and extracting cityscape $img_zip
fi

# re-organize image dir
for split in val train test; do
  if [ -d data/$split ]; then
    echo Not re-organizing $split image dir as it is already there
  else
    mkdir -p data/$split
    mv $dl_dir/leftImg8bit_trainvaltest/leftImg8bit/$split/*/*.png data/$split
    echo Done re-organizing $split image dir
  fi
done


### Train/Val/Test annotations ###
ann_dir=data/annotations
mkdir -p $ann_dir

if [ "$ann" != "s3" ]; then
  echo Download anns from official website and convert them to coco-like anns
  gt_dir=$dl_dir/gtFine_trainvaltest
  gt_zip=gtFine_trainvaltest.zip
  if [ -d $gt_dir ]; then
    echo Not downloading cityscape $gt_zip as it is already there
  else
    if [ ! -f $dl_dir/$gt_zip ]; then
      gt_url=https://www.cityscapes-dataset.com/file-handling/?packageID=1
      wget --load-cookies cookies.txt --content-disposition -P $dl_dir $gt_url
    fi
    echo Unzipping $gt_zip ...
    unzip -qq $dl_dir/$gt_zip -d $gt_dir || exit 1;
    echo Done downloading and extracting cityscape annotations
  fi



  if [ -f $ann_dir/instancesonly_filtered_gtFine_train.json ]; then
    echo Not converting to coco-like annotations as it is already there
  else 
    ./local/convert_cityscapes_to_coco.py --dataset cityscapes_instance_only \
					  --datadir $dl_dir \
					  --outdir $ann_dir
  fi
else
  echo Download converted anns from AWS s3 that we upload
  if [ -f $ann_dir/instancesonly_filtered_gtFine_train.json ]; then
    echo Not downloading cityscape annotations as they are already there
  else
    if [ ! -f $dl_dir/cityscape_ann.zip ]; then
      wget http://cityscape-anns.s3.amazonaws.com/cityscape_ann.zip -P $dl_dir
    fi
    echo Unzipping annotation file...
    unzip -qq $dl_dir/cityscape_ann.zip -d $ann_dir
    echo Done downloading and extracting cityscape annotations
  fi
fi
