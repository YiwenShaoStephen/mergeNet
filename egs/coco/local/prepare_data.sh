#!/bin/bash

# Copyright 2018 Yiwen Shao
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh;

dir=data
dl_dir=data/download
year=2017
mkdir -p $dl_dir

### Train/Val/Test images ###
for data_type in val train test; do
  img_dir=$dir/${data_type}${year}
  img_zip=${data_type}${year}.zip
  if [ -d $img_dir ]; then
    echo Not downloading COCO ${img_zip} as it is already there
  else
    if [ ! -f $dl_dir/${img_zip} ]; then
      img_url=http://images.cocodataset.org/zips/${img_zip}
      wget -P $dl_dir $img_url
    fi
    echo Unzipping $dl_dir/${img_zip}
    unzip -qq $dl_dir/${img_zip} -d $dir || exit 1;
    echo Done downloading and extracting COCO ${img_zip}
  fi
done

### Train/Val annotations ###
ann_dir=$dir/annotations
mkdir -p $ann_dir
ann_zip=annotations_trainval${year}.zip
ann_url=http://images.cocodataset.org/annotations/annotations_trainval${year}.zip
if [ -f $ann_dir/instances_val2017.json ]; then
  echo Not downloading COCO annotations as they are already there.
  else
    if [ ! -f $dl_dir/$ann_zip ]; then
      wget -P $dl_dir $ann_url
    fi
    echo Unzipping $ann_zip ...
    unzip -qq $dl_dir/$ann_zip -d $dir || exit 1;
    echo Done downloading and extracting COCO annotations
fi

### Test info ###
info_zip=image_info_test2017.zip
info_url=http://images.cocodataset.org/annotations/image_info_test2017.zip
if [ -f $ann_dir/image_info_test2017.json ]; then
  echo Not downloading COCO test info as they are already there.
else
  if [ ! -f $dl_dir/$info_zip ]; then
    wget -P $dl_dir $info_url
  fi
  echo Unzipping $info_zip ...
  unzip -qq $dl_dir/$info_zip -d $dir || exit 1;
  echo Done downloading and extracting COCO test info
fi
