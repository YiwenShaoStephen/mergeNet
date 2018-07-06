#!/bin/bash

# Copyright 2018 Yiwen Shao
# Apache 2.0

[ -f ./path.sh ] && . ./path.sh;

dl_dir=data/download
mkdir -p $dl_dir

ann_dir=data/annotations
mkdir -p $ann_dir
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
