#!/bin/bash

set -e # exit on error
. ./path.sh

stage=0

. parse_options.sh  # e.g. this parses the --stage option if supplied.


. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.

local/check_dependencies.sh

# download data
if [ $stage -le 0 ]; then
  local/prepare_data.sh
fi
exit 0

epochs=120
depth=5
dir=exp/fcn_vgg16_pretrain
if [ $stage -le 1 ]; then
  # training
  local/run_unet_all.sh --dir $dir --epochs $epochs --depth $depth
fi

segdir=segment_val
logdir=$dir/$segdir/log
nj=2
if [ $stage -le 2 ]; then
  echo "doing segmentation...."
    $cmd --mem 10G JOB=1:$nj $logdir/segment.JOB.log local/segment.py \
       --limits 2 \
       --train-image-size 256 \
       --seg-size 128 \
       --model model_best.pth.tar \
       --mode val \
       --segment $segdir \
       --job JOB --num-jobs $nj \
       --dir $dir \
       --img data/download/val2017 \
       --ann data/download/annotations/instances_val2017.json
fi
exit 0

if [ $stage -le 3 ]; then
  echo "doing evaluation..."
  local/evaluate.py \
    --segment-dir $segdir \
    --val-ann data/download/annotations/instances_val2017.json
fi
