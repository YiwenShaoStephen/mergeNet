#!/bin/bash
stage=0
dir=exp/pspnet_upernet50
mkdir -p $dir

class_dir=exp/pspnet_caffe
offset_dir=exp/ofs/upernet50_scale2_crop512_dice

caffe_model=$class_dir/pspnet101_cityscapes.caffemodel
class_model=$class_dir/pspnet101_cityscapes.pth.tar
offset_model=$offset_dir/model_best.pth.tar

. ./path.sh
. ../../utils/parse_options.sh

if [ $stage -le 0 ]; then
  if [ ! -f $class_model ]; then
    echo "$0: Need to convert model first"
    if [ ! -f $caffe_model ]; then
      echo "$0: $caffe_model doesn't exist, download it first"
      exit 0
    else
      python local/convert_caffe_to_pytorch.py \
	      --caffe-model $caffe_model \
	      --pytorch-model $class_model \
	      --dataset cityscapes || exit 1
    fi
  fi

  echo "$0: Doing class inference....."
  python local/class_infer.py \
	  --dir $class_dir \
	  --model $class_model \
	  --score \
	  --caffe \
	  --gpu || exit 1
fi


if [ $stage -le 1 ]; then
  echo "$0: Doing offset inference....."
  python local/offset_infer.py \
	  --dir $offset_dir \
	  --model $offset_model \
	  --mode val \
	  --arch upernet \
	  --score \
	  --gpu || exit 1
fi

segdir=segment_512
mkdir -p $dir/$segdir/img
mkdir -p $dir/$segdir/pkl
mkdir -p $dir/$segdir/result

if [ $stage -le 2 ]; then
    echo "$0: Doing segmentation...."
    python local/segment.py \
	   --dir $dir \
	   --class-dir $class_dir \
	   --offset-dir $offset_dir \
	   --segment $segdir \
	   --limits 1 \
	   --visualize || exit 1
fi

if [ $stage -le 3 ]; then
    echo "$0: Doing evaluation..."
    python local/evaluate.py \
	   --segment-dir $dir/$segdir || exit 1
fi

if [ $stage -le 4 ]; then
    echo "$0: Doing converting..."
    python local/submit.py \
	   --segment-dir $dir/$segdir || exit 1
fi
