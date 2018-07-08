#!/bin/bash

train_image_size=384

epochs=400
dir=exp/crop/pspfpnet101_alpha20

. ./path.sh
. ../../utils/parse_options.sh

echo "$0: Training the network....."
python3 local/train.py \
	--epochs $epochs \
	--train-image-size $train_image_size \
	--alpha 20 \
	--arch pspfpnet \
	--pretrain \
	--tensorboard \
	--crop \
	$dir

