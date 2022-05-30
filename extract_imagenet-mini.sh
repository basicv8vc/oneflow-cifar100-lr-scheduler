#!/bin/bash
#
# script to extract ImageNet-Mini dataset, refer to https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000?resource=download
# imagenet-mini.zip (about 4 GB)
# make sure imagenet-mini.zip in your current directory
#
#  Adapted from:
#  https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
#
#  imagenet/train/
#  ├── n01440764
#  │   ├── n01440764_10043.JPEG
#  │   ├── n01440764_12090.JPEG
#  │   ├── ......
#  ├── ......
#  imagenet/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00009111.JPEG
#  │   ├── ILSVRC2012_val_00030740.JPEG
#  │   ├── ILSVRC2012_val_00046252.JPEG
#  ├── ......
#
#
# Make imagenet_mini directory
#
mkdir imagenet_mini && mv imagenet-mini.zip imagenet_mini
cd imagenet_mini && unzip imagenet-mini.zip

# This results in a training directory like so:
#
#  imagenet_mini/train/
#  ├── n01440764
#  │   ├── n01440764_10043.JPEG
#  │   ├── n01440764_12090.JPEG
#  │   ├── ......
#  ├── ......
#
# This results in a validation directory like so:
#
#  imagenet_mini/val/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00009111.JPEG
#  │   ├── ILSVRC2012_val_00030740.JPEG
#  │   ├── ILSVRC2012_val_00046252.JPEG
#  ├── ......
#
#
# Check total files after extract
#
#  $ find train/ -name "*.JPEG" | wc -l
#  34745
#  $ find val/ -name "*.JPEG" | wc -l
#  3923
#
