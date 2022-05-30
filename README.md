# Demo on imagenet-mini using **OneFlow** with different learning rate schedulers

Source Code for Zhihu post "placeholder".

The structure of this codebase is borrowed from [flowvision/projects/classification
](https://github.com/Oneflow-Inc/vision/tree/main/projects/classification), big thanks and respect :heart:


## Requirements

This is my experiment eviroument
- python3.8
- oneflow==0.8.0.dev20220528+cu112
- flowvision==0.1.0
- opencv-python==4.4.0.46
- termcolor==1.1.0
- yacs==0.1.8

```bash
pip install -f https://staging.oneflow.info/branch/master/cu112 --pre oneflow

pip install flowvision==0.1.0 opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8

```



## Data prepartion

### ImageNet-Mini

ImageNet is good but too big, we use [ImageNet-Mini](https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000?resource=download) in this demo project.

ImageNet-Mini is about 4GB, contains 1K classes.

```bash
# download ImageNet-Mini from Kagge to current directory, https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000?resource=download
```

Then,

```bash
bash extract_imagenet-mini.sh
```

## Training

### Single GPU

```bash
python main.py
```

### Multiple GPUs

- ddp training with simple bash file
```bash
# set correct GPU_NUMS
bash ddp_training.sh
```
