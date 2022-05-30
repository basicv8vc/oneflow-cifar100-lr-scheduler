# Demo on cifar100 using **OneFlow** with different learning rate schedulers

Source Code for Zhihu post "placeholder".

The structure of this codebase is borrowed from [flowvision/projects/classification
](https://github.com/Oneflow-Inc/vision/tree/main/projects/classification), big thanks and respect :heart:

**Note** This is just a demo project to demonstrate oneflow's learning rate scheduler.

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

### CIFAR-100

You can download [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) manually and place it in specified directory, or flowvision will automatically download it for you.

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
