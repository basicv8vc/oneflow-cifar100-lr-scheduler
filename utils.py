"""
Modified from https://github.com/microsoft/Swin-Transformer/blob/main/utils.py
"""
import numpy as np

import oneflow as flow
from oneflow.utils.data import DataLoader

from flowvision import datasets, transforms
from flowvision.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from flowvision.data import create_transform
from flowvision.transforms.functional import str_to_interp_mode
from flowvision.data import Mixup


def get_grad_norm(parameters, norm_type=2):
    total_norm = flow.linalg.vector_norm(
        flow.stack(
            [flow.linalg.vector_norm(p.grad.detach(), norm_type) for p in parameters]
        ),
        norm_type,
    )
    return total_norm


def reduce_tensor(tensor):
    rt = tensor.clone()
    flow.comm.all_reduce(rt)
    rt /= flow.env.get_world_size()
    return rt


class SubsetRandomSampler(flow.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in flow.randperm(len(self.indices)).tolist())

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def build_loader(args):
    dataset_train = build_dataset(is_train=True, args=args)
    print(
        f"local rank {args.local_rank} / global rank {flow.env.get_rank()} successfully build train dataset"
    )
    dataset_val = build_dataset(is_train=False, args=args)
    print(
        f"local rank {args.local_rank} / global rank {flow.env.get_rank()} successfully build val dataset"
    )

    num_tasks = flow.env.get_world_size()
    global_rank = flow.env.get_rank()
    sampler_train = flow.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )

    indices = np.arange(
        flow.env.get_rank(), len(dataset_val), flow.env.get_world_size()
    )
    sampler_val = SubsetRandomSampler(indices)
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmin > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.label_smoothing,
            num_classes=args.num_classes,
        )

    return data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    dataset = datasets.CIFAR100(
        root=args.input,
        train=is_train,
        transform=transform,
        download=True,
    )

    return dataset


def build_transform(is_train, args):
    resize_im = args.img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.img_size,
            is_training=True,
            color_jitter=args.color_jitter if args.color_jitter > 0 else None,
            auto_augment=args.auto_augment,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            interpolation=args.interpolation,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(args.img_size, padding=4)
        return transform

    t = []
    if resize_im:
        if args.crop:
            size = int((256 / 224) * args.img_size)
            t.append(
                transforms.Resize(
                    size, interpolation=str_to_interp_mode(args.interpolation)
                ),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(args.img_size))
        else:
            t.append(
                transforms.Resize(
                    (args.img_size, args.img_size),
                    interpolation=str_to_interp_mode(args.interpolation),
                )
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
