"""
Modified from https://github.com/microsoft/Swin-Transformer/blob/main/main.py
"""

import os
import time
import argparse
import datetime
import numpy as np
import oneflow as flow
import oneflow.backends.cudnn as cudnn

from flowvision.loss.cross_entropy import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)
from flowvision.models import ModelCreator
from flowvision.utils.metrics import accuracy, AverageMeter

from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    build_loader,
    get_grad_norm,
    reduce_tensor,
)


def _add_base_training_args(parser):
    group = parser.add_argument_group(title="Base arguments")
    group.add_argument(
        "--exp",
        default=None,
        type=str,
        help="current experiment name, used for output dir",
    )
    group.add_argument(
        "--num_classes",
        default=100,
        type=int,
        choices=[100],
        help="Number of classes, 100(cifar100)",
    )
    group.add_argument(
        "--data_path",
        default="./cifar100/",
        type=str,
        help="path to store dataset, or you have put cifar-100-python.tar.gz in this data_path",
    )
    group.add_argument(
        "--model_arch",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "resnet34",
            "resnet50",
        ],
        help="model name",
    )
    group.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="input image size: (img_size, img_size), not 32",
    )

    group.add_argument(
        "--zip_mode",
        default=False,
        type=bool,
        help="Use zipped dataset instead of folder dataset",
    )

    group.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    group.add_argument(
        "--cache_mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="Cache Data in Memory: no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    group.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    group.add_argument("--tag", help="tag of experiment")
    group.add_argument("--throughput", action="store_true", help="Test throughput only")

    group.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    group.add_argument(
        "--pin_memory",  # oneflow都没有实现non_blocking
        default=True,
        type=bool,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    group.add_argument(
        "--interpolation",
        default="bicubic",
        type=str,
        choices=["random", "bilinear", "bicubic"],
        help="Interpolation to resize image (random, bilinear, bicubic)",
    )

    group.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="number of total epochs to run",
    )
    group.add_argument(
        "--batch_size",
        default=128,
        type=int,
        help="batch size for single GPU for one GPU or when using DidstributedDataParallel",
    )
    group.add_argument(
        "--optimizer",
        default="sgd",
        type=str,
        choices=["sgd", "adamw"],
        help="optimizer",
    )
    group.add_argument(
        "--lr",
        default=0.1,
        type=float,
        help="initial learning rate",
    )
    group.add_argument("--momentum", default=0.9, type=float, help="momentum for sgd")
    group.add_argument("--eps", default=1e-8, type=float, help="eps for adamw")
    group.add_argument(
        "--betas", default=(0.9, 0.999), type=tuple, help="betas for adamw"
    )
    group.add_argument(
        "--weight_decay",
        default=1e-4,
        type=float,
        help="weight decay (default: 1e-4)",
    )
    group.add_argument(
        "--drop_rate",
        default=0.1,
        type=float,
        help="dropout rate",
    )
    group.add_argument(
        "--label_smoothing",
        default=0.1,
        type=float,
        help="label smoothing",
    )

    group.add_argument(
        "--print_freq",
        default=20,
        type=int,
        help="print training log",
    )
    group.add_argument(
        "--seed", default=123, type=int, help="seed for initializing training"
    )
    group.add_argument(
        "--crop",
        default=True,
        type=bool,
        help="whether to use center crop when testing",
    )
    group.add_argument(
        "--sequential",
        default=False,
        type=bool,
        help="whether to use SequentialSampler as validation sampler ",
    )
    # group.add_argument(
    #     "--gpu", default=0, type=int, help="GPU id to use. If not None, disable DDP"
    # )
    return parser


def _add_augmentation_args(parser):
    group = parser.add_argument_group(title="Augmentation arguments")
    group.add_argument(
        "--auto_augment",
        default=None,
        type=str,
        help="",
    )
    group.add_argument(
        "--color_jitter",
        default=0.4,
        type=float,
        help="color jitter factor",
    )
    group.add_argument(
        "--reprob",
        default=0.25,
        type=float,
        help="random erase prob",
    )
    group.add_argument(
        "--remode",
        default="pixel",
        type=str,
        help="random erase mode",
    )
    group.add_argument(
        "--recount",
        default=1,
        type=int,
        help="random erase count",
    )
    group.add_argument(
        "--mixup",
        default=0.8,
        type=float,
        help="mixup alpha, mixup enabled if > 0",
    )
    group.add_argument(
        "--cutmix",
        default=1.0,
        type=float,
        help="cutmix alpha, cutmix enabled if > 0",
    )
    group.add_argument(
        "--cutmix_minmax",
        default=None,
        type=float,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set",
    )
    group.add_argument(
        "--mixup_prob",
        default=1.0,
        type=float,
        help="probability of performing mixup or cutmix when either/both is enabled",
    )
    group.add_argument(
        "--mixup_switch_prob",
        default=0.5,
        type=float,
        help="probability of switching to cutmix when both mixup and cutmix enabled",
    )
    group.add_argument(
        "--mixup_mode",
        default="batch",
        type=str,
        choices=["batch", "pair", "elem"],
        help="how to apply mixup/cutmix params. Per 'batch', 'pair', or 'elem'",
    )

    return parser


def _add_scheduler_args(parser):
    group = parser.add_argument_group(title="Scheduler arguments")

    group.add_argument(
        "--scheduler",
        default=None,
        type=str,
        choices=["CosineAnnealingLR"],
        help="learning rate scheduler",
    )
    group.add_argument(
        "--last_step",
        type=int,
        default=-1,
        help="The index of last iteration/epoch. Default: -1",
    )
    group.add_argument(
        "--T_max",
        # required="CosineAnnealingLR" in sys.argv,
        default=None,
        type=int,
        help="Maximum number of iterations/epochs",
    )  # only required if choose CosineAnnealingLR
    group.add_argument(
        "--eta_min",
        default=1e-5,
        # required="CosineAnnealingLR" in sys.argv,
        type=float,
        help="Minimum learning rate",
    )  # only required if --argument is given

    return parser


def _add_ddp_args(parser):
    group = parser.add_argument_group(title="DDP arguments")
    group.add_argument(
        "--local_rank",
        type=int,
        default=0,
        required=False,
        help="local rank for DistributedDataParallel",
    )

    return parser


def parse_args():
    """Parse all arguments."""
    parser = argparse.ArgumentParser(description="Scheduler Demo")
    parser = _add_base_training_args(parser)
    parser = _add_augmentation_args(parser)
    parser = _add_scheduler_args(parser)
    parser = _add_ddp_args(parser)

    args = parser.parse_args()
    return args


def main(args):
    (
        data_loader_train,
        data_loader_val,
        mixup_fn,
    ) = build_loader(args)

    logger.info(f"Creating model:{args.model_arch}")
    model = ModelCreator.create_model(
        args.model_arch, pretrained=False, num_classes=args.num_classes
    )
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(args, model)
    print("optimizer: {}".format(optimizer))
    model = flow.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
    # FIXME: model with DDP wrapper doesn't have model.module
    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, "flops"):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    lr_scheduler = build_scheduler(args, optimizer, len(data_loader_train))

    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.0:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = flow.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if args.throughput:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            args,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            mixup_fn,
            lr_scheduler,
        )
        acc1 = validate(model, data_loader_val, epoch, args.epochs)
        logger.info(f"Accuracy of the network on the test images: {acc1:.1f}%")
        max_accuracy = max(max_accuracy, acc1)
        logger.info(f"Max accuracy: {max_accuracy:.2f}%")
        if epoch % args.print_freq == 0:
            flow.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict()
                    if lr_scheduler is not None
                    else None,
                },
                args.output + "/{}.epoch-{}.flow".format(args.model_arch, epoch),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))


def train_one_epoch(
    args, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda()
        targets = targets.cuda()

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        outputs = model(samples)

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = get_grad_norm(model.parameters())
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_meter.update(loss.item(), targets.size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            lr = optimizer.param_groups[0]["lr"]
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
            )
    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )


@flow.no_grad()
def validate(model, data_loader, cur_epoch, epochs):
    criterion = flow.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    for images, target in data_loader:
        images = images.cuda()
        target = target.cuda()

        # compute output
        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info(
        f"Test: [{cur_epoch}/{epochs}]\t"
        f"Time {batch_time.avg:.3f}\t"
        f"Loss {loss_meter.avg:.4f}\t"
        f"Acc@1 {acc1_meter.avg:.3f}%\t"
        f"Acc@5 {acc5_meter.avg:.3f}%\t"
    )

    return acc1_meter.avg


@flow.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda()
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        # TODO: add flow.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)

        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = flow.env.get_rank()
        world_size = flow.env.get_world_size()
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    seed = args.seed + flow.env.get_rank()
    flow.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    output_name = (
        args.output + "/" + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        if args.exp is None
        else args.exp
    )
    args.output = output_name
    os.makedirs(output_name, exist_ok=True)
    logger = create_logger(
        output_dir=output_name,
        dist_rank=flow.env.get_rank(),
        name=f"{args.model_arch}",
    )

    if flow.env.get_rank() == 0:
        path = os.path.join(output_name, "config.json")
        with open(path, "w") as f:
            f.write(str(args))
        logger.info(f"Full config saved to {path}")

    main(args)
