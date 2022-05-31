"""
Modified from https://github.com/microsoft/Swin-Transformer/blob/main/main.py
"""

import os
import time
import argparse
import datetime
import numpy as np
import oneflow as flow
from oneflow import optim
import oneflow.backends.cudnn as cudnn


from flowvision.models import ModelCreator
from flowvision.utils.metrics import accuracy, AverageMeter
from flowvision.loss.cross_entropy import (
    LabelSmoothingCrossEntropy,
    SoftTargetCrossEntropy,
)


from arguments import parse_args
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import (
    build_loader,
    get_grad_norm,
    reduce_tensor,
)


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

    if args.checkpoint is not None:
        states_dict = flow.load(args.checkpoint)
        model.load_state_dict(states_dict["model"])
        optimizer.load_state_dict(states_dict["optimizer"])
        if states_dict["scheduler"] is not None:
            lr_scheduler.load_state_dict(states_dict["scheduler"])

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
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
