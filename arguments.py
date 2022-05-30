import argparse


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
