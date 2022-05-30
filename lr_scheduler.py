"""
Flowvision training scheduler by flowvision contributors
"""
import oneflow as flow

# from typing import List


# from flowvision.scheduler.cosine_lr import CosineLRScheduler
# from flowvision.scheduler.linear_lr import LinearLRScheduler
# from flowvision.scheduler.step_lr import StepLRScheduler
# from flowvision.scheduler.multistep_lr import MultiStepLRScheduler


def build_scheduler(args, optimizer, n_iter_per_epoch):
    num_steps = int(args.epochs * n_iter_per_epoch)

    # warmup_steps = int(config.TRAIN.WARMUP_EPOCHS * n_iter_per_epoch)
    # if isinstance(config.TRAIN.LR_SCHEDULER.MILESTONES, List):
    #     assert (
    #         config.TRAIN.LR_SCHEDULER.NAME == "multi_step"
    #     ), "decay_t must be a list of epoch indices which are increasing only when you're using multi-step lr scheduler."
    #     decay_steps = [
    #         decay_step * n_iter_per_epoch
    #         for decay_step in config.TRAIN.LR_SCHEDULER.MILESTONES
    #     ]
    # else:
    #     decay_steps = int(config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if args.scheduler == "CosineAnnealingLR":
        lr_scheduler = flow.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.T_max if args.T_max is not None else num_steps,
            eta_min=args.eta_min,
            last_step=args.last_step,
        )
        # lr_scheduler = CosineLRScheduler(
        #     optimizer,
        #     t_initial=num_steps,
        #     t_mul=1.0,
        #     lr_min=config.TRAIN.MIN_LR,
        #     warmup_lr_init=config.TRAIN.WARMUP_LR,
        #     warmup_t=warmup_steps,
        #     cycle_limit=1,
        #     t_in_epochs=False,
        # )
    # elif config.TRAIN.LR_SCHEDULER.NAME == "linear":
    #     lr_scheduler = LinearLRScheduler(
    #         optimizer,
    #         t_initial=num_steps,
    #         lr_min_rate=0.01,
    #         warmup_lr_init=config.TRAIN.WARMUP_LR,
    #         warmup_t=warmup_steps,
    #         t_in_epochs=False,
    #     )
    # elif config.TRAIN.LR_SCHEDULER.NAME == "step":
    #     lr_scheduler = StepLRScheduler(
    #         optimizer,
    #         decay_t=decay_steps,
    #         decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
    #         warmup_lr_init=config.TRAIN.WARMUP_LR,
    #         warmup_t=warmup_steps,
    #         t_in_epochs=False,
    #     )
    # elif config.TRAIN.LR_SCHEDULER.NAME == "multi_step":
    #     lr_scheduler = MultiStepLRScheduler(
    #         optimizer,
    #         decay_t=decay_steps,
    #         decay_rate=config.TRAIN.LR_SCHEDULER.DECAY_RATE,
    #         warmup_lr_init=config.TRAIN.WARMUP_LR,
    #         warmup_t=warmup_steps,
    #         t_in_epochs=False,
    #     )
    else:
        return None

    return lr_scheduler
