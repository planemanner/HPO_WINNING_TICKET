import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torch.distributed import init_process_group, barrier
from torch.nn.parallel import DistributedDataParallel
from ._trainer import classification_trainer
from .logger import tensorboard_logger
from .evaluators import classification_evaluator
import optuna
from torch.utils.data import DataLoader


def set_seed(seed):
    # Random Seed Setter
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def exist_bn(model):
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            return True
        elif isinstance(m, torch.nn.BatchNorm1d):
            return True
        elif isinstance(m, torch.nn.BatchNorm3d):
            return True
        else:
            return False


def train_worker(train_set, val_set,
                 model: torch.nn.Module,
                 args, DDP: bool,
                 eval_history: dict, trial=None):
    set_seed(args.seed)
    cudnn.benchmark = True
    device = f"cuda:{args.local_rank}"
    trainer = classification_trainer()
    loss_fn = torch.nn.CrossEntropyLoss()
    tb_logger = tensorboard_logger(num_classes=args.num_classes, log_dir=args.log_dir, logging_freq=args.logging_freq)
    validator = classification_evaluator(tb_logger=tb_logger)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=args.wd, nesterov=True)

    is_distributed = DDP
    scaler = torch.cuda.amp.GradScaler()

    if is_distributed:
        init_method = "tcp://%s:%s" % (args.MASTER_ADDR, args.MASTER_PORT)

        init_process_group(backend="nccl",
                           init_method=init_method,
                           world_size=args.world_size,
                           rank=args.local_rank)
        barrier()

    if isinstance(trial, optuna.Trial):
        trial_state = trial.number
    else:
        trial_state = None

    # DDP SETUP
    if is_distributed and (trial_state == 0 or trial_state is None):
        torch.cuda.set_device(args.local_rank)
        sampler = torch.utils.data.distributed.DistributedSampler
        if exist_bn(model):
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.local_rank)
        model = DistributedDataParallel(model,
                                        device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=True)

        # Because inputs are distributed in several GPUs, you should divide batch size with world size
        args.bs /= args.world_size

    else:
        model = model.to(device)
        sampler = None
    # If you want to resume previous training, just leave your check-point in the args of action_main.py
    train_loader = DataLoader(train_set,
                              batch_size=args.bs,
                              shuffle=True,
                              sampler=sampler,
                              num_workers=4)

    val_loader = DataLoader(val_set,
                            batch_size=args.bs,
                            num_workers=4)

    milestones = [int(args.epochs * 0.5), int(args.epochs * 0.75)]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones)

    # main training
    print("--------------Start to do main training--------------")
    for epoch in range(args.epochs):
        trainer.train(dataloader=train_loader, model=model, optimizer=optimizer,
                      scheduler=lr_scheduler, loss_fn=loss_fn, scaler=scaler,
                      tensorboard_logger=tb_logger, epoch=epoch+1, rank=args.local_rank,
                      verbose_freq=args.train_verbose_freq, batch_size=args.bs)

        if (epoch + 1) % args.eval_period == 0:
            validator.validate(epoch=epoch+1,
                               batch_size=args.bs,
                               criterion=loss_fn,
                               gpu=args.local_rank,
                               model=model,
                               dataloader=val_loader,
                               verbose_freq=args.eval_freq)
#             epoch, batch_size, criterion, gpu,
#                  verbose_freq, model, dataloader, ensemble=False
            if trial:
                """
                If you want to use hpo, `trial` instance gets evaluation result in each evaluation period. 
                During training, if training tendency is not good as much as previous one, optuna forcely terminates  
                the training.
                """
                last_eval_value = validator.get_main_eval_index()[-1]
                trial.report(last_eval_value, epoch + 1)

    # validator.get_eval_index() => The list of evaluation in every epochs. TYPE: List
    tb_logger.writer.close()
    eval_history["values"] = validator.get_main_eval_index()
    print("--------------Training is finished-----------------")