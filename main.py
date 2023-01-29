import argparse
from SSP.pruners import ResNet_Pruner
from model_src.resnet_cifar import ResNet_CIFAR
from torch.utils.data import DataLoader
from HPO.objective_func import do_hpo
from HPO.worker import train_worker
import os
import torch
from call_data import call_dataset
from model_src.get_models import get_model, AVAILABLE_MODEL_NAMES, get_model_args


if __name__ == "__main__":
    """
    pruning + hpo
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--min_channel_ratio", type=float, default=0.15)
    parser.add_argument("--percentile", type=float, default=70, help="The higher, the more compressed")
    parser.add_argument("--input_image_channels", type=int, default=3)
    parser.add_argument("--block_type", type=str, default="basicblock")
    parser.add_argument("--pruning_bs", type=int, default=512)
    parser.add_argument("--job_dir", type=str, default="hpo_with_pruned_model", help="hpo db dir")
    parser.add_argument("--job_name", type=str, default="resnet56", help="literally, job name")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_trials", type=int, default=100, help="the number of trials")
    parser.add_argument("--hpo_name", type=str, default="bo", help="hpo algorithm name")
    parser.add_argument("--world_size", type=int, default=1, help="The number of process")
    parser.add_argument("--lr", type=float, default=4, help="The number of process")
    parser.add_argument("--wd", type=float, default=5e-4, help="The number of process")
    parser.add_argument("--bs", type=int, default=256, help="The number of process")
    parser.add_argument("--num_classes", type=int, default=10, help="The number of process")
    parser.add_argument("--logging_freq", type=int, default=1, help="logging frequency")
    parser.add_argument("--local_rank", type=int, default=0, help="process rank")
    parser.add_argument("--log_dir", type=str, default="tensorboard_log", help="logging directory")
    parser.add_argument("--dataset_dir", type=str, default="./SSP", help="logging directory")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=2e-1)
    parser.add_argument("--epochs", type=int, default=160, help="The number of process")
    parser.add_argument("--train_verbose_freq", type=int, default=50)
    parser.add_argument("--eval_period", type=int, default=1)
    parser.add_argument("--eval_freq", type=int, default=50)
    parser.add_argument('--no_prune', action='store_true')
    parser.add_argument('--model_dir', type=str, default="./model_ckpts/base",
                        help="The directory of saving a pruned model's weight")
    parser.add_argument('--network_name', type=str, default="resnet56")
    parser.add_argument('--model_path', type=str, default="")
    parser.add_argument('--dataset_name', type=str, default="CIFAR100", choices=["CIFAR100", "TinyImageNet", "MNIST"])
    parser.add_argument('--model_name', type=str, default='resnet', choices=AVAILABLE_MODEL_NAMES)

    args = parser.parse_args()
    args.job_dir = args.job_dir + "_seed_" + str(args.seed) + "_pruning_rate_" + str(args.percentile)
    args.job_name = args.job_name + "_seed_" + str(args.seed) + "_pruning_rate_" + str(args.percentile)

    train_dataset, val_dataset = call_dataset(dataset_name=args.dataset_name,
                                              root_path=args.dataset_dir)

    if args.no_prune:
        model_args = get_model_args(model_name=args.model_name,
                                    input_image_channels=args.input_image_channels,
                                    num_classes=args.num_classes)
        model = get_model(args.model_name, 'cpu', **model_args)
        model_path = os.path.join(args.model_dir, f"{args.network_name}_base_model.pth")
        args.model_path = model_path
        torch.save(model, model_path)
        model = torch.load(args.model_path)
        do_hpo(args, train_set=train_dataset, val_set=val_dataset, DDP=False, worker=train_worker, model=model)
    else:
        model_args = get_model_args(model_name=args.model_name,
                                    input_image_channels=args.input_image_channels,
                                    num_classes=args.num_classes)

        pruning_dataloader = DataLoader(dataset=train_dataset, batch_size=args.pruning_bs)
        model = get_model(model_name=args.model_name, device='cpu', **model_args)
        Pruner = ResNet_Pruner(args=args, model=model, block_type=args.block_type)
        pruned_model = Pruner.do_pruning(pruning_dataloader)

        model_path = os.path.join(args.model_dir, f"{args.network_name}_pruned_model.pth")
        args.model_path = model_path
        torch.save(pruned_model, model_path)
        do_hpo(args, train_set=train_dataset, val_set=val_dataset, DDP=False, worker=train_worker, model=pruned_model)
