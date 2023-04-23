from SSP.pruning_utils import *
import torch
import sys
sys.path.append('..')
from model_src.resnet_cifar import ResNet_CIFAR
from torchvision.models.resnet import Bottleneck, BasicBlock
from torch.utils.data import DataLoader
from torchvision import transforms as ttf
from torchvision.datasets import CIFAR10
import numpy as np
import argparse


class Default_Pruner:
    def __init__(self, args, model: torch.nn.Module):
        self.prune_args = args
        self.model = model
        self.fanin_scores = {}
        self.fanout_scores = {}
        self.total_scores = []  # to get the threshold without additional hashing to fanout_scores.
        self.fanout_indices = {}
        self.fanin_indices = {}

    def scoring(self, dataloader):
        pass

    def get_pruning_graph(self):
        pass

    def pruning(self):
        pass

    def get_pruned_network(self):
        pass
        """
        <operation order>
        self.scoring()
        self.get_pruning_graph()
        self.pruning()  
        """


class ResNet_Pruner(Default_Pruner):
    def __init__(self, args, model, block_type):
        super(ResNet_Pruner, self).__init__(args, model)
        """
        torch provided models
        resnet18~resnet152

        layer operation parser
        conv, bn, downsample, fc
        """

        self.block_type = block_type.lower()
        self.input_image_channels = args.input_image_channels
        self.num_classes = args.num_classes
        if block_type == "bottleneck":
            block = Bottleneck
        elif block_type == "basicblock":
            block = BasicBlock
        else:
            raise AssertionError("Invalid block type of resnet")

        self.blocks, self.others = block_parser(self.model, block)


    def scoring(self, dataloader):
        # Channel-wise scoring
        data, labels = next(iter(dataloader))
        output = self.model(data.to(self.prune_args.device))
        torch.mean(output).backward()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'bias' not in name and 'downsample' not in name:
                    # downsample and bias are dependent to the 'weight'.
                    weight_scores = param * param.grad
                    layer = get_module(self.model, name)
                    if isinstance(layer, torch.nn.Conv2d):
                        pruning_scores = weight_scores.sum(dim=(1, 2, 3))
                        self.fanout_scores[name] = pruning_scores
                        self.fanin_scores[name] = weight_scores.sum(dim=(0, 2, 3))
                        self.total_scores.extend(pruning_scores)

                    elif isinstance(layer, torch.nn.BatchNorm2d):
                        self.fanin_scores[name] = weight_scores

                    elif isinstance(layer, torch.nn.Linear):
                        """
                        Since resnet has only one linear layer which is for classification, 
                        it is not involved in pruning
                        """
                        self.fanin_scores[name] = weight_scores.sum(dim=0)

    @torch.no_grad()
    def get_fanout_indices(self, min_channel_ratio=0.15, percentile=90):
        threshold = np.percentile(torch.stack(self.total_scores).cpu().numpy(), percentile)
        threshold = torch.tensor(threshold, device=self.prune_args.device)
        for tgt_name in self.fanout_scores:

            scores = self.fanout_scores[tgt_name]
            num_remain = (scores >= threshold).sum()

            minimum_channels = int(scores.shape[0] * min_channel_ratio)

            if num_remain <= minimum_channels:
                num_remain = minimum_channels

            self.fanout_indices[tgt_name] = get_channel_indices(scores,
                                                                num_remain,
                                                                device=self.prune_args.device)

    def block_pruning(self, block, in_channel_indices):

        self.fanin_indices[block[0]] = in_channel_indices  # The first conv of block
        last_indices = in_channel_indices
        in_channel_indices = self.fanout_indices[block[0]]

        if self.block_type == 'bottleneck':
            tmp_conv_cnt = 0
            downsample_cnt = 0
            for idx, a_layer in enumerate(block):
                if idx >= 1 and 'downsample' not in a_layer:
                    self.fanin_indices[a_layer] = in_channel_indices
                    m = get_module(self.model, a_layer)
                    # Second conv
                    if isinstance(m, torch.nn.Conv2d) and tmp_conv_cnt != 1:
                        output_indices = self.fanout_indices[a_layer]
                        in_channel_indices = output_indices
                        tmp_conv_cnt += 1
                    elif isinstance(m, torch.nn.Conv2d) and tmp_conv_cnt == 1:
                        self.fanout_indices[a_layer] = last_indices
                        in_channel_indices = self.fanout_indices[a_layer]
                    elif isinstance(m, torch.nn.BatchNorm2d) and tmp_conv_cnt == 1:
                        self.fanin_indices[a_layer] = in_channel_indices

                elif 'downsample' in a_layer:
                    # m = get_module(self.model, a_layer)  # Sequential

                    if downsample_cnt == 0:
                        # conv
                        self.fanin_indices[a_layer] = last_indices
                        self.fanout_indices[a_layer] = last_indices
                        downsample_cnt += 1

                    elif downsample_cnt == 1:
                        # bn
                        self.fanin_indices[a_layer] = last_indices

            return last_indices

        elif self.block_type == 'basicblock':
            downsample_cnt = 0
            for idx, a_layer in enumerate(block):
                if idx >= 1 and 'downsample' not in a_layer:
                    self.fanin_indices[a_layer] = in_channel_indices
                    m = get_module(self.model, a_layer)
                    # Second conv
                    if isinstance(m, torch.nn.Conv2d):
                        self.fanout_indices[a_layer] = last_indices
                        in_channel_indices = last_indices
                elif 'downsample' in a_layer:
                    m = get_module(self.model, a_layer)
                    if downsample_cnt == 0:
                        # conv
                        self.fanin_indices[a_layer] = last_indices
                        self.fanout_indices[a_layer] = last_indices
                        downsample_cnt += 1

                    elif downsample_cnt == 1:
                        # bn
                        self.fanin_indices[a_layer] = last_indices
            return last_indices
        else:
            raise AssertionError("Invalid block type. Check the block type !")

    def block_pruner(self, first_in_channel_indices):
        in_channel_indices = first_in_channel_indices
        for block in self.blocks:
            in_channel_indices = self.block_pruning(block, in_channel_indices)
        return in_channel_indices

    @torch.no_grad()
    def get_dependency(self):
        """
        The case of other layers to the resnet : conv1, bn1, fc
        """

        self.fanin_indices['conv1.weight'] = [i for i in range(self.input_image_channels)]
        first_in_channel_indices = self.fanout_indices['conv1.weight']
        self.fanin_indices['bn1.weight'] = first_in_channel_indices
        last_channel_indices = self.block_pruner(first_in_channel_indices=first_in_channel_indices)
        self.fanin_indices['fc.weight'] = last_channel_indices
        self.fanout_indices['fc.weight'] = [i for i in range(self.num_classes)]

    def replace_layers(self):
        for layer_name in self.others:
            m = get_module(self.model, layer_name)
            if isinstance(m, torch.nn.Conv2d):
                pruned_layer = replace_layer(layer=m,
                                             in_channel_indices=torch.tensor(self.fanin_indices[layer_name]),
                                             out_channel_indices=torch.tensor(self.fanout_indices[layer_name]))

                accelerate(self.model, layer_name, pruned_layer)
            elif isinstance(m, torch.nn.BatchNorm2d):
                # 헷갈림 방지를 위해 교정 필요
                pruned_layer = replace_layer(layer=m,
                                             out_channel_indices=torch.tensor(self.fanin_indices[layer_name]))
                accelerate(self.model, layer_name, pruned_layer)
            elif isinstance(m, torch.nn.Linear):
                pruned_layer = replace_layer(layer=m,
                                             in_channel_indices=torch.tensor(self.fanin_indices[layer_name]),
                                             out_channel_indices=torch.tensor(self.fanout_indices[layer_name]))
                accelerate(self.model, layer_name, pruned_layer)

        for block in self.blocks:
            downsample_cnt = 0
            for layer_name in block:

                if 'downsample' not in layer_name:
                    m = get_module(self.model, layer_name)
                    if isinstance(m, torch.nn.Conv2d):
                        pruned_layer = replace_layer(layer=m,
                                                     in_channel_indices=self.fanin_indices[layer_name],
                                                     out_channel_indices=self.fanout_indices[layer_name])
                        accelerate(self.model, layer_name, pruned_layer)

                    elif isinstance(m, torch.nn.BatchNorm2d):

                        pruned_layer = replace_layer(layer=m,
                                                     out_channel_indices=self.fanin_indices[layer_name])
                        accelerate(self.model, layer_name, pruned_layer)

                elif 'downsample' in layer_name:

                    m = get_module(self.model, layer_name[:-7])

                    if downsample_cnt == 0:

                        pruned_layer = replace_layer(layer=m[0],
                                                     in_channel_indices=self.fanin_indices[layer_name],
                                                     out_channel_indices=self.fanout_indices[layer_name])
                        accelerate(self.model, layer_name, pruned_layer)

                        downsample_cnt += 1
                    elif downsample_cnt == 1:

                        pruned_layer = replace_layer(layer=m[1],
                                                     out_channel_indices=self.fanin_indices[layer_name])
                        accelerate(self.model, layer_name, pruned_layer)

                        downsample_cnt += 1

    def do_pruning(self, dataloader):
        self.scoring(dataloader)
        self.get_fanout_indices(min_channel_ratio=self.prune_args.min_channel_ratio,
                                percentile=self.prune_args.percentile)
        self.get_dependency()
        self.replace_layers()

        return self.model


class toy_pruner(Default_Pruner):
    def __init__(self, args, model):
        super(toy_pruner, self).__init__(args, model)
        self.input_image_channels = args.input_image_channels
        self.num_classes = args.num_classes

    def scoring(self, dataloader):
        # Channel-wise scoring
        data, labels = next(iter(dataloader))
        output = self.model(data.to(self.prune_args.device))
        torch.mean(output).backward()

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if 'bias' not in name:
                    # downsample and bias are dependent to the 'weight'.
                    weight_scores = param * param.grad
                    layer = get_module(self.model, name)
                    if isinstance(layer, torch.nn.Linear):
                        """
                        Since resnet has only one linear layer which is for classification, 
                        it is not involved in pruning
                        """
                        self.fanin_scores[name] = weight_scores.sum(dim=0)
                        pruning_scores = weight_scores.sum(dim=1)
                        self.fanout_scores[name] = pruning_scores
                        self.total_scores.extend(pruning_scores)

    @torch.no_grad()
    def get_fanout_indices(self, min_channel_ratio=0.15, percentile=90):
        threshold = np.percentile(torch.stack(self.total_scores).cpu().numpy(), percentile)
        threshold = torch.tensor(threshold, device=self.prune_args.device)
        for tgt_name in self.fanout_scores:

            scores = self.fanout_scores[tgt_name]
            num_remain = (scores >= threshold).sum()

            minimum_channels = int(scores.shape[0] * min_channel_ratio)

            if num_remain <= minimum_channels:
                num_remain = minimum_channels

            self.fanout_indices[tgt_name] = get_channel_indices(scores,
                                                                num_remain,
                                                                device=self.prune_args.device)

    def get_dependency(self):
        self.fanin_indices['layer_1.weight'] = [i for i in range(self.input_image_channels)]
        self.fanin_indices['layer_2.weight'] = self.fanout_indices['layer_1.weight']
        self.fanin_indices['layer_3.weight'] = self.fanout_indices['layer_2.weight']
        self.fanout_indices['layer_3.weight'] = [i for i in range(self.num_classes)]

    def replace_layers(self):
        for layer_name in self.fanout_scores:
            m = get_module(self.model, layer_name)
            if isinstance(m, torch.nn.Linear):
                pruned_layer = replace_layer(layer=m,
                                             in_channel_indices=torch.tensor(self.fanin_indices[layer_name]),
                                             out_channel_indices=torch.tensor(self.fanout_indices[layer_name]))
                accelerate(self.model, layer_name, pruned_layer)

    def do_pruning(self, dataloader):
        self.scoring(dataloader)
        self.get_fanout_indices(min_channel_ratio=self.prune_args.min_channel_ratio,
                                percentile=self.prune_args.percentile)
        self.get_dependency()
        self.replace_layers()

        return self.model


class MobileNet_Pruner(Default_Pruner):
    def __init__(self, args, model):
        super(MobileNet_Pruner, self).__init__(args, model)


class VGG_Pruner(Default_Pruner):
    def __init__(self, args, model):
        super(VGG_Pruner, self).__init__(args, model)


class ViT_Pruner(Default_Pruner):
    def __init__(self, args, model):
        super(ViT_Pruner, self).__init__(args, model)


class BERT_Pruner(Default_Pruner):
    def __init__(self, args, model):
        super(BERT_Pruner, self).__init__(args, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--min_channel_ratio", type=float, default=0.15)
    parser.add_argument("--percentile", type=float, default=90)
    parser.add_argument("--input_image_channels", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--block_type", type=str, default="basicblock")
    args = parser.parse_args()
    from model_src.toy_model import toy_model
    # model = models.resnet50(num_classes=10)
    # model = ResNet_CIFAR(depth=56, num_classes=10, block_name=args.block_type)

    model = toy_model(first_channels=3*32*32)

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = ttf.Compose([
        ttf.RandomHorizontalFlip(),
        ttf.RandomCrop(32, 4),
        ttf.ToTensor(),
        ttf.Normalize(mean, std)])

    test_transform = ttf.Compose([
        ttf.ToTensor(),
        ttf.Normalize(mean, std)])

    dataset = CIFAR10(root="./", download=True, transform=train_transform)
    dataloader = DataLoader(dataset=dataset, batch_size=4)
    # Pruner = ResNet_Pruner(args=args, model=model, block_type=args.block_type)
    Pruner = toy_pruner(args=args, model=model)
    new_model = Pruner.do_pruning(dataloader)
    print(new_model)
    # print(new_model)
