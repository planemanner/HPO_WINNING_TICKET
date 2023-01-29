import torch

__all__ = ["get_channel_indices", "get_module", "block_parser", "accelerate", "replace_layer"]
DEFINED_OPERATION_TYPES = ["CONV1D", "CONV2D", "LINEAR", "BATCH_NORM2D", "LAYER_NORM"]


class get_parsed_weight:

    @staticmethod
    def get_parsed_linear_weight(in_channel_indices: torch.Tensor, out_channel_indices: torch.Tensor,
                                 weight: torch.Tensor):
        old_out_dim, old_in_dim = weight.shape
        new_out_dim, new_in_dim = len(out_channel_indices), len(in_channel_indices)

        in_channel_mask, out_channel_mask = torch.zeros_like(weight, device=weight.device), torch.zeros_like(weight,
                                                                                                             device=weight.device)

        in_channel_indices = in_channel_indices.contiguous().repeat(old_out_dim).view(old_out_dim,
                                                                                      new_in_dim)
        in_channel_indices = in_channel_indices.reshape(old_out_dim, new_in_dim)
        out_channel_indices = out_channel_indices.contiguous().repeat(old_in_dim).view(old_in_dim, new_out_dim)
        out_channel_indices = out_channel_indices.permute(1, 0)

        in_channel_mask = in_channel_mask.scatter_(1, in_channel_indices, 1.).bool()
        out_channel_mask = out_channel_mask.scatter_(0, out_channel_indices, 1).bool()

        total_mask = in_channel_mask & out_channel_mask

        return weight[total_mask].reshape(new_out_dim, new_in_dim)

    @staticmethod
    def get_parsed_normalization_weight(out_channel_indices: torch.Tensor, weight: torch.Tensor):
        # batch norm and layer norm
        out_channel_mask = torch.zeros_like(weight, device=weight.device)
        out_channel_mask = out_channel_mask.scatter_(0, out_channel_indices, 1.).bool()
        total_mask = out_channel_mask

        return weight[total_mask]

    @staticmethod
    def get_parsed_conv1d_weight(in_channel_indices: torch.Tensor, out_channel_indices: torch.Tensor,
                                 weight: torch.Tensor):
        old_out_dim, old_in_dim, kw = weight.shape
        new_out_dim, new_in_dim = len(out_channel_indices), len(in_channel_indices)
        in_channel_mask, out_channel_mask = torch.zeros_like(weight, device=weight.device), torch.zeros_like(weight,
                                                                                                             device=weight.device)
        in_channel_indices = in_channel_indices.contiguous().repeat(old_out_dim).view(old_out_dim,
                                                                                      new_in_dim).expand(kw,
                                                                                                         old_out_dim,
                                                                                                         new_in_dim)
        in_channel_indices = in_channel_indices.reshape(old_out_dim, new_in_dim, kw)
        # operation order is very important.
        out_channel_indices = out_channel_indices.contiguous().repeat(old_in_dim).view(old_in_dim,
                                                                                       new_out_dim).expand(kw,
                                                                                                           old_in_dim,
                                                                                                           new_out_dim)
        out_channel_indices = out_channel_indices.permute(2, 1, 0)

        in_channel_mask = in_channel_mask.scatter_(1, in_channel_indices, 1.).bool()
        out_channel_mask = out_channel_mask.scatter_(0, out_channel_indices, 1).bool()

        total_mask = in_channel_mask & out_channel_mask

        return weight[total_mask].reshape(new_out_dim, new_in_dim, kw)

    @staticmethod
    def get_parsed_conv2d_weight(in_channel_indices: torch.Tensor, out_channel_indices: torch.Tensor,
                                 weight: torch.Tensor):
        old_out_dim, old_in_dim, kw, kh = weight.shape

        new_out_dim, new_in_dim = len(out_channel_indices), len(in_channel_indices)
        in_channel_mask, out_channel_mask = torch.zeros_like(weight, device=weight.device), torch.zeros_like(weight,
                                                                                                             device=weight.device)

        in_channel_indices = in_channel_indices.contiguous().repeat(old_out_dim).view(old_out_dim,
                                                                                      new_in_dim).expand(kh,
                                                                                                         kw,
                                                                                                         old_out_dim,
                                                                                                         new_in_dim)
        in_channel_indices = in_channel_indices.permute(2, 3, 0, 1)
        # operation order is very important.
        out_channel_indices = out_channel_indices.contiguous().repeat(old_in_dim).view(old_in_dim, new_out_dim).expand(
            kw,
            kh,
            old_in_dim,
            new_out_dim)

        out_channel_indices = out_channel_indices.permute(3, 2, 1, 0)
        in_channel_mask = in_channel_mask.scatter_(1, in_channel_indices, 1.).bool()
        out_channel_mask = out_channel_mask.scatter_(0, out_channel_indices, 1).bool()

        total_mask = in_channel_mask & out_channel_mask

        return weight[total_mask].reshape(new_out_dim, new_in_dim, kh, kw)

    @staticmethod
    def get_parsed_bias(out_channel_indices: torch.Tensor, weight: torch.Tensor):
        bias_mask = torch.zeros_like(weight, device=weight.device)
        bias_mask = bias_mask.scatter_(0, out_channel_indices, 1.).bool()
        total_mask = bias_mask

        return weight[total_mask]

    def get_weight(self, layer_type, **kwmodule_attr_name):
        if layer_type == "CON1D":
            return self.get_parsed_conv1d_weight(**kwmodule_attr_name)
        elif layer_type == "CONV2D":
            return self.get_parsed_conv2d_weight(**kwmodule_attr_name)
        elif layer_type == "LINEAR":
            return self.get_parsed_linear_weight(**kwmodule_attr_name)
        elif layer_type in ["BATCH_NORM2D", "LAYER_NORM"]:
            return self.get_parsed_normalization_weight(**kwmodule_attr_name)
        else:
            raise AssertionError("It is not valid operation type of pruning. Check the layer type.")


def get_channel_indices(scores: torch.Tensor, num_remain, device):
    """
    :param scores: This must have same indices corresponding channels of a layer.
    :param num_remain: It indicates the number of channels after pruning.
    :return: tensor indices
    """
    channel_dim = scores.shape[0]
    if num_remain == channel_dim:
        return torch.tensor([i for i in range(channel_dim)], device=device)
    else:
        # automatically loaded on same device of scores.

        return torch.where(torch.argsort(scores) >= channel_dim - num_remain)[0]


def assign_new_layer(old_layer: torch.nn.Module, new_outdim: int, new_indim: int, groups=1):
    if hasattr(old_layer, 'bias') and old_layer.bias is not None:
        bias_flag = True
    else:
        bias_flag = False

    if isinstance(old_layer, torch.nn.Conv2d) or isinstance(old_layer, torch.nn.Conv1d):
        if groups > 1:
            new_layer = torch.nn.Conv2d(in_channels=groups,
                                        out_channels=new_outdim,
                                        kernel_size=old_layer.kernel_size,
                                        stride=old_layer.stride,
                                        padding=old_layer.padding,
                                        bias=bias_flag,
                                        groups=groups,
                                        device=old_layer.weight.device)
        elif groups == 1:
            new_layer = torch.nn.Conv2d(in_channels=new_indim,
                                        out_channels=new_outdim,
                                        kernel_size=old_layer.kernel_size,
                                        stride=old_layer.stride,
                                        padding=old_layer.padding,
                                        bias=bias_flag,
                                        device=old_layer.weight.device)
        else:
            raise ValueError("You should provide proper value of 'groups'.")

    elif isinstance(old_layer, torch.nn.BatchNorm2d):

        assert new_outdim == new_indim, "new_indim and new_indim are same to avoid a confusion."
        new_layer = torch.nn.BatchNorm2d(new_outdim)

    elif isinstance(old_layer, torch.nn.LayerNorm):

        assert new_outdim == new_indim, "new_indim and new_indim are same to avoid a confusion."
        new_layer = torch.nn.LayerNorm(new_outdim)

    elif isinstance(old_layer, torch.nn.Linear):
        new_layer = torch.nn.Linear(in_features=new_indim,
                                    out_features=new_outdim,
                                    bias=bias_flag,
                                    device=old_layer.weight.device)
    else:
        raise AssertionError("This type operation is not compatible with our package")

    return new_layer


def replace_layer(layer: torch.nn.Module, in_channel_indices=None, out_channel_indices=None, groups=1):
    weight_parser = get_parsed_weight()
    if isinstance(layer, torch.nn.Conv2d):
        weight = weight_parser.get_parsed_conv2d_weight(in_channel_indices=in_channel_indices,
                                                        out_channel_indices=out_channel_indices,
                                                        weight=layer.weight.data)
        if layer.bias is not None:
            bias = weight_parser.get_parsed_bias(out_channel_indices=out_channel_indices, weight=layer.bias.data)
        else:
            bias = None

        pruned_layer = assign_new_layer(old_layer=layer, new_outdim=len(out_channel_indices),
                                        new_indim=len(in_channel_indices), groups=groups)

    elif isinstance(layer, torch.nn.Conv1d):
        weight = weight_parser.get_parsed_conv1d_weight(in_channel_indices=in_channel_indices,
                                                        out_channel_indices=out_channel_indices,
                                                        weight=layer.weight.data)
        if layer.bias is not None:
            bias = weight_parser.get_parsed_bias(out_channel_indices=out_channel_indices, weight=layer.bias.data)
        else:
            bias = None

        pruned_layer = assign_new_layer(old_layer=layer, new_outdim=len(out_channel_indices),
                                        new_indim=len(in_channel_indices), groups=groups)

    elif any([isinstance(layer, torch.nn.LayerNorm), isinstance(layer, torch.nn.BatchNorm2d)]):
        weight = weight_parser.get_parsed_normalization_weight(out_channel_indices=out_channel_indices,
                                                               weight=layer.weight.data)
        if layer.bias is not None:
            bias = weight_parser.get_parsed_bias(out_channel_indices=out_channel_indices,
                                                 weight=layer.bias.data)
        else:
            bias = None

        pruned_layer = assign_new_layer(old_layer=layer,
                                        new_outdim=len(out_channel_indices),
                                        new_indim=len(out_channel_indices))

    elif isinstance(layer, torch.nn.Linear):
        weight = weight_parser.get_parsed_linear_weight(in_channel_indices=in_channel_indices,
                                                        out_channel_indices=out_channel_indices,
                                                        weight=layer.weight.data)
        if layer.bias is not None:
            bias = weight_parser.get_parsed_bias(out_channel_indices=out_channel_indices, weight=layer.bias.data)
        else:
            bias = None

        pruned_layer = assign_new_layer(old_layer=layer,
                                        new_outdim=len(out_channel_indices),
                                        new_indim=len(in_channel_indices))
    else:
        raise AssertionError("This type operation is not compatible with our package")

    pruned_layer.weight.data.copy_(weight)
    if bias is not None:
        pruned_layer.bias.data.copy_(bias)
    return pruned_layer


def get_module(model, attr_name, level=1):
    attr_name = attr_name.split('.')
    wrap_cnt = 0
    total_wrap = len(attr_name)
    module = model
    while (total_wrap - level) > wrap_cnt:
        module = getattr(module, attr_name[wrap_cnt])
        wrap_cnt += 1
    return module


def accelerate(model, module_attr_name, new_layer):
    parsed_name = module_attr_name.split('.')

    if len(parsed_name) == 2:
        setattr(get_module(model, module_attr_name, level=2), parsed_name[0], new_layer)
    elif len(parsed_name) == 3:
        upper = getattr(model, parsed_name[0])
        setattr(upper, parsed_name[1], new_layer)
    elif len(parsed_name) == 4:
        upper = getattr(model, parsed_name[0])
        middle = getattr(upper, parsed_name[1])
        setattr(middle, parsed_name[2], new_layer)
    elif len(parsed_name) == 5:
        upper = getattr(model, parsed_name[0])
        middle = getattr(upper, parsed_name[1])
        inner = getattr(middle, parsed_name[2])
        setattr(inner, parsed_name[3], new_layer)
    elif len(parsed_name) == 6:
        upper = getattr(model, parsed_name[0])
        middle_1 = getattr(upper, parsed_name[1])
        middle_2 = getattr(middle_1, parsed_name[2])
        inner = getattr(middle_2, parsed_name[3])
        setattr(inner, parsed_name[4], new_layer)
    elif len(parsed_name) == 7:
        upper = getattr(model, parsed_name[0])
        middle_1 = getattr(upper, parsed_name[1])
        middle_2 = getattr(middle_1, parsed_name[2])
        middle_3 = getattr(middle_2, parsed_name[3])
        inner = getattr(middle_3, parsed_name[4])
        setattr(inner, parsed_name[5], new_layer)


class Node:

    def __init__(self, op, name, score, op_type, parent, children, division):
        self.op = op  # operation
        self.name = name  # Layer name
        self.score = score  # Layer's score
        self.op_type = op_type  # Operation Type. ex) Conv2d, BN, LN...
        self.parent = parent  # previous layer or module
        self.children = children  # next layer or related operation after this layer.
        self.division = division  # It represents the type of this block and location and relationship


def block_parser(model, block):
    """
    :param model: a neural network
    :param block: BasicBlock, Bottleneck, and...
    :return:
    """
    blocks = []
    others = []
    for idx, (prefix, m) in enumerate(model.named_children()):
        if isinstance(m, torch.nn.Sequential):
            for idy, (sub_prefix, mm) in enumerate(m.named_children()):
                if isinstance(mm, block):
                    tmp_block = []
                    for idz, (name, mmm) in enumerate(mm.named_children()):
                        real_name = prefix + '.' + sub_prefix + '.' + name + '.' + 'weight'
                        if isinstance(mmm, torch.nn.Conv2d):
                            tmp_block += [real_name]
                        elif isinstance(mmm, torch.nn.BatchNorm2d):
                            tmp_block += [real_name]
                        elif isinstance(mmm, torch.nn.Linear):
                            tmp_block += [real_name]
                        elif isinstance(mmm, torch.nn.Sequential):
                            # In the case of downsample layer
                            for idh, (name_, mmmm) in enumerate(mmm.named_children()):

                                if isinstance(mmmm, torch.nn.Conv2d):
                                    real_name = prefix + '.' + sub_prefix + '.' + name + '.' + str(0) + '.' + 'weight'
                                    tmp_block += [real_name]
                                elif isinstance(mmmm, torch.nn.BatchNorm2d):
                                    real_name = prefix + '.' + sub_prefix + '.' + name + '.' + str(1) + '.' + 'weight'
                                    tmp_block += [real_name]

                    blocks += [tmp_block]

        elif isinstance(m, torch.nn.Conv2d):
            others += [prefix + '.weight']
        elif isinstance(m, torch.nn.BatchNorm2d):
            others += [prefix + '.weight']
        elif isinstance(m, torch.nn.Linear):
            others += [prefix + '.weight']

    return blocks, others