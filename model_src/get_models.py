from model_src.toy_model import toy_model
from model_src.resnet_cifar import ResNet_CIFAR
from model_src.mobilenetv2_cifar import MobileNetV2_CIFAR
from model_src.vgg16_bn import vgg16_bn
from torchvision.models import MobileNetV2

AVAILABLE_MODEL_NAMES = ["resnet", "toy_model", "mobilenetv2-cifar", "mobilenetv2"]


def get_model(model_name, device, **kwargs):
    """
    :param model_name: Literally, it is model name. available model names are follows.
    resnet, toy_model, mobilenetv2
    :return:
    """
    if model_name == "resnet":
        return ResNet_CIFAR(**kwargs).to(device)
    elif model_name == "mobilenetv2-cifar":
        return MobileNetV2_CIFAR(**kwargs).to(device)
    elif model_name == "mobilenetv2":
        return MobileNetV2(**kwargs).to(device)
    elif model_name == "toy_model":
        return toy_model(**kwargs).to(device)
    elif model_name == "vgg16_bn":
        return vgg16_bn(**kwargs).to(device)
    else:
        raise ValueError("It is not proper model name")


def get_model_args(model_name, input_image_channels, num_classes) -> dict:
    args = {"num_classes": num_classes,
            "input_image_channels": input_image_channels}
    if model_name == "resnet":
        args["depth"] = 56
        args["block_name"] = "BasicBlock"
    return args