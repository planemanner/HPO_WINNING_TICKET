import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.datasets import CIFAR100, MNIST, STL10
from torchvision.datasets import ImageFolder
import numpy as np
from PIL import Image
import os


class transform_wrapper:
    def __init__(self, albumentation_augmentor):
        self.augmentor = albumentation_augmentor

    def __call__(self, image: Image.Image) -> np.ndarray:
        img_arr = np.array(image)
        aug_img_arr = self.augmentor(image=img_arr)["image"]
        return aug_img_arr


def get_transform(dataset_name: str):

    if dataset_name == "CIFAR100":
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)

        train_transform = A.Compose([
            A.HorizontalFlip(),
            A.CenterCrop(28, 28, p=0.5),
            A.Resize(32, 32),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        test_transform = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()])

    elif dataset_name == "MNIST":
        mean = 0.5
        std = 0.5

        train_transform = A.Compose([
            A.HorizontalFlip(),
            A.Resize(36, 36),
            A.CenterCrop(32, 32),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        test_transform = A.Compose([
            A.Resize(32, 32),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()])

    elif dataset_name == "TINYIMAGENET":
        image_size = 128
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        train_transform = A.Compose([
            A.Resize(image_size+6, image_size+6),
            A.HorizontalFlip(),
            A.Rotate(15),
            A.CenterCrop(image_size, image_size),
            ToTensorV2(),
            A.Normalize(mean=mean, std=std)
        ])

        test_transform = A.Compose([
            A.Resize(image_size+16, image_size+16),
            A.CenterCrop(image_size, image_size),
            ToTensorV2(),
            A.Normalize(mean=mean, std=std)
        ])
    elif dataset_name == "STL10":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)

        train_transform = A.Compose([
            A.HorizontalFlip(),
            A.Resize(108, 108),
            A.CenterCrop(96, 96),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

        test_transform = A.Compose([
            A.Resize(96, 96),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()])
    else:
        raise NameError("It is invalid dataset name.")

    return transform_wrapper(train_transform), transform_wrapper(test_transform)


def call_dataset(dataset_name: str, root_path: str, download=False):
    dataset_name = dataset_name.upper()
    train_transform, test_transform = get_transform(dataset_name=dataset_name)

    if dataset_name == "TINYIMAGENET":
        train_set = ImageFolder(os.path.join(root_path, "train"), train_transform)
        test_set = ImageFolder(os.path.join(root_path, "val"), test_transform)

    elif dataset_name == "CIFAR100":
        train_set = CIFAR100(root=root_path, train=True, transform=train_transform, download=download)
        test_set = CIFAR100(root=root_path, train=False, transform=test_transform, download=download)

    elif dataset_name == "MNIST":
        train_set = MNIST(root=root_path, train=True, transform=train_transform, download=download)
        test_set = MNIST(root=root_path, train=False, transform=test_transform, download=download)

    elif dataset_name == "STL10":
        train_set = STL10(root=root_path, split="train", transform=train_transform, download=download)
        test_set = STL10(root=root_path, split="test", transform=test_transform, download=download)

    else:
        raise NameError("It is invalid dataset name. Please check your dataset name.")

    return train_set, test_set




train, test = call_dataset(dataset_name="STL10", root_path="./", download=True)