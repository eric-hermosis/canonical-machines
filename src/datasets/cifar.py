from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip
)

class CIFAR(Dataset):
    def __init__(self, train: bool) -> None:
        super().__init__()

        self.transform = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(p=0.5), 
            AutoAugment(AutoAugmentPolicy.CIFAR10),
            ToTensor(),
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )

        ]) if train else Compose([
            ToTensor(),
            Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616)
            )
        ])

        self.dataset = CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=self.transform
        )

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        image, label = self.dataset[index]
        return image, label

    def __len__(self) -> int:
        return len(self.dataset) 