from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import (
    Compose, 
    ToTensor, 
    Normalize, 
    RandomHorizontalFlip
)
 
class Fashion(Dataset):
    def __init__(self, train: bool) -> None:
        super().__init__() 

        if train: 
            self.transform = Compose([
                RandomHorizontalFlip(p=0.5), 
                ToTensor(),
                Normalize(mean=(0.2860,), std=(0.3530,))
            ])
        else:
            self.transform = Compose([
                ToTensor(),
                Normalize(mean=(0.2860,), std=(0.3530,))
            ])

        self.dataset = FashionMNIST(
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