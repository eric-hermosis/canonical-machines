import os
from torch import Tensor
from torch import device as Device
from torch import save
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler as Scheduler
from torchsystem.registry import getname, getarguments
from src.metrics import Metrics

class Classifier(Module):
    def __init__(
        self, 
        nn: Module, 
        criterion: Module, 
        optimizer: Optimizer,  
        scheduler: Scheduler, 
        metrics  : Metrics,
        device: Device,
        seed: int
    ) -> None:
        super().__init__()
        self.nn = nn
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics   = metrics.to(device)
        self.device = device
        self.seed = seed

    def forward(self, features: Tensor) -> Tensor:
        return self.nn(features)
    
    @property
    def lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"] 
     
    @property
    def name(self) -> str: 
        aliases = { 
            "hidden_dimension": "hdim",  
            "dropout_rate": "p"
        } 
        parts = [f"{getname(self.nn)}-s{self.seed}"]
        for key, value in sorted(getarguments(self.nn).items()):
            key = aliases.get(key, key)
            value = "x".join(map(str, value)) if isinstance(value, tuple) else value
            parts.append(f"{key}={value}")
        return "-".join(parts)
    
    def save(self, path: str): 
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save(self.nn.state_dict(), path)