from collections.abc import Callable
from typing import Any
from typing import Iterable
from typing import Protocol 

from torch import Tensor 
from torch import argmax
from torch import device as Device
from torch import no_grad
from torch.nn import Module
from torch.optim import Optimizer 
from src.metrics import Metrics
from src.aggregate import Classifier 
  
def fit(
    model: Module,
    criterion: Callable[[Tensor, Tensor], Tensor],
    optimizer: Optimizer,
    device: Device,
    loader: Iterable[tuple[Tensor, Tensor]],
    metrics: Metrics,
) -> dict[str, Tensor]:
    model.train()
    metrics.reset()

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        output = model(features)            
        loss = criterion(output, targets)    
        loss.backward()
        optimizer.step()  
        metrics.update(loss, argmax(output, dim=1), targets) 
    return metrics.compute()

def evaluate(
    model: Module,
    criterion: Callable[[Tensor, Tensor], Tensor],
    device: Device,
    loader: Iterable[tuple[Tensor, Tensor]],
    metrics: Metrics,
)-> dict[str, Tensor]:
    model.eval()
    metrics.reset()

    with no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            output = model(features)           
            loss = criterion(output, targets)    
            metrics.update(loss, argmax(output, dim=1), targets) 
    return metrics.compute()

def iterate(
    phase: str, 
    aggregate: Classifier, 
    loader: Iterable[tuple[Tensor, Tensor]]
) -> dict[str, Tensor]:
    
    if phase == 'train':
        results = fit(aggregate, aggregate.criterion, aggregate.optimizer, aggregate.device, loader, aggregate.metrics)
        return results
    else:
        results = evaluate(aggregate, aggregate.criterion,aggregate.device, loader, aggregate.metrics)
        return results 