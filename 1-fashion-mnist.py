import random
import torch
import os
import numpy as np  
from torch import Generator
from torch import save, device as Device, cuda, backends
from torch.nn import Module
from torch.nn import CrossEntropyLoss  
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.utils.data import DataLoader    
from torchsystem.registry import Registry

from src.models.perceptrons.MLP import MLPPerceptron
from src.models.perceptrons.GLU import GLUPerceptron
from src.models.perceptrons.SGLU import SGLUPerceptron
from src.models.perceptrons.DSGLU import DSGLUPerceptron
from src.models.perceptrons.TSGLU import TSGLUPerceptron
from src.datasets.fashion import Fashion
from src.training import iterate
from src.logging import Logger
from src.metrics import Metrics
from src.aggregate import Classifier
   
EPOCHS = 200 

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed(seed)
    cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def run(aggregate: Classifier, loaders: dict[str, DataLoader], logger: Logger):  
    path = f"weights/fashion/{aggregate.name}.pth"  
    if os.path.exists(path):
        print(f"Weights already exist for {aggregate.name}, skipping...")
        return 
       
    print(f"Running {aggregate.name}") 
    try:
        for epoch in range(EPOCHS): 
            for phase, loader in loaders.items():
                results = iterate(phase, aggregate, loader)
                logger.log(
                    epoch=epoch,
                    phase=phase,
                    lr=aggregate.lr,
                    **{key: float(value) if torch.is_tensor(value) else value for key, value in results.items()}
                )
            aggregate.scheduler.step()  
            logger.flush() 

    except Exception as exception:
        print(f"Error in {aggregate.name}: {exception}")
        raise exception
    else:
        aggregate.save(path)
        save(aggregate.state_dict(), path)
        print(f"Saved weights → {path}")
    finally:
        logger.close() 
  
registry = Registry[Module]()
registry.register(MLPPerceptron)
registry.register(GLUPerceptron)
registry.register(SGLUPerceptron) 
registry.register(DSGLUPerceptron)
registry.register(TSGLUPerceptron)

if __name__ == '__main__':    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False  
    torch.use_deterministic_algorithms(True)     
    for key, NN in registry.types.items():
        assert NN
        assert issubclass(NN, Module)
        for seed in [1,2,3]: 
            set_seed(seed)  
            cuda.empty_cache()  
            device    = Device("cuda" if cuda.is_available() else "cpu")
            nn        = NN(512, 0.3) 
            metrics   = Metrics(10)
            criterion = CrossEntropyLoss()
            optimizer = AdamW(nn.parameters(), lr=1e-3)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)  
            aggregate = Classifier(nn, criterion, optimizer, scheduler, metrics, device, seed).to(device)
            logger    = Logger(f"logs/fashion/{aggregate.name}.csv", ['epoch', 'phase', 'lr', 'loss', 'accuracy'])
            generator = Generator()
            generator.manual_seed(seed)  
            loaders = {
                "train": DataLoader(Fashion(train=True), batch_size=128, shuffle=True, pin_memory=True, num_workers=4, worker_init_fn=seed_worker, generator=generator),
                "evaluation": DataLoader(Fashion(train=False), batch_size=128, shuffle=False, pin_memory=True, num_workers=4, worker_init_fn=seed_worker, generator=generator)
            }  
            run(aggregate, loaders, logger)