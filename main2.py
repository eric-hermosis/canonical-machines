import random
import torch
import os
import numpy as np  
from torch import Generator
from torch.nn import CrossEntropyLoss 
from torch import save, device as Device, cuda, backends
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchsystem.registry import register 
from torch.utils.data import DataLoader   

from src.models.perceptrons.GLU import GLUPerceptron
from src.models.perceptrons.SGLU import SGLUPerceptron
from src.models.perceptrons.DSGLU import DSGLUPerceptron
from src.datasets.fashion import Fashion
from src.training import iterate
from src.logging import Logger
from src.metrics import Metrics
from src.aggregate import Classifier 

from torchsystem.registry import Registry

registry = Registry()
registry.register(GLUPerceptron)
registry.register(SGLUPerceptron)
registry.register(DSGLUPerceptron) 
 
if __name__ == '__main__':     
    nn        = registry.get('GLUPerceptron')(512, 0.3) 
    print(nn)