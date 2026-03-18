from math import sqrt 
from typing import Optional
from torch import zeros, concat 
from torch import Tensor  
from torch.nn import Module, Parameter, ModuleList
from torch.nn import Linear, LayerNorm, Conv2d
from torch.nn import Dropout
from torch import sigmoid
from torch.nn.functional import scaled_dot_product_attention as attention

class Perceptron(Module): 
    def __init__(
        self,
        model_dimension : int, 
        hidden_dimension: int,
        dropout_rate: float 
    )-> None:
        super().__init__()
        self.norm = LayerNorm(model_dimension, eps=1e-6) 
        self.z_projector = Linear(model_dimension, hidden_dimension)
        self.v_projector = Linear(model_dimension, hidden_dimension)
        self.o_projector = Linear(hidden_dimension, model_dimension)
        self.dropout = Dropout(dropout_rate)

    def forward(self, features): 
        features = self.norm(features)  
        features =  self.v_projector(features) * sigmoid(self.z_projector(features))
        features = self.dropout(features)
        features = self.o_projector(features)
        return self.dropout(features)

class Attention(Module): 
    def __init__(
        self, 
        model_dimension: int, 
        number_of_heads: int,
        dropout_rate: float
    )-> None:
        super().__init__()
        assert model_dimension % number_of_heads == 0
        self.norm  = LayerNorm(model_dimension, eps=1e-6)
        self.q_projector = Linear(model_dimension, model_dimension)
        self.k_projector = Linear(model_dimension, model_dimension)
        self.v_projector = Linear(model_dimension, model_dimension)  
        self.o_projector = Linear(model_dimension, model_dimension)
        self.dropout = Dropout(dropout_rate)
        self.number_of_heads = number_of_heads  

    def split(self, sequence: Tensor) -> Tensor:
        batch_size, sequence_length, model_dimension = sequence.shape 
        return sequence.view(batch_size, sequence_length, self.number_of_heads, model_dimension // self.number_of_heads).transpose(1, 2)
 
    def merge(self, sequence: Tensor) -> Tensor: 
        sequence = sequence.transpose(1, 2)
        return sequence.reshape(*sequence.shape[:-2], -1) 

 
    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        sequence = self.norm(sequence)

        q = self.q_projector(sequence)
        k = self.k_projector(sequence)
        v = self.v_projector(sequence) 

        q, k, v = self.split(q), self.split(k), self.split(v) 

        sequence = attention(q, k, v, mask if mask else None)
        sequence = self.merge(sequence)
        sequence = self.o_projector(sequence)
        return self.dropout(sequence)

class Encoder(Module): 
    def __init__(
        self, 
        model_dimension : int, 
        number_of_heads : int, 
        hidden_dimension: int, 
        dropout_rate: float
    )-> None:
        super().__init__()
        self.perceptron = Perceptron(model_dimension, hidden_dimension, dropout_rate) 
        self.attention  = Attention(model_dimension, number_of_heads, dropout_rate) 

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor: 
        sequence = sequence + self.attention(sequence, mask) 
        sequence = sequence + self.perceptron(sequence) 
        return sequence


class Transformer(Module): 
    def __init__(
        self, 
        number_of_layers: int, 
        model_dimension : int, 
        number_of_heads : int, 
        hidden_dimension: int,
        dropout_rate: float
    )-> None:
        super().__init__()
        self.encoders = ModuleList([
            Encoder(model_dimension, number_of_heads, hidden_dimension, dropout_rate) for _ in range(number_of_layers)
        ])

    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        for encoder in self.encoders:
            sequence = encoder(sequence, mask)
        return sequence
    

class Positions(Module): 
    def __init__(
        self, 
        sequence_lenght: int,
        model_dimension: int
    )-> None:
        super().__init__()
        self.embeddings = Parameter(zeros(1, sequence_lenght, model_dimension))
    
    def forward(self, sequence: Tensor) -> Tensor: 
        return sequence + self.embeddings
    

class Labels(Module):
    def __init__(
        self, 
        model_dimension: int
    )-> None:
        super().__init__()
        self.embeddings = Parameter(zeros(1, 1, model_dimension)) 

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = sequence.flatten(2).transpose(1, 2) 
        return concat((self.embeddings.expand(sequence.size(0), -1, -1), sequence), dim=1) 


class GLUViT(Module):  
    def __init__(
        self,  
        image_size: tuple[int, int], 
        patch_size: tuple[int, int],
        model_dimension   : int,
        hidden_dimension  : int, 
        number_of_heads   : int,
        number_of_layers  : int,
        number_of_classes : int, 
        number_of_channels: int,
        dropout_rate: float 
    )-> None:
        super().__init__()    
        self.image_size = image_size                 
        h, w = image_size
        fh, fw = patch_size
        gh, gw = h // fh, w // fw    
 
        self.patcher   = Conv2d(number_of_channels, model_dimension, kernel_size=(fh, fw), stride=(fh, fw))
        self.labels    = Labels(model_dimension) 
        self.positions = Positions(gh * gw + 1, model_dimension)
         
        self.transformer = Transformer(
            number_of_layers=number_of_layers, 
            model_dimension=model_dimension, 
            number_of_heads=number_of_heads, 
            hidden_dimension=hidden_dimension,
            dropout_rate=dropout_rate
        ) 
 
        self.norm = LayerNorm(model_dimension, eps=1e-6)
        self.head = Linear(model_dimension, number_of_classes)  
        self.dropout = Dropout(dropout_rate)
          
    def forward(self, sequence: Tensor) -> Tensor:  
        if sequence.dim() == 3:
            sequence = sequence.unsqueeze(0)
        sequence = self.patcher(sequence)  
        sequence = self.labels(sequence)  
        sequence = self.positions(sequence)   
        sequence = self.dropout(sequence)
        sequence = self.transformer(sequence) 
        sequence = self.norm(sequence)[:, 0]  
        sequence = self.dropout(sequence)
        sequence = self.head(sequence)  
        return sequence  