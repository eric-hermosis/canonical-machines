from torch import Tensor
from torch import sigmoid, tanh
from torch import zeros
from torch.nn import Module
from torch.nn import Linear 
from torch.nn import Parameter, LayerNorm 
from torch.nn import Flatten, Sequential, Dropout
 
class TSGLU(Module):
    def __init__(
        self,
        input_dimension : int, 
        output_dimension: int
    )-> None:
        super().__init__()   
        self.norm = LayerNorm(input_dimension)
        self.theta = Parameter(zeros(1,1))
        self.z_projector = Linear(input_dimension, output_dimension)
        self.v_projector = Linear(input_dimension, output_dimension) 

    def forward(self, features: Tensor) -> Tensor:     
        features = self.norm(features)
        z = self.z_projector(features)
        v = self.v_projector(features)
        return v * sigmoid(z - self.theta)  * tanh(z)
    
    @property
    def theta_0(self):
        return self.theta.item()  

class TSGLUPerceptron(Module):
    def __init__(
        self, 
        hidden_dimension: int, 
        dropout_rate: float
    ) -> None:
        super().__init__()
        self.flatten = Flatten()    
        self.layers = Sequential(
            TSGLU(784, hidden_dimension),
            Dropout(dropout_rate),
            TSGLU(hidden_dimension, hidden_dimension // 2),
            Dropout(dropout_rate), 
            Linear(hidden_dimension // 2, 10)
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(self.flatten(features))