# Model implemented as classes
import jax
import jax.numpy as jnp
from jax.nn import tanh, relu, leaky_relu, silu
from jax.nn.initializers import glorot_normal
from flax import linen as nn
from typing import Any, Callable, Sequence
from initialization import *
jax.config.update('jax_enable_x64', True)

class MLP(nn.Module):
    features: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
            
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        if x.ndim > 1:
            x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, kernel_init=glorot_normal(), name=f'Dense_Layer_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                x = jnp.sin(x)
            else:
                x = self.activation(x)
        return x
    
class res_MLP(nn.Module):
    features: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
    
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        if x.ndim > 1:
            x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, kernel_init=glorot_normal(), name=f'Dense_Layer_with_Skip_Conn_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                if i+1 < len(self.features) and feat == self.features[i+1]:
                    x = x + jnp.sin(x) # skip connection where L_i == L_i+1
                else:
                    x = jnp.sin(x)
            else:
                if i+1 < len(self.features) and feat == self.features[i+1]:
                    x = x + self.activation(x) # skip connection where L_i == L_i+1
                else:
                    x = self.activation(x)
        return x
    
class CNN_MLP(nn.Module): # NHWC
    features_CNN: Sequence[list] # [#outC, (kernel), (stride), (padding)]
    features_MLP: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
  
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        for i, feat in enumerate(self.features_CNN):
            x = nn.Conv(features=feat[0], kernel_size=feat[1], strides=feat[2], padding=feat[3], name=f'Conv_Layer_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                x = jnp.sin(x)
            else:
                x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features_MLP):
            x = nn.Dense(features=feat, kernel_init=glorot_normal(), name=f'Dense_Layer_{i}',dtype=jnp.float64)(x)
            x = self.activation(x)
        return x
    
class CNN_res_MLP(nn.Module):
    features_CNN: Sequence[list] # [#outC, (kernel), (stride)]
    features_MLP: Sequence[int]
    activation: Callable
    init_scale: float
    init_sine: bool
  
    @nn.compact
    def __call__(self, inputs):
        x = self.init_scale*inputs
        for i, feat in enumerate(self.features_CNN):
            x = nn.Conv(features=feat[0], kernel_size=feat[1], strides=feat[2], padding='VALID', name=f'Conv_Layer_{i}',dtype=jnp.float64)(x)
            if i == 0 and self.init_sine == True:
                x = jnp.sin(x)
            else:
                x = self.activation(x)
        x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features_MLP):
            x = nn.Dense(feat, kernel_init=glorot_normal(), name=f'Dense_Layer_with_Skip_Conn_{i}',dtype=jnp.float64)(x)
            if i+1 < len(self.features_MLP) and feat == self.features_MLP[i+1]:
                x = x + self.activation(x) # skip connection where L_i == L_i+1
            else:
                x = self.activation(x)
        return x        

lrelu = lambda x: leaky_relu(x,negative_slope=0.1)
    
## Structure
C = 150 # Number of hidden layer units
L = 3 # Number of layers

## Branches
branch_layers_1 = [C] * (L-1)
branch_layers_2 = [C] * (L-1)

## Trunk
trunk_layers =  [C] * (L)

## Activation, etc (All fixed)
activation_branch_1 = silu 
init_scale_branch_1 = 1
init_sine_branch_1 = False

activation_branch_2 = silu
init_scale_branch_2 = 1
init_sine_branch_2 = False

activation_trunk = silu
init_scale_trunk = 1
init_sine_trunk = False

## Network construction
model_settings = [(branch_layers_1,activation_branch_1,init_scale_branch_1,init_sine_branch_1),
                  (branch_layers_2,activation_branch_2,init_scale_branch_2,init_sine_branch_2),
                  (trunk_layers,activation_trunk,init_scale_trunk,init_sine_trunk)]

branch_model_1 = MLP(*model_settings[0])
branch_model_2 = MLP(*model_settings[1])
trunk_model = MLP(*model_settings[2])