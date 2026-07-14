# Network and computations
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from models import branch_model_1, branch_model_2, trunk_model
jax.config.update('jax_enable_x64', True)

# Define MIONet architecture with Doubly batched input; u and (x,y) have totally individual batched input
@jit
def operator_net(params, uf, ug, xy):
    B1 = branch_model_1.apply(params['branch1'], uf) ## P1 by J1
    B2 = branch_model_2.apply(params['branch2'], ug) ## P2 by J2
    T = trunk_model.apply(params['trunk'], xy) ## Q by I
    W1 = params['last1'] ## J1 by I (already transposed)
    W2 = params['last2'] ## J2 by I (already transposed)
    BW1 = B1 @ W1 ## P1 by I
    BW2 = B2 @ W2 ## P2 by I
    BO = jnp.vstack([jnp.kron(BW1[:, k], BW2[:, k]) for k in range(BW2.shape[1])]).T ## P1P2 by I
    outputs = BO @ T.T ## P1P2 by Q
    return outputs

# Paired-batches for forward only
# Define MIONet architecture with pair-batched input; u and (x,y) have totally individual batched input
@jit
def operator_net_paired(params, uf, ug, xy):
    B1 = branch_model_1.apply(params['branch1'], uf) ## Px by J1
    B2 = branch_model_2.apply(params['branch2'], ug) ## Px by J2
    T = trunk_model.apply(params['trunk'], xy) ## Q by I
    W1 = params['last1'] ## J1 by I (already transposed)
    W2 = params['last2'] ## J2 by I (already transposed)
    BW1 = B1 @ W1 ## Px by I
    BW2 = B2 @ W2 ## Px by I
    BDiag = BW1 * BW2 ## Px by I, Hadamard product, "Diag-entry" element of KR product
    outputs = BDiag @ T.T ## Px by Q
    return outputs
