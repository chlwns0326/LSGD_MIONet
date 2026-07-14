# Network and computations
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
from models import branch_model_1, branch_model_2, trunk_model
jax.config.update('jax_enable_x64', True)

# derivative of trunk net w.r.t x,y
@partial(jit, static_argnums=(1,))
def trunk_only_net_deriv(trunk_params, xy):
    ufun = lambda xy0: trunk_model.apply(trunk_params, xy0) ##
    _ux = lambda xy0: jax.jvp(ufun,(xy0,),(jnp.array([1.0,0.0]),))[1] # ux
    _uy = lambda xy0: jax.jvp(ufun,(xy0,),(jnp.array([0.0,1.0]),))[1] # uy
    _ux_x = lambda xy0: jax.jvp(_ux,(xy0,),(jnp.array([1.0,0.0]),))[1] # (ux)x
    _uy_y = lambda xy0: jax.jvp(_uy,(xy0,),(jnp.array([0.0,1.0]),))[1] # (uy)y
    ux_x = vmap(_ux_x)
    uy_y = vmap(_uy_y)
    Tres = -ux_x(xy)-uy_y(xy)
    return Tres

# Define DeepONet architecture with Doubly batched input; u and (x,y) have totally individual batched input
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

# Batched version of deriv of separable DeepONet w.r.t the trunk input variables
@jit
def operator_net_deriv(params, uf, ug, xy):
    B1 = branch_model_1.apply(params['branch1'], uf) ##
    B2 = branch_model_2.apply(params['branch2'], ug) ##
    Tres = trunk_only_net_deriv(params['trunk'], xy)
    W1 = params['last1']
    W2 = params['last2']
    BW1 = B1 @ W1
    BW2 = B2 @ W2
    BO = jnp.vstack([jnp.kron(BW1[:, k], BW2[:, k]) for k in range(BW2.shape[1])]).T
    out_res = BO @ Tres.T
    return out_res

# Define PDE residual; batched version as P1P2 by Q matrix
@jit
def residual_net(params, uf, ug, xy):
    out = operator_net_deriv(params, uf, ug, xy)
    ftilde = jnp.kron(uf[:,1:-1,1:-1,:].transpose(0,2,1,3),jnp.ones((jnp.shape(ug)[0],1,1,1))) # P1P2 by Qx by Qt by 1
    res = ftilde.reshape((jnp.shape(ftilde)[0],-1)) - out # F - Ls;
    return res

# Paired-batches for forward only
# Define DeepONet architecture with pair-batched input; u and (x,y) have totally individual batched input
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

# Batched version of deriv of separable DeepONet w.r.t the trunk input variables
@jit
def operator_net_deriv_paired(params, uf, ug, xy):
    B1 = branch_model_1.apply(params['branch1'], uf) ##
    B2 = branch_model_2.apply(params['branch2'], ug) ##
    Tres = trunk_only_net_deriv(params['trunk'], xy)
    W1 = params['last1']
    W2 = params['last2']
    BW1 = B1 @ W1
    BW2 = B2 @ W2
    BDiag = BW1 * BW2 ##
    out_res = BDiag @ Tres.T
    return out_res

# Define PDE residual; pair-batched version as Px by Q matrix (P1==P2)
@jit
def residual_net_paired(params, uf, ug, xy):
    out = operator_net_deriv_paired(params, uf, ug, xy)
    ftilde = uf[:,1:-1,1:-1,:].transpose(0,2,1,3) ####
    res = ftilde.reshape((jnp.shape(ftilde)[0],-1)) - out # F - Ls;
    return res