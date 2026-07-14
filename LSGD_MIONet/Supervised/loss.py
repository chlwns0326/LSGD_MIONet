# Losses
import jax
import jax.numpy as jnp
from jax import jit
from networks import *
jax.config.update('jax_enable_x64', True)

# Define data loss
@jit
def loss_data(params, uin_f, uin_g, uout, xy_data):
    u_pred = operator_net(params, uin_f, uin_g, xy_data)
    u_out_ = jnp.transpose(uout,(0,2,1)).reshape((uout.shape[0], -1))
    loss = jnp.mean((u_out_-u_pred)**2) # denominator : P1P2(#u) * Q(#xy) 
    return loss     

# Define last layer regularization loss
@jit
def loss_reguls(params):
    loss = jnp.sum(params['last1']**2) + jnp.sum(params['last2']**2)
    return loss

# Define total loss
@jit
def loss(params, uin_f, uin_g, uout, xy_data, weights):
    loss_data_value = loss_data(params, uin_f, uin_g, uout, xy_data)
    loss_reguls_value = loss_reguls(params)
    loss = weights[0]*loss_data_value + weights[1]*loss_reguls_value
    return loss

# Define l2 loss # Forward only, diag, P1 == P2
@jit
def loss_l2_diag(params, uin_f, uin_g, uout, xy_data):
    u_pred_diag = operator_net_paired(params, uin_f, uin_g, xy_data)
    u_out_ = jnp.transpose(uout,(0,2,1)).reshape((uout.shape[0], -1))
    axis = tuple(range(1,u_out_.ndim))        
    diff_sq = jnp.mean((u_out_ - u_pred_diag)**2,axis=axis)
    data_sq = jnp.mean((u_out_)**2,axis=axis)
    l2relsqs = diff_sq/data_sq
    l2err = jnp.mean(jnp.sqrt(diff_sq)) 
    l2rel = jnp.mean(jnp.sqrt(l2relsqs))
    return l2err, l2rel

# Define data loss
@jit
def loss_data_diag(params, uin_f, uin_g, uout, xy_data):
    u_pred = operator_net_paired(params, uin_f, uin_g, xy_data)
    u_out_ = jnp.transpose(uout,(0,2,1)).reshape((uout.shape[0], -1))
    loss = jnp.mean((u_out_-u_pred)**2) # denominator : Px(#u) * Q(#xy) 
    return loss     

# return each components
@jit
def loss_comps(params, uin_f, uin_g, uout, xy_data, weights):
    loss_data_value = loss_data_diag(params, uin_f, uin_g, uout, xy_data)
    loss_reguls_value = loss_reguls(params)
    loss = weights[0]*loss_data_value + weights[1]*loss_reguls_value
    l2err, l2rel = loss_l2_diag(params, uin_f, uin_g, uout, xy_data)
    return jnp.array([loss, weights[0]*loss_data_value, weights[1]*loss_reguls_value, l2rel])