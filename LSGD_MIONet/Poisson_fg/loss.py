# Losses
import jax
import jax.numpy as jnp
from jax import jit
from networks import *
jax.config.update('jax_enable_x64', True)

# Define physics loss
@jit
def loss_phys(params, uin_f, uin_g, xy_phys):
    pred = residual_net(params, uin_f, uin_g, xy_phys)
    loss = jnp.mean((pred)**2) # denominator : P1P2(#u) * Q(#xy)
    return loss

# Define data loss
@jit
def loss_data(params, uin_f, uin_g, xy_data):
    ibc_pred = operator_net(params, uin_f, uin_g, xy_data)
    gtilde = jnp.kron(jnp.ones((jnp.shape(uin_f)[0],1)),uin_g[:,:-1]) ####
    loss = jnp.mean((gtilde-ibc_pred)**2) # denominator : P1P2(#u) * Qb(#xy) 
    return loss     

# Define last layer regularization loss
@jit
def loss_reguls(params):
    loss = jnp.sum(params['last1']**2) + jnp.sum(params['last2']**2)
    return loss

# Define total loss
@jit
def loss(params, uin_f, uin_g, xy_phys, xy_data, weights):
    loss_phys_value = loss_phys(params, uin_f, uin_g, xy_phys)
    loss_data_value = loss_data(params, uin_f, uin_g, xy_data)
    loss_reguls_value = loss_reguls(params)
    loss =  weights[0]*loss_phys_value + weights[1]*loss_data_value + weights[2]*loss_reguls_value
    return loss

# Define l2 loss # Forward only, diag, P1 == P2
@jit
def loss_l2_diag(params, uin_f, uin_g, u_out, xy):
    u_pred_diag = operator_net_paired(params, uin_f, uin_g, xy)
    u_out_ = jnp.transpose(u_out,(0,2,1)).reshape((u_out.shape[0], -1))
    axis = tuple(range(1,u_out_.ndim))        
    diff_sq = jnp.mean((u_out_ - u_pred_diag)**2,axis=axis)
    data_sq = jnp.mean((u_out_)**2,axis=axis)
    l2relsqs = diff_sq/data_sq
    l2err = jnp.mean(jnp.sqrt(diff_sq)) 
    l2rel = jnp.mean(jnp.sqrt(l2relsqs))
    return l2err, l2rel

# Define physics loss
@jit
def loss_phys_diag(params, uin_f, uin_g, xy_phys):
    pred = residual_net_paired(params, uin_f, uin_g, xy_phys)
    loss = jnp.mean((pred)**2) # denominator : Px(#u) * Q(#xy)
    return loss

# Define data loss
@jit
def loss_data_diag(params, uin_f, uin_g, xy_data):
    ibc_pred = operator_net_paired(params, uin_f, uin_g, xy_data)
    gtilde = uin_g[:,:-1] ####
    loss = jnp.mean((gtilde-ibc_pred)**2) # denominator : Px(#u) * Qb(#xy) 
    return loss     

# return each components
@jit
def loss_comps(params, uin_f, uin_g, u_out, xy, xy_phys, xy_data, weights):
    loss_phys_value = loss_phys_diag(params, uin_f, uin_g, xy_phys)
    loss_data_value = loss_data_diag(params, uin_f, uin_g, xy_data)
    loss_reguls_value = loss_reguls(params)
    loss = weights[0]*loss_phys_value + weights[1]*loss_data_value + weights[2]*loss_reguls_value
    l2err, l2rel = loss_l2_diag(params, uin_f, uin_g, u_out, xy)
    return jnp.array([loss, weights[0]*loss_phys_value, weights[1]*loss_data_value, weights[2]*loss_reguls_value, l2err, l2rel])