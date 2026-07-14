# optimizer step
import jax, optax
from jax import grad, jit
from functools import partial
from loss import *
from LSGD import *
jax.config.update('jax_enable_x64', True)

# Define a compiled update step for GD
@partial(jit, static_argnums=(1,))
def step_GD(params, optimizer, opt_state, uin_f, uin_g, xy_phys, xy_data, weights):
    grads = grad(loss)(params, uin_f, uin_g, xy_phys, xy_data, weights)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# Define a compiled update step for LS # Full batch
def step_LS_alt_order(params, uin_f, uin_g, xy_res, xy_ibc, weights, order=0):
    digit = order % 2
    if digit == 0:
        param_last1 = solve_LS_C1_PI(params, uin_f, uin_g, xy_res, xy_ibc, weights)
        params['last1'] = param_last1
        param_last2 = solve_LS_C2_PI(params, uin_f, uin_g, xy_res, xy_ibc, weights)
        params['last2'] = param_last2
        return params
    
    else:
        param_last2 = solve_LS_C2_PI(params, uin_f, uin_g, xy_res, xy_ibc, weights)
        params['last2'] = param_last2
        param_last1 = solve_LS_C1_PI(params, uin_f, uin_g, xy_res, xy_ibc, weights)
        params['last1'] = param_last1
        return params