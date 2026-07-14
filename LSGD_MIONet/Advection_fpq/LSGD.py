# LSGD system formation and solve
import jax
import jax.numpy as jnp
from jax import jit
from networks import *
jax.config.update('jax_enable_x64', True)

# Construct LS system
@jit
def construct_LS_PI(params, uin_f, uin_g, xy_phys, xy_data):
    B1 = branch_model_1.apply(params['branch1'], uin_f) # P1 by J1 ##
    B2 = branch_model_2.apply(params['branch2'], uin_g) # P2 by J2 ##
    Tphys = trunk_only_net_deriv(params['trunk'], xy_phys) # Q by I
    Fphys = jnp.kron(uin_f[:,1:,None].transpose(0,2,1),jnp.ones((1,jnp.shape(uin_f)[1]-1,1))).reshape((jnp.shape(uin_f)[0],-1)) # P1 by Q
    Tdata = trunk_model.apply(params['trunk'], xy_data) # Qb by I ##
    Fdata = uin_g # P2 by Qb
    return B1, B2, Tphys, Fphys, Tdata, Fdata

# Solve LS system for Last layer of Branch 1
@jit
def solve_LS_C1_PI(params, uin_f, uin_g, xy_phys, xy_data, weights):
    B1, B2, Tphys, Fphys, Tdata, Fdata = construct_LS_PI(params, uin_f, uin_g, xy_phys, xy_data)
    W2 = params['last2']
    
    numP1 = jnp.shape(B1)[0]
    numP2 = jnp.shape(B2)[0]
    numQphys = jnp.shape(Tphys)[0]
    numQdata = jnp.shape(Tdata)[0]
    
    lamb0 = weights[1]/weights[0]
    lamb1 = weights[2]/weights[0]
        
    # LS scale is magnified by PQ times from the loss scale
    LSSlamb_data = lamb0 * numQphys / numQdata # lamb BC
    LSSlamb_regul = lamb1 * numP1 * numP2 * numQphys # lamb regul
    
    BW2 = B2 @ W2
    BW2_colsum = jnp.sum(BW2, axis=0, keepdims=True) # 1 by I
    FLTphys = BW2_colsum * (Fphys @ Tphys) # P1 by I, Left vector is broadcasted
    
    FTdata_augmented = BW2.T @ Fdata @ Tdata # I by I 
    FTdata_reduced = FTdata_augmented.diagonal().reshape((1,-1)) # 1 by I
    FTdata = jnp.tile(FTdata_reduced,(numP1,1))
    
    E = B1.T @ (FLTphys+LSSlamb_data*FTdata)
    
    # Normal matrix construction and EigDecomposition
    TTT = (Tphys.T @ Tphys + LSSlamb_data*Tdata.T @ Tdata) * (BW2.T @ BW2)
    BTB = B1.T @ B1

    Q1, d1, Q1T = jnp.linalg.svd(BTB, hermitian=True)
    Q2, d2, Q2T = jnp.linalg.svd(TTT, hermitian=True)

    E_tilde = Q1T @ E @ Q2
    h_coeff = jnp.outer(d1,d2) + LSSlamb_regul*jnp.ones_like(jnp.outer(d1,d2))
    Y = jnp.reciprocal(h_coeff) * E_tilde

    W1 = Q1 @ Y @ Q2T

    return W1

# Solve LS system for Last layer of Branch 2
@jit
def solve_LS_C2_PI(params, uin_f, uin_g, xy_phys, xy_data, weights):
    B1, B2, Tphys, Fphys, Tdata, Fdata = construct_LS_PI(params, uin_f, uin_g, xy_phys, xy_data)
    W1 = params['last1']
    
    numP1 = jnp.shape(B1)[0]
    numP2 = jnp.shape(B2)[0]
    numQphys = jnp.shape(Tphys)[0]
    numQdata = jnp.shape(Tdata)[0]
    
    lamb0 = weights[1]/weights[0]
    lamb1 = weights[2]/weights[0]
        
    # LS scale is magnified by PQ times from the loss scale
    LSSlamb_data = lamb0 * numQphys / numQdata # lamb BC
    LSSlamb_regul = lamb1 * numP1 * numP2 * numQphys # lamb regul
    
    BW1 = B1 @ W1
    BW1_colsum = jnp.sum(BW1, axis=0, keepdims=True) # 1 by I
    FTdata = BW1_colsum * (Fdata @ Tdata) # P2 by I, Left vector is broadcasted
    
    FLTphys_augmented = BW1.T @ Fphys @ Tphys # I by I 
    FLTphys_reduced = FLTphys_augmented.diagonal().reshape((1,-1)) # 1 by I
    FLTphys = jnp.tile(FLTphys_reduced,(numP2,1))

    E = B2.T @ (FLTphys+LSSlamb_data*FTdata)
    
    # Normal matrix construction and EigDecomposition
    TTT = (Tphys.T @ Tphys + LSSlamb_data*Tdata.T @ Tdata) * (BW1.T @ BW1)
    BTB = B2.T @ B2

    Q1, d1, Q1T = jnp.linalg.svd(BTB, hermitian=True)
    Q2, d2, Q2T = jnp.linalg.svd(TTT, hermitian=True)
    
    E_tilde = Q1T @ E @ Q2
    h_coeff = jnp.outer(d1,d2) + LSSlamb_regul*jnp.ones_like(jnp.outer(d1,d2))
    Y = jnp.reciprocal(h_coeff) * E_tilde

    W2 = Q1 @ Y @ Q2T

    return W2