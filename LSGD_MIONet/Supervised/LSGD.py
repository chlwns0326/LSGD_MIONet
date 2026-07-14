# LSGD system formation and solve
import jax
import jax.numpy as jnp
from jax import jit
from networks import *
jax.config.update('jax_enable_x64', True)

@jit
def Khatri_Rao(A,B):
    # Input 
    # A: n by k matrix
    # B: m by k matrix
    # output: Khatri-Rao product of A and B; nm by k matrix
    
    res = jnp.vstack([jnp.kron(A[:,k], B[:,k]) for k in range(B.shape[1])]).T
    
    return res

# Construct LS system
@jit
def construct_LS(params, uin_f, uin_g, xy_data):
    B1 = branch_model_1.apply(params['branch1'], uin_f) # P1 by J1 ##
    B2 = branch_model_2.apply(params['branch2'], uin_g) # P2 by J2 ##
    Tdata = trunk_model.apply(params['trunk'], xy_data) # Q by I ##
    # Fdata = uin_g # P2 by Qb
    return B1, B2, Tdata

# Solve LS system for Last layer of Branch 1
@jit
def solve_LS_C1(params, uin_f, uin_g, uout, xy_data, weights):
    B1, B2, Tdata = construct_LS(params, uin_f, uin_g, xy_data)
    W2 = params['last2']
    
    numP1 = jnp.shape(B1)[0]
    numP2 = jnp.shape(B2)[0]
    numQ = jnp.shape(Tdata)[0]
    
    Fdata = jnp.transpose(uout,(0,1,3,2)).reshape(numP1,-1) # P1 by P2Q
    lamb0 = weights[1]/weights[0]
        
    # LS scale is magnified by PQ times from the loss scale
    LSSlamb_regul = lamb0 * numP1 * numP2 * numQ # lamb regul
    
    BW2 = B2 @ W2    
    GT = Khatri_Rao(BW2,Tdata) # P2Q by I
    FTdata = Fdata @ GT # P1 by I
    
    E = B1.T @ (FTdata)
    
    # Normal matrix construction and EigDecomposition
    TTT = (Tdata.T @ Tdata) * (BW2.T @ BW2)
    BTB = B1.T @ B1

    Q1, d1, _ = jnp.linalg.svd(BTB, hermitian=True)
    Q2, d2, _ = jnp.linalg.svd(TTT, hermitian=True)

    E_tilde = Q1.T @ E @ Q2
    h_coeff = jnp.outer(d1,d2) + LSSlamb_regul*jnp.ones_like(jnp.outer(d1,d2))
    Y = jnp.reciprocal(h_coeff) * E_tilde

    W1 = Q1 @ Y @ Q2.T

    return W1

# Solve LS system for Last layer of Branch 2
@jit
def solve_LS_C2(params, uin_f, uin_g, uout, xy_data, weights):
    B1, B2, Tdata = construct_LS(params, uin_f, uin_g, xy_data)
    W1 = params['last1']
    
    numP1 = jnp.shape(B1)[0]
    numP2 = jnp.shape(B2)[0]
    numQ = jnp.shape(Tdata)[0]

    Fdata = jnp.transpose(uout,(1,0,3,2)).reshape(numP2,-1) # P2 by P1Q
    lamb0 = weights[1]/weights[0]
        
    # LS scale is magnified by PQ times from the loss scale
    LSSlamb_regul = lamb0 * numP1 * numP2 * numQ # lamb regul
    
    BW1 = B1 @ W1
    GT = Khatri_Rao(BW1,Tdata) # P1Q by I
    FTdata = Fdata @ GT # P2 by I

    E = B2.T @ (FTdata)
    
    # Normal matrix construction and EigDecomposition
    TTT = (Tdata.T @ Tdata) * (BW1.T @ BW1)
    BTB = B2.T @ B2

    Q1, d1, _ = jnp.linalg.svd(BTB, hermitian=True)
    Q2, d2, _ = jnp.linalg.svd(TTT, hermitian=True)
    
    E_tilde = Q1.T @ E @ Q2
    h_coeff = jnp.outer(d1,d2) + LSSlamb_regul*jnp.ones_like(jnp.outer(d1,d2))
    Y = jnp.reciprocal(h_coeff) * E_tilde

    W2 = Q1 @ Y @ Q2.T

    return W2