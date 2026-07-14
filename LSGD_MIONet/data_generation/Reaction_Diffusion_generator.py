import os, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #################################

# imports
import scipy.io as io
import pickle
import jax
import jax.numpy as jnp
from jax import lax, random
jax.config.update('jax_enable_x64', True)
from tqdm import tqdm

def RBF(x1, x2, var, l):
    diff = jnp.abs(x1[None,:]-x2[:,None])
    return var * jnp.exp(-diff**2/(2*l**2))

# Original source from https://zenodo.org/records/5206676
def gen_f(key, NX, l, xmin=0, xmax=1):
    subkeys = random.split(key, 2)
    # Generate a GP sample
    jitter = 1e-12
    X = jnp.linspace(xmin, xmax, NX+1)
    K = RBF(X, X, 1, l)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(NX+1))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (NX+1,)))
    # Create a callable interpolation function  
    f = lambda x: jnp.interp(x, X.flatten(), gp_sample)    
    return f

# Advection-diffusion-reaction problem FDM(Crank-Nicolson) solver 
def solve_ADR(Nx, Nt, D, v, R, dR, f, u0):
    """
    Solve 1D Advection-Diffusion-Reaction (conservative form) on the unit interval 
    u_t = div (D(x) grad_u - v(x) u) + R(u) + f(x)
        = (D u_x - vu)_x + R(u) + f
        = D_x u_x + D u_xx - v_x u - v u_x + R(u) + f
    with initial u0 and zero Dirichlet boundary conditions 
    using the Crank-Nicolson scheme and 1st Taylor approx of R.
    
    Un: u_next, Uc: u_current.
    The Crank-Nicolson scheme is
    (Un+Uc)/dt = 1/2 (F(Un) + F(Uc)),
    where F(u) = D_x u_x + D u_xx - v_x u - v u_x + R(u) + f 
    and the 1st Taylor approximation is given as R(Un) = R(Uc) + R'(Uc) (Un-Uc).
    So, 
    1/dt * Un - 1/2 * (D_x Un_x + D Un_xx - v_x Un - v Un_x) - 1/2 * R'(Uc)Un
    = 1/dt * Uc + 1/2 * (D_x Uc_x + D Uc_xx - v_x Uc - v Uc_x) - 1/2 * R'(Uc)Uc + R(Uc) + f.
    This reads to the system of variable Un:
    A1 Un = A2 Uc + b2, 
    where A1, A2, Uc, b2 are given.
    """
    
    # Create grid
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    x = jnp.linspace(xmin, xmax, Nx+1)
    t = jnp.linspace(tmin, tmax, Nt+1)
    dx = x[1]-x[0]
    dt = t[1]-t[0]

    # Compute time independent values
    D_ = D(x)
    v_ = v(x)
    f_ = f(x)
    
    # Compute finite difference operators
    Diff0 = jnp.eye(Nx+1)
    Diff1 = 1/(2*dx) * (jnp.eye(Nx+1,k=1) - jnp.eye(Nx+1,k=-1))
    Diff2 = 1/dx**2 * (jnp.eye(Nx+1,k=1) + jnp.eye(Nx+1,k=-1) - 2*jnp.eye(Nx+1))
    
    D_term = (jnp.diag(Diff1@D_) @ Diff1 + jnp.diag(D_) @ Diff2)[1:-1,1:-1]
    v_term = -(jnp.diag(Diff1@v_) @ Diff0 + jnp.diag(v_) @ Diff1)[1:-1,1:-1]
    Dv_term = D_term + v_term

    # Initialize solution and apply initial condition
    u = jnp.zeros((Nx+1, Nt+1))
    u = u.at[:,0].set(u0(x))
    
    # Timestep update (Crank-Nicolson)
    def next_timestep(i,u):
        Ri = R(u[1:-1,i])
        dRi = dR(u[1:-1,i])
        R_Taylor_correction = 1/2 * jnp.diag(dRi)
        A_left = 1/dt * Diff0[1:-1,1:-1] - 1/2 * Dv_term - R_Taylor_correction
        A_right = 1/dt * Diff0[1:-1,1:-1] + 1/2 * Dv_term - R_Taylor_correction
        b1 = A_right @ u[1:-1,i].T
        b2 = f_[1:-1] + Ri
        u = u.at[1:-1,i+1].set(jnp.linalg.solve(A_left,b1+b2))
        return u
    
    # Run loop
    UU = lax.fori_loop(0, Nt, next_timestep, u)
    return UU

# Directory and hyperparams
data_dir = '../data/'
suffix = '_huge'
mode = 'fD' # 'f' 'v' 'R' 

N_train_f, N_train_D = 1000, 1000 # 1000, 1000
N_test = 4000 # 4000
Nx, Nt = 128, 256
jx, jt = Nx//32, Nt//32
pi = jnp.pi
D = lambda x: 0.01*jnp.ones_like(x)
R = lambda x: 0.01*x**2
dR = lambda x: 0.02*x
u0 = lambda x: jnp.zeros_like(x)
v = lambda x: 0.0*jnp.sin(pi*x)
xmin, xmax = 0, 1
x = jnp.linspace(0, 1, Nx+1)

# train/val generation
key = random.key(0)
key2 = random.key(987654)
keys_f = random.split(key,N_train_f)
keys_D = random.split(key2,N_train_D)

rkey = random.key(113355)
rkey2 = random.key(224466)
rkeys_f = random.split(rkey,N_test)
rkeys_D = random.split(rkey2,N_test)
output_train = jnp.zeros((N_train_f,N_train_D,32+1,32+1))

# f input: scale = 1, l = 0.2, 
# D input: scale = 0.25, l = 0.2,
l_f, scale_f = 0.2, 1
l_D, scale_D = 0.2, 0.35 
input_train_f = scale_f*jax.vmap(lambda key: gen_f(key, Nx, l_f)(x))(keys_f) 
input_train_D_pre = scale_D*jax.vmap(lambda key: gen_f(key, Nx, l_D)(x))(keys_D) 
input_train_D = 0.01 * jnp.exp(input_train_D_pre)

for ind in tqdm(range(N_train_D)):
    D = lambda xq: jnp.interp(xq, x, input_train_D[ind,:]) 
    output_train_row = jax.vmap(lambda fvec: solve_ADR(Nx, Nt, D, v, R, dR, lambda xq: jnp.interp(xq, x, fvec), u0))(input_train_f)
    output_train = output_train.at[:,ind,:,:].set(output_train_row[:,::jx,::jt])

input_val_f = scale_f*jax.vmap(lambda key: gen_f(key, Nx, l_f)(x))(rkeys_f)
input_val_D_pre = scale_D*jax.vmap(lambda key: gen_f(key, Nx, l_D)(x))(rkeys_D) 
input_val_D = 0.01 * jnp.exp(input_val_D_pre)

output_val = jax.vmap(lambda fvec, Dvec: 
    solve_ADR(Nx, Nt, lambda xq: jnp.interp(xq, x, Dvec), v, R, dR, 
              lambda xq: jnp.interp(xq, x, fvec), u0))(input_val_f,input_val_D)
    
data_name = 'ADR_' + mode + '_data' + suffix 
mdic = {"input_train_f": input_train_f[:,::jx], "input_train_D": input_train_D[:,::jx], 
        "output_train": output_train, # 4D large array 
        "input_val_f": input_val_f[:,::jx], "input_val_D": input_val_D[:,::jx], 
        "output_val": output_val[:,::jx,::jt]}
# io.savemat(data_dir+data_name+'.mat',mdic)

# when file is large so that need to be saved as pkl  
with open(data_dir+data_name+'.pkl', 'wb') as file:
    pickle.dump(mdic, file)

