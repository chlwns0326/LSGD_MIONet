import os, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #################################

# imports
import scipy.io as io
import jax
import jax.numpy as jnp
from jax import random
jax.config.update('jax_enable_x64', True)

def RBF(x1, x2, var, l):
    # Original source from https://zenodo.org/records/5206676
    diff = jnp.abs(x1[None,:]-x2[:,None])
    return var * jnp.exp(-diff**2/(2*l**2))

def gen_pq(key, NX, a, length_scale):
    # Original source from https://zenodo.org/records/5206676
    subkeys = random.split(key, 2)
    # Generate a GP sample
    xmin, xmax = 0, 1
    jitter = 1e-12
    X = jnp.linspace(xmin, xmax, NX+1)
    AX = jnp.linspace(-a*xmax, xmax, int(a*NX+NX+1))
    K = RBF(AX, AX, 1, length_scale)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(int(a*NX+NX+1)))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (int(a*NX+NX+1),)))
    # Create a callable interpolation function  
    pq = lambda x: jnp.interp(x, AX.flatten(), gp_sample)    
    return pq

def gen_src(key, NX, length_scale):
    # Original source from https://zenodo.org/records/5206676
    subkeys = random.split(key, 2)
    # Generate a GP sample
    xmin, xmax = 0, 1
    jitter = 1e-12
    X = jnp.linspace(xmin, xmax, NX+1)
    K = RBF(X, X, 1, length_scale)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(NX+1))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (NX+1,)))
    f = lambda x: jnp.interp(x, X.flatten(), gp_sample)
    f0 = lambda x: f(x) - f(jnp.zeros_like(x))    
    return f0

# Constant coefficient Advection solver
def solve_advec_const_src(Nx, Nt, a, pq, f):
    """Solve 1D
    u_t + au_x = f
    a(x)>0
    with given initial(t=0) and boundary(x=0) conditions.
    """
    # Create grid
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    x = jnp.linspace(xmin, xmax, Nx+1)
    t = jnp.linspace(tmin, tmax, Nt+1)
    xx,tt = jnp.meshgrid(x,t,indexing='ij')
    # Antiderivative as cumsum
    f_prev = jnp.concat((-f[0:1],f[:-1]),axis=0)
    Fx = (xmax-xmin)/(2*Nx) * jnp.cumsum(f_prev+f)    
    # Exact solution generation
    pq_ = lambda x: jnp.interp(x, jnp.linspace(-a*xmax, xmax, jnp.shape(pq)[0]), pq)
    # F_ = lambda x: jnp.interp(x, jnp.linspace(xmin, xmax, jnp.shape(Fx)[0]), Fx) if x > 0 else Fx[0]
    F_ = lambda x: jnp.where(x > 0, jnp.interp(x, jnp.linspace(xmin, xmax, jnp.shape(Fx)[0]), Fx), Fx[0])
    UU = pq_(xx-a*tt)
    F = 1/a * (F_(xx) - F_(xx-a*tt))
    return F + UU

def elongate(a, vec):
    xmin, xmax = 0, 1
    ax = jnp.linspace(-a*xmax, 0, jnp.shape(vec)[0])
    ax2 = jnp.linspace(-a*xmax, 0, 2*jnp.shape(vec)[0]-1)
    U = jnp.interp(ax2, ax, vec)
    return U

# Directories and hyperparams
data_dir = 'data/'
data_name = 'Advection_f_PQ_data.mat' 

N_train, N_test = 2000, 5000
lf = 0.2
lg = 0.2
Nx, Nt = 128, 128
jx, jt = Nx//32, Nt//32
a = 0.5
q = lambda t: jnp.zeros_like(t)
xmin, xmax = 0, 1
x = jnp.linspace(xmin, xmax, Nx+1)
ax = jnp.linspace(-a*xmax, xmax, int(a*Nx+Nx+1))

# train generation
key = random.key(0)
key1, key2 = random.split(key,2)
keys_f = random.split(key,N_train)
keys_pq = random.split(key2,N_train)
pre_inputs_f = jax.vmap(lambda k: gen_src(k, Nx, lg)(x))(keys_f)
pre_inputs_pq = jax.vmap(lambda k: gen_pq(k, Nx, a, lf)(ax))(keys_pq)
output_train = jax.vmap(lambda F, PQ: solve_advec_const_src(Nx, Nt, a, PQ, F),in_axes=(0,0))(pre_inputs_f,pre_inputs_pq)
Qpart = jax.vmap(lambda pq: elongate(a,pq))(pre_inputs_pq[:,:int(a*Nx+1)]) # vmap a*Nx+1 -> Nx+1
Ppart = pre_inputs_pq[:,int(a*Nx+1):] 
input_f_train = pre_inputs_f
input_g_train = jnp.concatenate((Qpart,Ppart),axis=1)

# val generation
key = random.key(113355)
key1, key2 = random.split(key,2)
keys_f = random.split(key,N_test)
keys_pq = random.split(key2,N_test)
pre_inputs_f = jax.vmap(lambda k: gen_src(k, Nx, lg)(x))(keys_f)
pre_inputs_pq = jax.vmap(lambda k: gen_pq(k, Nx, a, lf)(ax))(keys_pq)
output_val = jax.vmap(lambda F, PQ: solve_advec_const_src(Nx, Nt, a, PQ, F),in_axes=(0,0))(pre_inputs_f,pre_inputs_pq)
Qpart = jax.vmap(lambda pq: elongate(a,pq))(pre_inputs_pq[:,:int(a*Nx+1)]) # vmap a*Nx+1 -> Nx+1
Ppart = pre_inputs_pq[:,int(a*Nx+1):] 
input_f_val = pre_inputs_f
input_g_val = jnp.concatenate((Qpart,Ppart),axis=1)

mdic = {"input_f_train": input_f_train[:,::jx],"input_g_train": input_g_train[:,::jx], "output_train": output_train[:,::jx,::jt],
            "input_f_val": input_f_val[:,::jx],"input_g_val": input_g_val[:,::jx], "output_val": output_val[:,::jx,::jt]}
io.savemat(data_dir+data_name,mdic)