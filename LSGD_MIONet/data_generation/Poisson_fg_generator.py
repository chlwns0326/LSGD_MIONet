import os, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '0' #################################

# imports
import scipy.io as io
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
jax.config.update('jax_enable_x64', True)

def RBF_exp_quad_2D(xl, yl, var, lx, ly):
    diff1 = jnp.abs(xl[None,:]-xl[:,None])
    diff2 = jnp.abs(yl[None,:]-yl[:,None])
    return var * jnp.exp(-diff1**2/(2*lx**2)-diff2**2/(2*ly**2))

def gen_f_2D(key, x, y, var, lx, ly):
    # Generate subkeys
    subkeys = random.split(key, 2)
    Nx, Ny = jnp.shape(x)[0]-1, jnp.shape(y)[0]-1
    xx, yy = jnp.meshgrid(x,y,indexing='xy')
    xx_long = xx.transpose((1,0)).flatten()
    yy_long = yy.transpose((1,0)).flatten()
    # Generate a GP sample
    jitter = 1e-12
    K = RBF_exp_quad_2D(xx_long, yy_long, var, lx, ly)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye((Nx+1)*(Ny+1)))
    pre_out = jnp.dot(L, random.normal(subkeys[0], ((Nx+1)*(Ny+1),)))
    out = pre_out.reshape((Ny+1,Nx+1)).transpose((1,0))   
    return out

# periodic kernel
def RBF_periodic(x1, x2, var, l, p):
    # Original source from https://zenodo.org/records/5206676
    diff = jnp.abs(x1[None,:]-x2[:,None])
    return var * jnp.exp(-(2/l**2)*jnp.sin(jnp.pi*diff/p)**2)

def gen_g(key, Nx, var, length_scale, period):
    # Original source from https://zenodo.org/records/5206676
    subkeys = random.split(key, 2)
    # Generate a GP sample
    xmin, xmax = 0, period
    jitter = 1e-12
    X = jnp.linspace(xmin, xmax, Nx+1)
    K = RBF_periodic(X, X, var, length_scale, period)
    L = jnp.linalg.cholesky(K + jitter*jnp.eye(Nx+1))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (Nx+1,)))
    # Create a callable interpolation function  
    g = lambda x: jnp.interp(x, X.flatten(), gp_sample)    
    return g

# FDM
def solve_Poisson(BC,f):
    """Solve 1D
    -div(grad(u)) = f
    with given Dirichlet BCs on unit square.
    """
    # Create grid
    Nx, Ny = np.shape(f)[0]-1, np.shape(f)[1]-1
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    x = np.linspace(xmin, xmax, Nx+1)
    y = np.linspace(ymin, ymax, Ny+1)
    dx = x[1]-x[0]
    dy = y[1]-y[0]
            
    # Initialize solution and apply initial condition
    u = np.zeros((Nx+1, Ny+1))

    u[:-1,0] = (BC[0:Nx])
    u[-1,:-1] = (BC[Nx:Nx+Ny])
    u[Nx:0:-1,-1] = (BC[Nx+Ny:2*Nx+Ny])
    u[0,Ny:0:-1] = (BC[2*Nx+Ny:-1])
    
    matL = np.zeros(((Nx+1)*(Ny+1),(Nx+1)*(Ny+1)))
    matL_inds = np.zeros(((Nx-1)*(Ny-1),),dtype=np.int64)
    rhs_BC = np.zeros((Nx+1,Ny+1))

    for i in np.arange(start=1,stop=Nx):
        for j in np.arange(start=1,stop=Ny):
            bctemp = 0
            indc = (Nx+1)*j+i
            ind_matL = (Nx-1)*(j-1)+(i-1)
            indl,indr,indd,indu = indc-1,indc+1,indc-(Nx+1),indc+(Nx+1)

            matL[indc,indc] = (4*Nx*Ny)
            matL[indc,indl] = (-Nx*Ny)
            matL[indc,indr] = (-Nx*Ny)
            matL[indc,indd] = (-Nx*Ny)
            matL[indc,indu] = (-Nx*Ny)
            matL_inds[ind_matL] = indc
            if i == 1:
                bctemp = bctemp + Nx*Ny*u[0,j]
            if i == Nx-1:
                bctemp = bctemp + Nx*Ny*u[-1,j]
            if j == 1:
                bctemp = bctemp + Nx*Ny*u[i,0]
            if j == Ny-1:
                bctemp = bctemp + Nx*Ny*u[i,-1]
            rhs_BC[i,j] = bctemp

            
    vecb = f[1:-1,1:-1].transpose((1,0)).reshape(((Nx-1)*(Ny-1),1))
    vecb_BC = rhs_BC[1:-1,1:-1].transpose((1,0)).reshape(((Nx-1)*(Ny-1),1))

    flatsol = jnp.linalg.solve(matL[matL_inds,:][:,matL_inds],vecb+vecb_BC)
    
    # u = u.at[1:-1,1:-1].set(flatsol.reshape((Nx-1,Ny-1)).transpose((1,0)))
    u[1:-1,1:-1] = (flatsol.reshape((Nx-1,Ny-1)).transpose((1,0)))
    
    return u

def main():
    # Directories and hyperparams
    data_dir = 'data/'
    data_name = 'Poisson_fg_scale_data.mat' 
    
    N_train, N_test = 2000, 5000
    Nx, Ny = 128, 128
    jx, jy = Nx//32, Ny//32
    var_f = 1 # 1
    lfx, lfy = 0.2, 0.2 
    scale_f = 0.1
    var_g = 1
    lg = 0.3
    scale_g = 0.1 # 1
    xmin, xmax = 0, 1
    xbmin, xbmax = 0, 4
    period = xbmax - xbmin
    x = np.linspace(xmin, xmax, Nx+1)
    y = np.linspace(xmin, xmax, Ny+1)
    x_bc = np.linspace(xbmin, xbmax, 4*Nx+1)

    # train generation
    key = random.key(0)
    keys = random.split(key,N_train)
    input_f_train = scale_f * jax.vmap(lambda k: gen_f_2D(k, x, y, var_f, lfx, lfy))(keys)
    input_g_train = scale_g * jax.vmap(lambda k: gen_g(k, 4*Nx, var_g, lg, period)(x_bc))(keys)
    output_train = np.zeros((N_train,Nx+1,Ny+1))
    for ind in tqdm(range(N_train)):
        output_train[ind,:,:] = solve_Poisson(input_g_train[ind,:],input_f_train[ind,:,:])

    # val generation
    key = random.key(113355)
    keys = random.split(key,N_test)
    input_f_val = scale_f * jax.vmap(lambda k: gen_f_2D(k, x, y, var_f, lfx, lfy))(keys)
    input_g_val = scale_g * jax.vmap(lambda k: gen_g(k, 4*Nx, var_g, lg, period)(x_bc))(keys)
    output_val = np.zeros((N_test,Nx+1,Ny+1))
    for ind in tqdm(range(N_test)):
        output_val[ind,:,:] = solve_Poisson(input_g_val[ind,:],input_f_val[ind,:,:])
     
    mdic = {"input_f_train": input_f_train[:,::jx,::jy],"input_g_train": input_g_train[:,::jx], "output_train": output_train[:,::jx,::jy],
            "input_f_val": input_f_val[:,::jx,::jy],"input_g_val": input_g_val[:,::jx], "output_val": output_val[:,::jx,::jy]}
    io.savemat(data_dir+data_name,mdic)
    
if __name__ == '__main__':    
    main()