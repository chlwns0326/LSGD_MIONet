import os, warnings
warnings.filterwarnings('ignore')
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true' 
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.5' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# imports
import scipy.io as io
import jax, optax
import jax.numpy as jnp
from jax import random
from jax.nn.initializers import glorot_normal, he_normal
jax.config.update('jax_enable_x64', True)

from models import branch_model_1, branch_model_2, trunk_model, model_settings
from train import *
from init_param import *
from networks import *
from loss import *
from LSGD import *
from step import *
from misc import *

# Seed control for reproducibility
MasterKey = 1
seedA, seedB = random.randint(random.key(MasterKey),shape=(2,),minval=0,maxval=2**31)

# Directories (Current directory: /LSGD_MIONet/Poisson_fg)
folder = 'Models/'
keyword = 'Poisson_fg_PI' # 
model_desc = 'L3W150'
datafolder = '../data/'

# Hyperparameters
LS_num = 1 # Adam = 0, LS+Adam > 0
weights_init = [1e-4,1,1e-12]  # weights for physics, data, regularization term, resp. 
N_train = [1000,1000] # f,g 
N_val = 4000 # Large enough
delay_init_pre = 50 # Delayed LS step after the Adam epochs
lr, b1, b2 = 1e-3, 0.99, 0.999 # Adam learning rate/1st moment/2nd moment
phy_lambda = '{:.0e}'.format(weights_init[0])
reg_lambda = '{:.0e}'.format(weights_init[2])
bs, bn = 100, 50

# Adam vs LS+Adam
if LS_num > 0: # LS+Adam
    batch_size = [bs,bs] # predetermined adam batch 
    batch_new = [bn,bn]
    adam_num = 1 # Adam epochs for each WU (LS step performed for each adam_num epochs)
    total_WU = 10000 // adam_num 
    delay_init = delay_init_pre//adam_num
    model_dir = folder + keyword + '_ALSAdam_L' + str(LS_num) +'R' + str(adam_num) + '_' + str(delay_init_pre) + \
        '_HHH_SSS_bat' + str(bs) + 'SQto' + str(bn) + 'SQ_' + \
        model_desc + '_seed_' + str(MasterKey) + '_phy_' + phy_lambda + '_regul_' + reg_lambda 
else: # Adam
    batch_size = [bs,bs] # f,g 
    batch_new = batch_size
    adam_num = 1 # Always epochwise
    weights_init[2] = 0 # No last layer regularization weight
    total_WU = 20000 // adam_num 
    delay_init = delay_init_pre
    model_dir = folder + keyword + '_Adam_NoLS_HHH_SSS_bat'+ str(bs) + 'SQ_' + \
        model_desc + '_seed_' + str(MasterKey) + '_phy_' + phy_lambda
        
createFolder(model_dir +'/models')
createFolder(model_dir +'/losses')

# Hyperparams for models
Nx, Ny = 32, 32 #
xmin, xmax = 0.0,1.0
ymin, ymax = 0.0,1.0
disp_count = int(0.01*total_WU) # tqdm progress display as new lines 

# Initialize
m = (Nx+1)*(Ny+1) # number of trunk input sensors
u_f_in_foo = jnp.zeros((batch_size[0],Nx+1,Ny+1,1)) # 2D input
u_g_in_foo = jnp.zeros((batch_size[1],4*Nx+1)) # 1D input 
xy_in_foo = jnp.zeros((m,2))

# All He Normal init + Optimizer
key = random.key(seedA)
key, *keys = random.split(key,6)
branch_params_1 = branch_model_1.init(keys[0], u_f_in_foo)
branch_params_2 = branch_model_2.init(keys[1], u_g_in_foo)
trunk_params = trunk_model.init(keys[2], xy_in_foo)
last_params_1 = he_normal()(keys[3],(model_settings[0][-4][-1],model_settings[-1][-4][-1]))
last_params_2 = he_normal()(keys[4],(model_settings[1][-4][-1],model_settings[-1][-4][-1]))

key, *keys = random.split(key,4)
branch_params_1 = apply_he(branch_params_1, model_settings, 0, keys[0], gmode='N',scale_b=0)
branch_params_2 = apply_he(branch_params_2, model_settings, 1, keys[1], gmode='N',scale_b=0)
trunk_params = apply_he(trunk_params, model_settings, 2, keys[2], gmode='N',scale_b=0)
params = {'branch1': branch_params_1, 'branch2': branch_params_2, 'trunk': trunk_params, 'last1': last_params_1, 'last2': last_params_2}
optimizer = optax.multi_transform({'adam': optax.inject_hyperparams(optax.adam)(lr,b1=b1,b2=b2), 'zero': optax.set_to_zero()},
            {'branch1':'adam', 'branch2':'adam', 'trunk':'adam', 'last1':'adam', 'last2':'adam'}) 
opt_state = optimizer.init(params)

# Loggers
loss_WU = []
weight_WU = []
loss_WU_val = []
loss_logs = {'loss_WU':loss_WU,'weight_WU':weight_WU,'loss_WU_val':loss_WU_val}

# data load & generation
dataname = 'Poisson_fg_scale_data'
data_dir = datafolder + dataname
data = io.loadmat(data_dir)
j = 1 # jump/stride
uin_f_train = jnp.asarray(data['input_f_train'].astype('float64'))[:N_train[0],:,:,None][:,::j,:,:][:,:,::j,:]
uin_f_val = jnp.asarray(data['input_f_val'].astype('float64'))[:N_val,:,:,None][:,::j,:,:][:,:,::j,:]
uin_g_train = jnp.asarray(data['input_g_train'].astype('float64'))[:N_train[1],:][:,::j]
uin_g_val = jnp.asarray(data['input_g_val'].astype('float64'))[:N_val,::j]
uout_train = jnp.asarray(data['output_train'].astype('float64'))[:N_train[0],:,:][:,::j,:][:,:,::j]
uout_val = jnp.asarray(data['output_val'].astype('float64'))[:N_val,::j,:][:,:,::j]

# Output sensors
x_pre = jnp.linspace(xmin,xmax,Nx+1)
y_pre = jnp.linspace(ymin,ymax,Ny+1)
x_in,y_in = jnp.meshgrid(x_pre,y_pre,indexing='xy')
xy_in = jnp.stack((x_in,y_in),axis=-1)
xy_full = jnp.reshape(xy_in,(-1,2))
xy_phys = jnp.reshape(xy_in[1:-1,1:-1,:],(-1,2))
xy_data = jnp.concatenate((xy_in[0,0:-1,:],xy_in[0:-1,-1,:],xy_in[-1,-1:-Nx-1:-1,:],xy_in[-1:-Ny-1:-1,0,:]),axis=0) # (50+50+50+51 by 2)

train(params=params, optimizer=optimizer, seed=seedB, delay_init=delay_init,
      uin_f_train=uin_f_train, uin_g_train=uin_g_train, uout_train=uout_train,
      uin_f_val=uin_f_val, uin_g_val=uin_g_val, uout_val=uout_val,
      xy=xy_full, xy_phys=xy_phys, xy_data=xy_data,
      weights_init=weights_init,
      batch_size=batch_size, batch_new=batch_new, adam_num=adam_num, LS_num=LS_num,
      loss_logs=loss_logs, model_dir=model_dir,
      nIter=total_WU, disp_count=disp_count)