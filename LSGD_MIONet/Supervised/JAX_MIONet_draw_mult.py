import warnings
warnings.filterwarnings('ignore')

import scipy.io as io
import jax
import jax.numpy as jnp
from jax import random, jit
from jax.nn.initializers import glorot_normal, he_normal
jax.config.update('jax_enable_x64', True)
import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jax.nn import silu
from models import MLP, CNN_MLP 
from init_param import *
from misc import *

def draw_sample(xy_in,uin_f,uin_g,uout,uout_pred,inds,keyword,savedir):
    # Sample data draw
    x_in, y_in = xy_in[:,:,0], xy_in[:,:,1]
    px = 1/plt.rcParams['figure.dpi']  # pixel in inches
    d2ind = jnp.shape(uin_f)[0]
    for ind in inds:
        uout_ind,uout_pred_ind = uout[ind,:,:],uout_pred[d2ind*ind+ind,:,:]
        fig = plt.figure(figsize=(1920*px,1080*px))
        uin_f_ind,uin_g_ind = uin_f[ind,:], uin_g[ind,:]
        
        ax = fig.add_subplot(2,3,1)
        im = ax.plot(jnp.linspace(0,1,jnp.shape(uin_f_ind)[0]), uin_f_ind)
        ax.set_title(f'Input function F')
        
        ax = fig.add_subplot(2,3,2)
        im = ax.plot(jnp.linspace(0,1,jnp.shape(uin_g_ind)[0]), uin_g_ind)
        ax.set_title(f'Input function G')
        
        ax = fig.add_subplot(2,3,4,projection='3d')
        im = ax.plot_surface(x_in, y_in, uout_ind.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title(f'Label')
        plt.colorbar(im,ax=ax)
        
        ax = fig.add_subplot(2,3,5,projection='3d')
        im = ax.plot_surface(x_in, y_in, uout_pred_ind.T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title(f'Output function prediction')
        plt.colorbar(im,ax=ax)
        
        ax = fig.add_subplot(2,3,6,projection='3d')
        im = ax.plot_surface(x_in, y_in, (uout_ind-uout_pred_ind).T, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title(f'Error')
        plt.colorbar(im,ax=ax)

        fig.suptitle('Data ' + str(ind+1))
        plt.savefig(savedir+'/Data_'+str(ind+1)+'.png')
        plt.close()
        
def data_result_to_npy(uin_f, uin_g, uout, xy_in, Q_size, model_dir, in_ftn, save_path, model_path='/models/model_save_besttrain.pickle', suffix=''):
    # Train/Test data save to npy and plot some of them
    params = model_load(path=model_dir+model_path)
    xy_fold = jnp.reshape(xy_in,(-1,2))

    # network output data
    uout_pred_pre = operator_net(params, uin_f, uin_g, xy_fold) # P by Q
    uout_pred = uout_pred_pre.reshape((-1,Q_size[0],Q_size[1])).swapaxes(1,2) # P by Qx by Qy
    d2ind = jnp.shape(uin_f)[0]
    uout_pred_diag = uout_pred[::d2ind+1,:,:]
    jnp.save(model_dir + save_path + '/u_out_diag' + suffix + '.npy',uout_pred_diag)
    jnp.save(model_dir + save_path + '/last_param_1.npy',params['last1'])
    jnp.save(model_dir + save_path + '/last_param_2.npy',params['last2'])
    
    # sample pics
    inds = range(4)
    draw_sample(xy_in,uin_f,uin_g,uout,uout_pred,inds,in_ftn,model_dir+save_path)
    
    # output l2 errors
    N = jnp.shape(uout)[0]
    stat = jnp.zeros((N,2))
    for ind in range(N):
        l2err, l2rel = loss_l2_diag(params,uin_f[ind:ind+1,:], uin_g[ind:ind+1,:], uout[ind:ind+1,:,:], xy_fold)
        stat = stat.at[ind,:].set([l2err,l2rel])
    jnp.save(model_dir + save_path + '/u_out_stat' + suffix + '.npy',stat)

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

# Define MIONet architecture with Doubly batched input; u and (x,y) have totally individual batched input
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

# Paired-batches for forward only
# Define MIONet architecture with pair-batched input; u and (x,y) have totally individual batched input
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

# Directories (Current directory: /LSGD_MIONet/Supervised)
folder = 'Models/'
keywords = ['DiffReact']
model_descs = ['L3W150']
lambdas = [1e-8]
etimes = [10000] # tighter
datafolder = '../data/'
ALS_nums = [0,1] # Adam = 0, ALS+Adam > 0
seeds = ['1','2','3']

# Common hyperparams
N_train, N_test = 100, 100  # Number of training/validation function dataset to use
delay_init_pre = 50 # Delayed LS step after the Adam epochs
Nx, Ny = 32, 32 # Uniform grid of size (32+1)*(32+1)
xmin, xmax = 0.0, 1.0 
ymin, ymax = 0.0, 1.0
Q_size = [Nx+1, Ny+1]

saveech =  [10,20,30,50, *range(100,201,20), *range(250,501,50), *range(600,1001,100), 
                *range(1200,2001,200), *range(2500,5001,500), *range(6000,10001,1000), 
                *range(12000,20001,2000), *range(25000,50001,5000), *range(60000,100001,10000), 
                *range(120000,200001,20000), *range(250000,500001,50000), *range(600000,1000001,100000)]    

for keyword,model_desc,lamb,etime in zip(keywords,model_descs,lambdas,etimes):
    # Hyperparameters
    weights_init = [1,lamb] # weights for data, regularization term, resp. 
  
    # Initialize & Data load
    if keyword == 'DiffReact':
        u_f_in_foo = jnp.zeros((1,Nx+1)) # 1D input
        u_g_in_foo = jnp.zeros((1,Nx+1)) # 1D input
        dataname = 'ADR_fD_data_huge'
        model_settings = [([150]*2,silu,1,False),([150]*2,silu,1,False),([150]*3,silu,1,False)]
        branch_model_1 = MLP(*model_settings[0])
        branch_model_2 = MLP(*model_settings[1])
        trunk_model = MLP(*model_settings[2])
    m = (Nx+1)*(Ny+1)  # number of trunk input sensors
    xy_in_foo = jnp.zeros((m,2))

    key = random.key(1234)
    key, *keys = random.split(key,6)
    branch_params_1 = branch_model_1.init(keys[0], u_f_in_foo)
    branch_params_2 = branch_model_2.init(keys[1], u_g_in_foo)
    trunk_params = trunk_model.init(keys[2], xy_in_foo)
    last_params_1 = he_normal()(keys[3],(model_settings[0][-4][-1],model_settings[-1][-4][-1]))
    last_params_2 = he_normal()(keys[4],(model_settings[1][-4][-1],model_settings[-1][-4][-1]))

    key, *keys = random.split(key,4)
    branch_params_1 = apply_he(branch_params_1, model_settings, 0, keys[0], gmode='N',scale_b=0)
    branch_params_2 = apply_he(branch_params_2, model_settings, 0, keys[1], gmode='N',scale_b=0)
    trunk_params = apply_he(trunk_params, model_settings, 1, keys[2], gmode='N',scale_b=0)
    params = {'branch1': branch_params_1, 'branch2': branch_params_2, 'trunk': trunk_params, 'last1': last_params_1, 'last2': last_params_2}
    
    # data load & generation
    data_dir = datafolder + dataname
    with open(data_dir + '.pkl', 'rb') as file:
        data = pickle.load(file)
    j = 1 # jump/stride
    uin_f_train = jnp.asarray(data['input_train_f'].astype('float64'))[:N_train,:][:,::j]
    uin_f_val = jnp.asarray(data['input_val_f'].astype('float64'))[:N_test,:][:,::j]
    uin_g_train = jnp.asarray(data['input_train_D'].astype('float64'))[:N_train,:][:,::j]
    uin_g_val = jnp.asarray(data['input_val_D'].astype('float64'))[:N_test,:][:,::j]
    uout_train_pre = jnp.asarray(data['output_train'].astype('float64'))[:N_train,:N_train,:,:][:,:,::j,:][:,:,:,::j]
    uout_train = jnp.transpose(jnp.diagonal(uout_train_pre, axis1=0, axis2=1),(2,0,1))
    uout_val = jnp.asarray(data['output_val'].astype('float64'))[:N_test,:,:][:,::j,:][:,:,::j]

    # Output sensors
    x_pre = jnp.linspace(xmin,xmax,Nx+1)
    y_pre = jnp.linspace(ymin,ymax,Ny+1)
    x_in,y_in = jnp.meshgrid(x_pre,y_pre,indexing='xy')
    xy_in = jnp.stack((x_in,y_in),axis=-1)
    xy_full = jnp.reshape(xy_in,(-1,2)) 
    
    for seed in seeds:
        for ALS_num in ALS_nums:
            # Adam vs LS+Adam
            if ALS_num > 0: # LS+Adam
                batch_size, batch_new = 100, 50 # predetermined adam batch 100->50
                adam_num = 1 # Adam epochs for each WU (LS step performed for each adam_num epochs)
                reg_lambda = '{:.0e}'.format(weights_init[1])
                model_dir = folder + keyword + '_ALSAdam_L' + str(ALS_num) + 'R' + str(adam_num) + '_' + \
                    str(delay_init_pre) + '_HHH_SSS_bat' + str(batch_size) + 'SQto' + str(batch_new) + 'SQ_' + \
                    model_desc + '_seed_' + seed + '_regul_' + reg_lambda 
            else: # Adam
                batch_size, batch_new = 100, 100 # predetermined adam batch 
                model_dir = folder + keyword + '_Adam_NoLS_HHH_SSS_bat' + str(batch_size) + 'SQ_' + \
                    model_desc + '_seed_' + seed 
            # load train time and find the max epoch less than the train time
            mdir_time = model_dir + '/losses/training_time.npy'
            max_ech = jnp.argmax(jnp.load(mdir_time) > etime) - 1
            # Train/Test data save and plot for the best saved models 
            createFolder(model_dir +'/train_result_time_'+str(etime))
            createFolder(model_dir +'/test_result_time_'+str(etime))
            loadech = [i for i in saveech if i <= max_ech][::-1] 
            for j in loadech:
                # Train data -> Best model with train loss
                try:
                    data_result_to_npy(uin_f_train, uin_g_train, uout_train, xy_in, Q_size, model_dir, keyword,
                                    save_path='/train_result_time_'+str(etime), model_path='/models/model_save_besttrain_'+str(j)+'.pickle') # train
                    break
                except:
                    continue
            for j in loadech:
                # Test data -> Best model with validation accuracy (rel L2 error)
                try:
                    data_result_to_npy(uin_f_val, uin_g_val, uout_val, xy_in, Q_size, model_dir, keyword,
                                    save_path='/test_result_time_'+str(etime), model_path='/models/model_save_bestval_'+str(j)+'.pickle') # test
                    break
                except:
                    continue