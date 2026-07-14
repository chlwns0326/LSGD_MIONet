import sys, logging, warnings
warnings.filterwarnings('ignore')

# imports
import time, jax
import jax.numpy as jnp
from jax import random
jax.config.update('jax_enable_x64', True)
from loss import *
from LSGD import *
from step import *
from misc import *
from tqdm import tqdm
import itertools

# Training stage
def train(params, optimizer, seed, delay_init,
        uin_f_train, uin_g_train, uout_train,
        uin_f_val, uin_g_val, uout_val,
        xy, xy_phys, xy_data,
        weights_init, 
        batch_size, batch_new, adam_num, LS_num, 
        loss_logs, model_dir,
        nIter, disp_count):
    
    N_train_f = jnp.shape(uin_f_train)[0]
    N_train_g = jnp.shape(uin_g_train)[0]
    N_val = jnp.shape(uin_f_val)[0]
    forward_train_batch = 1000
    forward_val_batch = 1000

    opt_state = optimizer.init(params)
    iter_per_epoch_a = N_train_f//batch_size[0]
    iter_per_epoch_b = N_train_g//batch_size[1]
    val_iter = N_val//forward_val_batch
    key = random.key(seed) # 1133 default
    
    savewu =  [10,20,30,50, *range(100,201,20), *range(250,501,50), *range(600,1001,100), 
                *range(1200,2001,200), *range(2500,5001,500), *range(6000,10001,1000), 
                *range(12000,20001,2000), *range(25000,50001,5000), *range(60000,100001,10000), 
                *range(120000,200001,20000), *range(250000,500001,50000), *range(600000,1000001,100000)]
    
    weights = [weights_init[0],weights_init[1],0] # Adam LL
    fg_order = 0 # 0: F->G, 1: G->F, 1 default
    
    start_time = time.time()
    total_time = 0    
    WU_time = [] # 

    key, subkey = random.split(key)
    perm_train = random.permutation(key, N_train_f)
    uin_f = uin_f_train[perm_train[:forward_train_batch],:,:,:]
    uin_g = uin_g_train[perm_train[:forward_train_batch],:]
    uout = uout_train[perm_train[:forward_train_batch],:,:]
    
    num_iters = iter_per_epoch_a*iter_per_epoch_b
    
    # Main training loop
    for wu in tqdm(range(nIter)):
        # timer init
        part_time_1, part_time_2, part_time_3 = 0,0,0
        
        ## Manual weight decay
        ## Decaying weights from a(start) to a*b(end)
        ## Uncomment below section to activate weight decay
        # decay_target = 1e-3 # b
        # decay_start = 100 # Start point of decay, constant before decay_start WU
        # decay_end = 1000 # End point of decay, stay constant afterward
        # if wu > decay_start and wu <= decay_end:
        #     rate = decay_target**(1/(decay_end-decay_start)) # Gradual decay for each WU
        #     p2 = weights[2] * rate
        #     weights = [weights_init[0],weights_init[1],p2] 
        
        # LL regul coeff
        if wu == delay_init:
            weights = [weights_init[0],weights_init[1],weights_init[2]] # LS LL
        # GD step 
        if wu == 0: # No Adam, init state
            # only forward
            losses = loss_comps(params, uin_f, uin_g, uout, xy, xy_phys, xy_data, weights)
        else: # Adam, training backprop
            for ad_repeat in range(adam_num):
                # Batch formation
                params['last1'].block_until_ready() ### Foo code 
                timer_1 = time.perf_counter()
                key, subkey = random.split(key,2)
                perm = random.permutation(key, N_train_f)
                perm2 = random.permutation(subkey, N_train_g)
                finds = jnp.split(perm,iter_per_epoch_a) 
                ginds = jnp.split(perm2,iter_per_epoch_b)
                
                key, subkey = random.split(key,2)
                bat_order = random.permutation(key, num_iters)
                pre_blocks = [i for i in itertools.product(finds,ginds)]
                batch_blocks = [pre_blocks[j] for j in bat_order]
                batch_blocks[0][0].block_until_ready() ### Foo code 
                part_time_1 = part_time_1 + (time.perf_counter()-timer_1)
                
                # Adam iters
                for inum, (find, gind) in enumerate(batch_blocks):
                    params['last1'].block_until_ready() ### Foo code 
                    timer_2 = time.perf_counter()
                    key, subkey = random.split(key)
                    uin_f_batch = uin_f_train[find,:,:,:]
                    uin_g_batch = uin_g_train[gind,:]
                    params, opt_state = step_GD(params, optimizer, opt_state, uin_f_batch, uin_g_batch, xy_phys, xy_data, weights)
                    params['last1'].block_until_ready() ### Foo code 
                    part_time_2 = part_time_2 + (time.perf_counter()-timer_2)
                    
        # LS step 
        if LS_num > 0 and wu >= delay_init:
            if wu == delay_init: # change Adam to ALSAdam       
                lr = opt_state[0]['adam'][0].hyperparams['learning_rate']
                b1 = opt_state[0]['adam'][0].hyperparams['b1']
                b2 = opt_state[0]['adam'][0].hyperparams['b2']
                optimizer = optax.multi_transform({'adam': optax.inject_hyperparams(optax.adam)(lr,b1=b1,b2=b2), 'zero': optax.set_to_zero()},
                    {'branch1':'adam', 'branch2':'adam', 'trunk':'adam', 'last1':'zero', 'last2':'zero'})
                opt_state = optimizer.init(params)
                iter_per_epoch_a = N_train_f//batch_new[0]
                iter_per_epoch_b = N_train_g//batch_new[1]
                num_iters = iter_per_epoch_a*iter_per_epoch_b 
            
            for repLS in range(LS_num): # 1
                params['last1'].block_until_ready() ### Foo code 
                timer_3 = time.perf_counter()
                params = step_LS_alt_order(params, uin_f_train, uin_g_train, xy_phys, xy_data, weights, order = fg_order) # 0 okey[0]
                params['last1'].block_until_ready() ### Foo code 
                part_time_3 = part_time_3 + (time.perf_counter()-timer_3)
        
        # Time computation
        part_time = part_time_1 + part_time_2 + part_time_3
        total_time = total_time + part_time
        WU_time.append(total_time)
        
        # Forward        
        losses = loss_comps(params, uin_f, uin_g, uout, xy, xy_phys, xy_data, weights)
        loss_logs['loss_WU'].append(losses)
        loss_logs['weight_WU'].append(weights)
        
        # Validation
        losses_val = jnp.zeros((6,))
        for i in range(val_iter):
            i1 = i * forward_val_batch
            i2 = (i+1) * forward_val_batch
            losses_val = losses_val + loss_comps(params, uin_f_val[i1:i2,:,:,:], uin_g_val[i1:i2,:], uout_val[i1:i2,:,:], xy, xy_phys, xy_data, weights)/val_iter
        loss_logs['loss_WU_val'].append(losses_val)

        if (wu+1) % disp_count == 0:
            # Print losses
            logger1 = (f'WU {wu+1:d} :\t Train total loss :\t\t{losses[0]:.6e}\tPhysics loss :\t{losses[1]:.6e}\tData loss :\t{losses[2]:.6e}\t'
                       f'Regularization loss :\t{losses[3]:.6e}\tL2 Error :\t{losses[4]:.6e}\tRelative L2 Error :\t{losses[5]:.6e}')
            logger2 = (f'WU {wu+1:d} :\t Validation total loss :\t{losses_val[0]:.6e}\tPhysics loss :\t{losses_val[1]:.6e}\tData loss :\t{losses_val[2]:.6e}\t'
                       f'Regularization loss :\t{losses_val[3]:.6e}\tL2 Error :\t{losses_val[4]:.6e}\tRelative L2 Error :\t{losses_val[5]:.6e}')
            logger3 = (f'Work Unit {wu+1:d} :\t Training time: \t{total_time:0.2f} seconds') 
            with open(model_dir + '/result.txt', 'a') as f:
                print(logger1)
                print(logger2)
                print(logger3)
                f.write(logger1 + '\n')  
                f.write(logger2 + '\n') 
                f.write(logger3 + '\n')  
            
        if (wu+1) in savewu or wu+1 == nIter:
            # loss plot
            loss_WU_val = jnp.array(loss_logs['loss_WU'])
            weight_WU_val = jnp.array(loss_logs['weight_WU'])
            loss_WU_val_val = jnp.array(loss_logs['loss_WU_val'])
            colors = ['red','blue','green','yellow','black']
            legend_unsqueeze = ['Physics Loss','Data Loss','Regularization Loss','L2 error','rel L2 error']
            legend_weights = [r'λ_{physics}',r'λ_{data}',r'λ_{LL Regul}']
            title = f'Loss'
            labels_WU = {
                'colors':colors,'legend':legend_unsqueeze,'title':title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU.png'}
            labels_WU_total = {
                'colors':colors, 'legend':['Total Loss'],'title':title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU_total.png'}
            labels_weights = {
                'colors':colors, 'legend':legend_weights,'title':'Weights','xlabel':'Work Unit','ylabel':'weight',
                'save_dir':model_dir+'/losses/weight_WU.png'}
            labels_WU_val = {
                'colors':colors,'legend':legend_unsqueeze,'title':'Validation '+title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU_val.png'}
            labels_WU_val_total = {
                'colors':colors, 'legend':['Total Loss'],'title':'Validation Total '+title,'xlabel':'Work Unit','ylabel':'Loss',
                'save_dir':model_dir+'/losses/loss_curve_WU_val_total.png'}
            
            loss_plot(loss_WU_val[:,1:], labels_WU, logplot=True)
            loss_plot(loss_WU_val[:,0:1], labels_WU_total, logplot=True)
            loss_plot(weight_WU_val[:,:], labels_weights, logplot=True)
            loss_plot(loss_WU_val_val[:,1:], labels_WU_val, logplot=True)
            loss_plot(loss_WU_val_val[:,0:1], labels_WU_val_total, logplot=True)
            
            jnp.save(model_dir+'/losses/weight',weight_WU_val)
            jnp.save(model_dir+'/losses/training_loss',loss_WU_val)
            jnp.save(model_dir+'/losses/training_loss_val',loss_WU_val_val)
            jnp.save(model_dir+'/losses/training_time',jnp.array(WU_time))
            
        if wu == 0:
            min_loss = losses[0]
            min_loss_val = losses_val[-1]
            
        if wu > 0 and losses[0] < min_loss:
            min_loss = losses[0]
            if wu+1 > 10-1:
                # Model save if best train loss
                wustr = str(min(i for i in savewu if i >= wu+1))
                model_save(data=params, path=model_dir+'/models/model_save_besttrain_'+wustr+'.pickle',overwrite=True)
                
        if wu > 0 and losses_val[-1] < min_loss_val:
            min_loss_val = losses_val[-1]
            if wu+1 > 10-1:
                # Model save if best validation accuracy 
                wustr = str(min(i for i in savewu if i >= wu+1))
                model_save(data=params, path=model_dir+'/models/model_save_bestval_'+wustr+'.pickle',overwrite=True)

    full_time = time.time() - start_time
    logger1 = (f'Training done:\t Epoch {wu+1:d} :\t Training time: {full_time:0.2f} seconds')
    with open(model_dir + '/result.txt', 'a') as f:
        print(logger1)
        f.write(logger1 + '\n') 
