# -*- coding: utf-8 -*-
"""

Description: Description: This file contains utility functions.

"""

import os

from tqdm import tqdm
import numpy as np
import pandas as pd
from time import time
from pathlib import Path

import tensorflow as tf
from losses import MSEDrawLoss, CrossEntropyDrawLoss
from schedules import CosineDecaySchedule
from optimizers import SGDOptimizerCosineDecayScheduleWrapper, AdamOptimizerCosineDecayScheduleWrapper, RMSpropOptimizerCosineDecayScheduleWrapper
from metrics import init_metrics, update_metrics, summarize_metrics, save_metrics, update_metrics2
from data import sk_to_tf_ds, scale_to_minus_one_to_one, scale_to_min_max

from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingRegressor

def get_loss_function(problem_type):
    if problem_type == 'classification':
        loss_function = CrossEntropyDrawLoss()
    elif problem_type == 'regression':
        loss_function = MSEDrawLoss()
    else:
        raise ValueError('get_loss_function: This problem type is not implemented')
    
    return loss_function

def get_subsetted_tf_ds(ds_sk, idx, categorical_indicator, embeddings, problem_type):
    
    x_train = ds_sk.get('x_train') #pd.DataFrame [n,p]
    y_train_sk = ds_sk.get('y_train') #np.array [n,]
    
    if problem_type == 'classification':
        n_classes = np.max(y_train_sk).astype(np.int32) + 1
        y_train = np.eye(n_classes,dtype=np.float32)[y_train_sk.astype(np.int32)]
    elif problem_type == 'regression':
        if len(y_train_sk.shape) == 1:
            y_train = np.expand_dims(y_train_sk,-1)
        else:
            y_train = y_train_sk.copy()
    
    x_train_s = np.take(x_train,idx,0)
    y_train_s = np.take(y_train,idx,0)
    
    x_train_tf = {key: np.expand_dims(value,-1) for key, value in x_train_s.items()} #{'feature_name':[N,1] dtype=any (e.g. string,float,int)}
    
    for feature_name,categorical_feature in categorical_indicator.items():
        x_train_tf.update({feature_name:embeddings.get(feature_name)(x_train_tf.get(feature_name))})
    train_ds_tf = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train_s)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    return train_ds_tf

def get_optimizer(optimizer_type,momentum,init_lr,lr_schedule,decay_steps):
    if lr_schedule == 'constant':
        lr = init_lr
    elif lr_schedule == 'cosine':
        lr = CosineDecaySchedule(initial_learning_rate=init_lr,decay_steps=decay_steps,alpha=0.0)
    else:
        raise ValueError('get_optimizer: This lr_schedule value is not implemented')
    
    if optimizer_type == 'SGD' and lr_schedule == 'cosine':
        opt = SGDOptimizerCosineDecayScheduleWrapper(learning_rate=lr,momentum=momentum,nesterov=True)
    elif optimizer_type == 'SGD' and lr_schedule == 'constant':
        opt = tf.keras.optimizers.SGD(learning_rate=lr,momentum=momentum,nesterov=True)
    elif optimizer_type == 'Adam' and lr_schedule == 'constant':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer_type == 'Adam' and lr_schedule == 'cosine':
        print('Warning (get_optimizer): Have not tested this optimizer schedule combination')
        opt = AdamOptimizerCosineDecayScheduleWrapper(learning_rate=lr)
    elif optimizer_type == 'RMSprop' and lr_schedule == 'constant':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr,rho=0.9,momentum=momentum,epsilon=1e-07,centered=False)
    elif optimizer_type == 'RMSprop' and lr_schedule == 'cosine':
        opt = RMSpropOptimizerCosineDecayScheduleWrapper(learning_rate=lr,rho=0.9,momentum=momentum,epsilon=1e-07,centered=False)
    else:
        raise ValueError('get_optimizer: This (optimizer_type,lr_schedule) combination is not implemented')
        
    return opt

def simpleSupervisedTrainTF(model,train_dataset,val_dataset,test_dataset,batch_size,
                            epochs,loss_fn,opt,prefix,seed=1,buffer_size=1024):
    start_time_training = time()
    metrics_dict = init_metrics()
    
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size,seed=seed).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=buffer_size,seed=seed).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=buffer_size,seed=seed).batch(batch_size)
    
    #--------------------------------------------------------------------------
    @tf.function
    def train_step(x,y): #x: [n,?1?], y: [n,?2?]
        with tf.GradientTape() as t:
            y_hat = model(x,training=True) #[n,?2?]
            
            y_hat = tf.expand_dims(y_hat,1) #[n,1,?2?] <- to make it compatible with the multi-region losses (i.e. r = 1)
            primary_loss = loss_fn(y_true=y,y_pred=y_hat,member_weights=None) #float
            regularization_loss = tf.reduce_sum(model.losses) #float
            loss = primary_loss
            loss += regularization_loss
            
        g = t.gradient(target=loss,sources=model.trainable_variables,unconnected_gradients='zero')
        opt.apply_gradients(zip(g,model.trainable_variables))
        
        metrics_dict = {'regularization_loss':regularization_loss,
                        'primary_loss':primary_loss}
        
        return metrics_dict
                
    #--------------------------------------------------------------------------
    
    for epoch in range(epochs):
        start_time_epoch = time()
        print('---------------------------------\nStart of epoch %d' % (epoch,))
        
        for step,(x,y) in enumerate(tqdm(train_dataset)):
            train_out = train_step(x=x,y=y)
            update_metrics(metrics_dict=metrics_dict,new_metrics_dict=train_out)
        
        t_min = np.floor((time()-start_time_epoch)/60.)
        t_sec = np.mod(time()-start_time_epoch,60.)
        t_min_2 = np.floor((time()-start_time_training)/60.)
        t_sec_2 = np.mod(time()-start_time_training,60.)
        print('\nEnd of epoch %d \n - Epoch runtime (min:sec): %.0f:%.0f \n - Total training time elapsed (min:sec): %.0f:%.0f' % (epoch,t_min,t_sec,t_min_2,t_sec_2))
        print('---------------------------------')
        
        save_metrics(metrics_dict=metrics_dict,prefix=prefix)
    
    return model

def allocatorTrainTF(model,train_dataset,val_dataset,test_dataset,epochs,opt,prefix,batch_size=64,seed=0,buffer_size=1024):
    start_time_training = time()
    metrics_dict = init_metrics()
    
    train_dataset = train_dataset.shuffle(buffer_size=buffer_size,seed=seed).batch(batch_size)
    val_dataset = val_dataset.shuffle(buffer_size=buffer_size,seed=seed).batch(batch_size)
    test_dataset = test_dataset.shuffle(buffer_size=buffer_size,seed=seed).batch(batch_size)
    
    #--------------------------------------------------------------------------
    @tf.function
    def train_step(x,y): #x: [n,?1?], y: [n,1]
        with tf.GradientTape() as t:
            y_hat = model(x,training=True) #[n,?2?]
            primary_loss = tf.reduce_mean((tf.cast(y_hat,tf.float32)-tf.cast(y,tf.float32))**2) #float
            regularization_loss = tf.reduce_sum(model.losses) #float
            loss = primary_loss
            loss += regularization_loss
            
        g = t.gradient(target=loss,sources=model.trainable_variables,unconnected_gradients='zero')
        opt.apply_gradients(zip(g,model.trainable_variables))
        
        metrics_dict = {'regularization_loss':regularization_loss,
                        'primary_loss':primary_loss}
        
        return metrics_dict
                
    #--------------------------------------------------------------------------
    
    for epoch in range(epochs):
        start_time_epoch = time()
        print('---------------------------------\nStart of epoch %d' % (epoch,))
        
        epoch_metrics_dict = init_metrics()
        for step,(x,y) in enumerate(tqdm(train_dataset)):
            train_out = train_step(x=x,y=y)
            epoch_metrics_dict = update_metrics(metrics_dict=epoch_metrics_dict,new_metrics_dict=train_out)
        
        t_min = np.floor((time()-start_time_epoch)/60.)
        t_sec = np.mod(time()-start_time_epoch,60.)
        t_min_2 = np.floor((time()-start_time_training)/60.)
        t_sec_2 = np.mod(time()-start_time_training,60.)
        print('\nEnd of epoch %d \n - Epoch runtime (min:sec): %.0f:%.0f \n - Total training time elapsed (min:sec): %.0f:%.0f' % (epoch,t_min,t_sec,t_min_2,t_sec_2))
        print('---------------------------------')
        
        epoch_metrics_dict = summarize_metrics(metrics_dict=epoch_metrics_dict)
        metrics_dict = update_metrics2(metrics_dict=metrics_dict,new_metrics_dict=epoch_metrics_dict)
        save_metrics(metrics_dict=metrics_dict,prefix=prefix)
    
    return model


def tf_predict_to_np(model,ds,batch_size):
    ds = ds.batch(batch_size)
    y_hat = []
    for step,(x_i,y_i) in enumerate(tqdm(ds)):
        y_hat_i = model(x_i,training=False)
        y_hat.append(y_hat_i)
    y_hat = tf.concat(y_hat,0)
    y_hat = y_hat.numpy()
    return y_hat
    


def get_percentiles_of_a_wrt_b(b,a=None,distr_min=-2.,distr_max=2.): #a: [n_a,1], b: [n_b,1]
            
    if distr_min is None or distr_max is None:
        raise ValueError('get_percentiles_of_a_wrt_b: Unknown distr min and/or max is not implemented')
    
    mm = np.expand_dims(np.array([distr_min,distr_max]),-1)
    b = np.concatenate([b,mm])
    sort_idxs_b = b.argsort(axis=0)
    r_b = sort_idxs_b.argsort(axis=0)
    p_b = r_b/(r_b.shape[0]-1)
    brp = np.concatenate([b,r_b,p_b],1)
    brp_sorted = brp[brp[:,-1].argsort()]
    
    if a is None:
        p_a = p_b[:-2,:] #[n_b,1]
    else:
        U_idx = np.digitize(a.flatten(),bins=brp_sorted[:,0])
        L_idx = U_idx - 1
        q = (brp_sorted[U_idx,0]-a.flatten())/(brp_sorted[U_idx,0] - brp_sorted[L_idx,0])
        q[np.isnan(q)] = 0.
        p_a = np.expand_dims(brp_sorted[L_idx,2]*q+brp_sorted[U_idx,2]*(1-q),-1)
        
    return p_a #[n_a,1]

def train_allocator_tensorflow(dir_prefix,model_type,model,gb_model_type,bb_model_type,ds_dict,epochs,opt,batch_size,categorical_features_indicator_dict,embeddings,r_cutoff_type,rank_type,feat_type,dist_type):
    start_time = time()
    
    model_dir = os.path.join(dir_prefix,model_type)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    problem_type = ds_dict.get('problem_type')
    
    dir_model_gb = os.path.join(dir_prefix,gb_model_type)
    dir_model_bb = os.path.join(dir_prefix,bb_model_type)
    
    ds_sklearn = ds_dict.get('ds_sklearn')
    
    #train---------------------------------------------------------------------
    loss_gb_train = pd.read_csv(os.path.join(dir_model_gb,'losses_train_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
    loss_bb_train = pd.read_csv(os.path.join(dir_model_bb,'losses_train_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
    
    if problem_type == 'classification':
        loss_cutoff = np.log(ds_dict.get('pred_dim')) #>= this value is a misclassification
    elif problem_type == 'regression':
        if r_cutoff_type == 'train':
            loss_cutoff = np.minimum(np.mean(loss_gb_train),np.mean(loss_bb_train)) #>= this value is a misclassification
        elif r_cutoff_type == 'val':
            loss_gb_val = pd.read_csv(os.path.join(dir_model_gb,'losses_val_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
            loss_bb_val = pd.read_csv(os.path.join(dir_model_bb,'losses_val_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
            loss_cutoff = np.minimum(np.mean(loss_gb_val),np.mean(loss_bb_val)) #>= this value is a misclassification
        else:
            raise ValueError('train_allocator_tensorflow: This r_cutoff_type is not implemented')
    else:
        raise ValueError('train_allocator_tensorflow: This problem_type is not implemented')
    
    gb_c = (loss_gb_train<loss_cutoff).astype(np.int32)
    bb_c = (loss_bb_train<loss_cutoff).astype(np.int32)
    if rank_type == 'gb':
        gb_alloc_desireability = (2*gb_c-bb_c-1./(1.+np.exp(-loss_gb_train))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    elif rank_type == 'bb_minus_gb':
        gb_alloc_desireability = (2*gb_c-bb_c-1./(1.+np.exp(loss_bb_train-loss_gb_train))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    else:
        raise ValueError('train_allocator_tensorflow: This rank_type not implemented')
    
    r_train = get_percentiles_of_a_wrt_b(b=gb_alloc_desireability,a=None,distr_min=-2.,distr_max=2.) #[n,1] #[0,1] gb_alloc_desireability percentile (big number means obs should be allocated to gb - it is in the pth percentile of gb desireability)
    
    y_hat_gb = pd.read_csv(os.path.join(dir_model_gb,'y_hat_train_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    y_hat_bb = pd.read_csv(os.path.join(dir_model_bb,'y_hat_train_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    #------------------------------
    if dist_type == 'ce':
        d_gb_bb_0 = np.expand_dims(-np.sum(y_hat_bb*np.log(np.maximum(np.minimum(y_hat_gb,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1 = np.zeros_like(d_gb_bb_0) #[n,1]
    elif dist_type == 'mse':
        d_gb_bb_1 = np.expand_dims(np.sum((y_hat_bb-y_hat_gb)**2,1),-1) 
        d_gb_bb_0 = np.zeros_like(d_gb_bb_1) #[n,1]
    elif dist_type == 'cemse':
        d_gb_bb_0 = np.expand_dims(-np.sum(y_hat_bb*np.log(np.maximum(np.minimum(y_hat_gb,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1 = np.expand_dims(np.sum((y_hat_bb-y_hat_gb)**2,1),-1) 
    else: 
        raise ValueError('train_allocator_tensorflow: This dist_type is not implemented')  
    d_gb_bb = np.hstack([d_gb_bb_0,d_gb_bb_1])
    #------------------------------
    if feat_type == 'x' or feat_type == 'x_dist' or feat_type == 'x_yhats' or feat_type == 'all':
        x_original_train = ds_sklearn.get('ds_train').get('x_train').reset_index(drop=True)
    else:
        temp_x_train = ds_sklearn.get('ds_train').get('x_train').reset_index(drop=True)
        x_original_train = pd.DataFrame(np.zeros_like(temp_x_train),columns=temp_x_train.columns)
    #------------------------------
    if feat_type == 'x_yhats' or feat_type == 'all' or feat_type == 'yhats' or feat_type == 'yhats_dist':
        y_hat_gb_rescaled,mn_gb,mx_gb = scale_to_minus_one_to_one(y_hat_gb)
        y_hat_bb_rescaled,mn_bb,mx_bb = scale_to_minus_one_to_one(y_hat_bb)
    else:
        y_hat_gb_rescaled,mn_gb,mx_gb = np.zeros_like(y_hat_gb),0.,0.
        y_hat_bb_rescaled,mn_bb,mx_bb = np.zeros_like(y_hat_bb),0.,0.
    #------------------------------
    if feat_type == 'x_dist' or feat_type == 'all' or feat_type == 'dist' or feat_type == 'yhats_dist':
        d_gb_bb_rescaled,mn_d,mx_d = scale_to_minus_one_to_one(d_gb_bb)
    else:
        d_gb_bb_rescaled,mn_d,mx_d = np.zeros_like(d_gb_bb),0.,0.
    #------------------------------
    
    x_train = pd.concat([x_original_train,
                         pd.DataFrame(y_hat_gb_rescaled,columns=['y_hat_gb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(y_hat_bb_rescaled,columns=['y_hat_bb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(d_gb_bb_rescaled,columns=['d_gb_bb_ce','d_gb_bb_mse'])],axis=1)
    #train---------------------------------------------------------------------
    
    #val-----------------------------------------------------------------------
    loss_gb_val = pd.read_csv(os.path.join(dir_model_gb,'losses_val_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
    loss_bb_val = pd.read_csv(os.path.join(dir_model_bb,'losses_val_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
        
    gb_c_val = (loss_gb_val<loss_cutoff).astype(np.int32)
    bb_c_val = (loss_bb_val<loss_cutoff).astype(np.int32)
    if rank_type == 'gb':
        gb_alloc_desireability_val = (2*gb_c_val-bb_c_val-1./(1.+np.exp(-loss_gb_val))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    elif rank_type == 'bb_minus_gb':
        gb_alloc_desireability_val = (2*gb_c_val-bb_c_val-1./(1.+np.exp(loss_bb_val-loss_gb_val))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    else:
        raise ValueError('train_allocator_tensorflow: This rank_type not implemented')
        
    r_val = get_percentiles_of_a_wrt_b(b=gb_alloc_desireability,a=gb_alloc_desireability_val,distr_min=-2.,distr_max=2.) #[0,1] gb_alloc_desireability percentile (big number means obs should be allocated to gb - it is in the pth percentile of gb desireability wrt distribution of training gb desireabilities)
    
    y_hat_gb_val = pd.read_csv(os.path.join(dir_model_gb,'y_hat_val_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    y_hat_bb_val = pd.read_csv(os.path.join(dir_model_bb,'y_hat_val_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    #------------------------------
    if dist_type == 'ce':
        d_gb_bb_0_val = np.expand_dims(-np.sum(y_hat_bb_val*np.log(np.maximum(np.minimum(y_hat_gb_val,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1_val = np.zeros_like(d_gb_bb_0_val) #[n,1]
    elif dist_type == 'mse':
        d_gb_bb_1_val = np.expand_dims(np.sum((y_hat_bb_val-y_hat_gb_val)**2,1),-1) 
        d_gb_bb_0_val = np.zeros_like(d_gb_bb_1_val) #[n,1]
    elif dist_type == 'cemse':
        d_gb_bb_0_val = np.expand_dims(-np.sum(y_hat_bb_val*np.log(np.maximum(np.minimum(y_hat_gb_val,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1_val = np.expand_dims(np.sum((y_hat_bb_val-y_hat_gb_val)**2,1),-1)
    else:
        raise ValueError('train_allocator_tensorflow: This dist_type is not implemented')  
    d_gb_bb_val = np.hstack([d_gb_bb_0_val,d_gb_bb_1_val])
    #------------------------------
    if feat_type == 'x' or feat_type == 'x_dist' or feat_type == 'x_yhats' or feat_type == 'all':
        x_original_val = ds_sklearn.get('ds_val').get('x_val').reset_index(drop=True)
    else:
        temp_x_val = ds_sklearn.get('ds_val').get('x_val').reset_index(drop=True)
        x_original_val = pd.DataFrame(np.zeros_like(temp_x_val),columns=temp_x_val.columns)
    #------------------------------
    if feat_type == 'x_yhats' or feat_type == 'all' or feat_type == 'yhats' or feat_type == 'yhats_dist':
        y_hat_gb_val_rescaled = scale_to_min_max(y_hat_gb_val,mn_gb,mx_gb)
        y_hat_bb_val_rescaled = scale_to_min_max(y_hat_bb_val,mn_bb,mx_bb)
    else:
        y_hat_gb_val_rescaled = np.zeros_like(y_hat_gb_val)
        y_hat_bb_val_rescaled = np.zeros_like(y_hat_bb_val)
    #------------------------------
    if feat_type == 'x_dist' or feat_type == 'all' or feat_type == 'dist' or feat_type == 'yhats_dist':
        d_gb_bb_val_rescaled = scale_to_min_max(d_gb_bb_val,mn_d,mx_d)
    else:
        d_gb_bb_val_rescaled = np.zeros_like(d_gb_bb_val)
    #------------------------------
    
    x_val = pd.concat([x_original_val,
                         pd.DataFrame(y_hat_gb_val_rescaled,columns=['y_hat_gb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(y_hat_bb_val_rescaled,columns=['y_hat_bb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(d_gb_bb_val_rescaled,columns=['d_gb_bb_ce','d_gb_bb_mse'])],axis=1)
    
    #val-----------------------------------------------------------------------
    
    #test----------------------------------------------------------------------
    loss_gb_test = pd.read_csv(os.path.join(dir_model_gb,'losses_test_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
    loss_bb_test = pd.read_csv(os.path.join(dir_model_bb,'losses_test_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
        
    gb_c_test = (loss_gb_test<loss_cutoff).astype(np.int32)
    bb_c_test = (loss_bb_test<loss_cutoff).astype(np.int32)
    if rank_type == 'gb':
        gb_alloc_desireability_test = (2*gb_c_test-bb_c_test-1./(1.+np.exp(-loss_gb_test))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    elif rank_type == 'bb_minus_gb':
        gb_alloc_desireability_test = (2*gb_c_test-bb_c_test-1./(1.+np.exp(loss_bb_test-loss_gb_test))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    else:
        raise ValueError('train_allocator_tensorflow: This rank_type not implemented')
        
    r_test = get_percentiles_of_a_wrt_b(b=gb_alloc_desireability,a=gb_alloc_desireability_test,distr_min=-2.,distr_max=2.) #[0,1] gb_alloc_desireability percentile (big number means obs should be allocated to gb - it is in the pth percentile of gb desireability wrt distribution of training gb desireabilities)
    
    y_hat_gb_test = pd.read_csv(os.path.join(dir_model_gb,'y_hat_test_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    y_hat_bb_test = pd.read_csv(os.path.join(dir_model_bb,'y_hat_test_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    
    #------------------------------
    if dist_type == 'ce':
        d_gb_bb_0_test = np.expand_dims(-np.sum(y_hat_bb_test*np.log(np.maximum(np.minimum(y_hat_gb_test,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1_test = np.zeros_like(d_gb_bb_0_test) #[n,1]
    elif dist_type == 'mse':
        d_gb_bb_1_test = np.expand_dims(np.sum((y_hat_bb_test-y_hat_gb_test)**2,1),-1) #[n,1] 
        d_gb_bb_0_test = np.zeros_like(d_gb_bb_1_test) #[n,1]
    elif dist_type == 'cemse':
        d_gb_bb_0_test = np.expand_dims(-np.sum(y_hat_bb_test*np.log(np.maximum(np.minimum(y_hat_gb_test,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1_test = np.expand_dims(np.sum((y_hat_bb_test-y_hat_gb_test)**2,1),-1) #[n,1]
    else: 
        raise ValueError('train_allocator_tensorflow: This dist_type is not implemented')  
    d_gb_bb_test = np.hstack([d_gb_bb_0_test,d_gb_bb_1_test])
    #------------------------------
    if feat_type == 'x' or feat_type == 'x_dist' or feat_type == 'x_yhats' or feat_type == 'all':
        x_original_test = ds_sklearn.get('ds_test').get('x_test').reset_index(drop=True)
    else:
        temp_x_test = ds_sklearn.get('ds_test').get('x_test').reset_index(drop=True)
        x_original_test = pd.DataFrame(np.zeros_like(temp_x_test),columns=temp_x_test.columns)
    #------------------------------
    if feat_type == 'x_yhats' or feat_type == 'all' or feat_type == 'yhats' or feat_type == 'yhats_dist':
        y_hat_gb_test_rescaled = scale_to_min_max(y_hat_gb_test,mn_gb,mx_gb)
        y_hat_bb_test_rescaled = scale_to_min_max(y_hat_bb_test,mn_bb,mx_bb)
    else: 
        y_hat_gb_test_rescaled = np.zeros_like(y_hat_gb_test)
        y_hat_bb_test_rescaled = np.zeros_like(y_hat_bb_test)
    #------------------------------
    if feat_type == 'x_dist' or feat_type == 'all' or feat_type == 'dist' or feat_type == 'yhats_dist':
        d_gb_bb_test_rescaled = scale_to_min_max(d_gb_bb_test,mn_d,mx_d)
    else:
        d_gb_bb_test_rescaled = np.zeros_like(d_gb_bb_test)
    #------------------------------
    x_test = pd.concat([x_original_test,
                         pd.DataFrame(y_hat_gb_test_rescaled,columns=['y_hat_gb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(y_hat_bb_test_rescaled,columns=['y_hat_bb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(d_gb_bb_test_rescaled,columns=['d_gb_bb_ce','d_gb_bb_mse'])],axis=1)
    
    #test----------------------------------------------------------------------
    
    tf_ds_dict = sk_to_tf_ds(x_train=x_train,y_train=r_train,
                             x_val=x_val,y_val=r_val,
                             x_test=x_test,y_test=r_test,
                             categorical_indicator=categorical_features_indicator_dict,
                             embeddings=embeddings)
    
    model = allocatorTrainTF(model=model,
                             train_dataset=tf_ds_dict.get('ds_train'),
                             val_dataset=tf_ds_dict.get('ds_val'),
                             test_dataset=tf_ds_dict.get('ds_test'),
                             epochs=epochs,
                             opt=opt,
                             batch_size=batch_size,
                             prefix=model_dir)
    
    ###########################################################################
    
    if problem_type == 'classification':
        y_train = np.eye(ds_dict.get('pred_dim'),dtype=np.float32)[ds_sklearn.get('ds_train').get('y_train').astype(np.int32)] #[n,p]
        y_val = np.eye(ds_dict.get('pred_dim'),dtype=np.float32)[ds_sklearn.get('ds_val').get('y_val').astype(np.int32)] #[n,p]
        y_test = np.eye(ds_dict.get('pred_dim'),dtype=np.float32)[ds_sklearn.get('ds_test').get('y_test').astype(np.int32)] #[n,p]
    elif problem_type == 'regression':
        y_train = np.expand_dims(ds_sklearn.get('ds_train').get('y_train'),-1) #[n,1]
        y_val = np.expand_dims(ds_sklearn.get('ds_val').get('y_val'),-1) #[n,1]
        y_test = np.expand_dims(ds_sklearn.get('ds_test').get('y_test'),-1) #[n,1]
    else:
        raise ValueError('train_allocator_tensorflow: This problem_type is not implemented')
    
    r_hat_train = tf_predict_to_np(model=model,ds=tf_ds_dict.get('ds_train'),batch_size=batch_size)
    r_hat_val = tf_predict_to_np(model=model,ds=tf_ds_dict.get('ds_val'),batch_size=batch_size)
    r_hat_test = tf_predict_to_np(model=model,ds=tf_ds_dict.get('ds_test'),batch_size=batch_size)
    
    out_train = np.concatenate([r_train,r_hat_train,y_train,y_hat_gb,y_hat_bb],1)
    out_val = np.concatenate([r_val,r_hat_val,y_val,y_hat_gb_val,y_hat_bb_val],1)
    out_test = np.concatenate([r_test,r_hat_test,y_test,y_hat_gb_test,y_hat_bb_test],1)
        
    col_names_train = ['r_train']+['r_hat_train']+['y_train_'+str(i) for i in range(y_train.shape[1])]+['y_hat_wb_train_'+str(i) for i in range(y_train.shape[1])]+['y_hat_bb_train_'+str(i) for i in range(y_train.shape[1])]
    out_train = pd.DataFrame(out_train,columns=col_names_train)
    col_names_val = ['r_val']+['r_hat_val']+['y_val_'+str(i) for i in range(y_val.shape[1])]+['y_hat_wb_val_'+str(i) for i in range(y_val.shape[1])]+['y_hat_bb_val_'+str(i) for i in range(y_val.shape[1])]
    out_val = pd.DataFrame(out_val,columns=col_names_val)
    col_names_test = ['r_test']+['r_hat_test']+['y_test_'+str(i) for i in range(y_test.shape[1])]+['y_hat_wb_test_'+str(i) for i in range(y_test.shape[1])]+['y_hat_bb_test_'+str(i) for i in range(y_test.shape[1])]
    out_test = pd.DataFrame(out_test,columns=col_names_test)
    
    out_train.to_csv(os.path.join(model_dir,'out_train.csv'))
    out_val.to_csv(os.path.join(model_dir,'out_val.csv'))
    out_test.to_csv(os.path.join(model_dir,'out_test.csv'))
    
    stop_time = time()
    run_time = np.round((stop_time-start_time)/60,2)
    print('\ntrain_allocator_tensorflow: total runtime (min)', run_time)
    
    pd.DataFrame([[run_time]],columns=['total_run_time_min']).to_csv(os.path.join(model_dir,'pipeline_step_4_total_run_time.csv'))

    ###########################################################################
    ###########################################################################
        
    error_random_train = np.sqrt(np.mean(np.sum((r_train[np.random.permutation(r_train.shape[0])]-r_train)**2,1),0)+1e-7)
    error_train = np.sqrt(np.mean(np.sum((r_train-r_hat_train)**2,1),0)+1e-7)
    
    error_random_val = np.sqrt(np.mean(np.sum((r_val[np.random.permutation(r_val.shape[0])]-r_val)**2,1),0)+1e-7)
    error_val = np.sqrt(np.mean(np.sum((r_val-r_hat_val)**2,1),0)+1e-7)
    
    error_random_test = np.sqrt(np.mean(np.sum((r_test[np.random.permutation(r_test.shape[0])]-r_test)**2,1),0)+1e-7)
    error_test = np.sqrt(np.mean(np.sum((r_test-r_hat_test)**2,1),0)+1e-7)
    
    print('\nerrors: train (random,wb)', np.round(error_random_train,2), np.round(error_train,2))
    print('errors: val (random,wb)', np.round(error_random_val,2), np.round(error_val,2))
    print('errors: test (random,wb)', np.round(error_random_test,2), np.round(error_test,2))
    
    pd_acc = pd.DataFrame([[error_random_train, error_train],
                           [error_random_val, error_val],
                           [error_random_test, error_test]],
                          index=['train_ds','val_ds','test_ds'],
                          columns=['random_underlying_rmse','model_underlying_rmse'])
    
    pd_acc.to_csv(os.path.join(model_dir,'errors_underlying_random_and_pred.csv'))
    

def train_allocator_sklearn(dir_prefix,model_type,gb_model_type,bb_model_type,ds_dict,rep,r_cutoff_type,hyper_set,rank_type,feat_type,dist_type):
    start_time = time()
    
    model_dir = os.path.join(dir_prefix,'allocator_'+model_type)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    problem_type = ds_dict.get('problem_type')
    
    dir_model_gb = os.path.join(dir_prefix,gb_model_type)
    dir_model_bb = os.path.join(dir_prefix,bb_model_type)
    
    ds_sklearn = ds_dict.get('ds_sklearn')
    
    #train---------------------------------------------------------------------
    loss_gb_train = pd.read_csv(os.path.join(dir_model_gb,'losses_train_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
    loss_bb_train = pd.read_csv(os.path.join(dir_model_bb,'losses_train_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
    
    if problem_type == 'classification':
        loss_cutoff = np.log(ds_dict.get('pred_dim')) #>= this value is a misclassification
    elif problem_type == 'regression':
        if r_cutoff_type == 'train':
            loss_cutoff = np.minimum(np.mean(loss_gb_train),np.mean(loss_bb_train)) #>= this value is a misclassification
        elif r_cutoff_type == 'val':
            loss_gb_val = pd.read_csv(os.path.join(dir_model_gb,'losses_val_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
            loss_bb_val = pd.read_csv(os.path.join(dir_model_bb,'losses_val_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
            loss_cutoff = np.minimum(np.mean(loss_gb_val),np.mean(loss_bb_val)) #>= this value is a misclassification
        else:
            raise ValueError('train_allocator_sklearn: This r_cutoff_type is not implemented')
    else:
        raise ValueError('train_allocator_sklearn: This problem_type is not implemented')
    
    gb_c = (loss_gb_train<loss_cutoff).astype(np.int32)
    bb_c = (loss_bb_train<loss_cutoff).astype(np.int32)
    if rank_type == 'gb':
        gb_alloc_desireability = (2*gb_c-bb_c-1./(1.+np.exp(-loss_gb_train))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    elif rank_type == 'bb_minus_gb':
        gb_alloc_desireability = (2*gb_c-bb_c-1./(1.+np.exp(loss_bb_train-loss_gb_train))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    else:
        raise ValueError('train_allocator_tensorflow: This rank_type not implemented')
    
    r_train = get_percentiles_of_a_wrt_b(b=gb_alloc_desireability,a=None,distr_min=-2.,distr_max=2.) #[0,1] gb_alloc_desireability percentile (big number means obs should be allocated to gb - it is in the pth percentile of gb desireability wrt distribution of training gb desireabilities)
    
    y_hat_gb = pd.read_csv(os.path.join(dir_model_gb,'y_hat_train_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    y_hat_bb = pd.read_csv(os.path.join(dir_model_bb,'y_hat_train_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    #------------------------------
    if dist_type == 'ce':
        d_gb_bb_0 = np.expand_dims(-np.sum(y_hat_bb*np.log(np.maximum(np.minimum(y_hat_gb,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1 = np.zeros_like(d_gb_bb_0) #[n,1]
    elif dist_type == 'mse':
        d_gb_bb_1 = np.expand_dims(np.sum((y_hat_bb-y_hat_gb)**2,1),-1) #[n,1] 
        d_gb_bb_0 = np.zeros_like(d_gb_bb_1) #[n,1]
    elif dist_type == 'cemse':
        d_gb_bb_0 = np.expand_dims(-np.sum(y_hat_bb*np.log(np.maximum(np.minimum(y_hat_gb,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1 = np.expand_dims(np.sum((y_hat_bb-y_hat_gb)**2,1),-1) #[n,1] 
    else: 
        raise ValueError('train_allocator_sklearn: This dist_type is not implemented')  
    d_gb_bb = np.hstack([d_gb_bb_0,d_gb_bb_1])
    #------------------------------
    if feat_type == 'x' or feat_type == 'x_dist' or feat_type == 'x_yhats' or feat_type == 'all':
        x_original_train = ds_sklearn.get('ds_train').get('x_train').reset_index(drop=True)
    else:
        temp_x_train = ds_sklearn.get('ds_train').get('x_train').reset_index(drop=True)
        x_original_train = pd.DataFrame(np.zeros_like(temp_x_train),columns=temp_x_train.columns)
    #------------------------------
    if feat_type == 'x_yhats' or feat_type == 'all' or feat_type == 'yhats' or feat_type == 'yhats_dist':
        y_hat_gb_rescaled,mn_gb,mx_gb = scale_to_minus_one_to_one(y_hat_gb)
        y_hat_bb_rescaled,mn_bb,mx_bb = scale_to_minus_one_to_one(y_hat_bb)
    else:
        y_hat_gb_rescaled,mn_gb,mx_gb = np.zeros_like(y_hat_gb),0.,0.
        y_hat_bb_rescaled,mn_bb,mx_bb = np.zeros_like(y_hat_bb),0.,0.
    #------------------------------
    if feat_type == 'x_dist' or feat_type == 'all' or feat_type == 'dist' or feat_type == 'yhats_dist':
        d_gb_bb_rescaled,mn_d,mx_d = scale_to_minus_one_to_one(d_gb_bb)
    else:
        d_gb_bb_rescaled,mn_d,mx_d = np.zeros_like(d_gb_bb),0.,0.
    #------------------------------
    
    x_train = pd.concat([x_original_train,
                         pd.DataFrame(y_hat_gb_rescaled,columns=['y_hat_gb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(y_hat_bb_rescaled,columns=['y_hat_bb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(d_gb_bb_rescaled,columns=['d_gb_bb_ce','d_gb_bb_mse'])],axis=1)
    x_train = x_train.to_numpy() 
    #train---------------------------------------------------------------------
    
    #val-----------------------------------------------------------------------
    loss_gb_val = pd.read_csv(os.path.join(dir_model_gb,'losses_val_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
    loss_bb_val = pd.read_csv(os.path.join(dir_model_bb,'losses_val_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
        
    gb_c_val = (loss_gb_val<loss_cutoff).astype(np.int32)
    bb_c_val = (loss_bb_val<loss_cutoff).astype(np.int32)
    if rank_type == 'gb':
        gb_alloc_desireability_val = (2*gb_c_val-bb_c_val-1./(1.+np.exp(-loss_gb_val))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    elif rank_type == 'bb_minus_gb':
        gb_alloc_desireability_val = (2*gb_c_val-bb_c_val-1./(1.+np.exp(loss_bb_val-loss_gb_val))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    else:
        raise ValueError('train_allocator_tensorflow: This rank_type not implemented')
    
    r_val = get_percentiles_of_a_wrt_b(b=gb_alloc_desireability,a=gb_alloc_desireability_val,distr_min=-2.,distr_max=2.) #[0,1] gb_alloc_desireability percentile (big number means obs should be allocated to gb - it is in the pth percentile of gb desireability wrt distribution of training gb desireabilities)
    
    y_hat_gb_val = pd.read_csv(os.path.join(dir_model_gb,'y_hat_val_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    y_hat_bb_val = pd.read_csv(os.path.join(dir_model_bb,'y_hat_val_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    #------------------------------
    if dist_type == 'ce':
        d_gb_bb_0_val = np.expand_dims(-np.sum(y_hat_bb_val*np.log(np.maximum(np.minimum(y_hat_gb_val,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1_val = np.zeros_like(d_gb_bb_0_val) #[n,1]
    elif dist_type == 'mse':
        d_gb_bb_1_val = np.expand_dims(np.sum((y_hat_bb_val-y_hat_gb_val)**2,1),-1) #[n,1] 
        d_gb_bb_0_val = np.zeros_like(d_gb_bb_1_val) #[n,1]
    elif dist_type == 'cemse':
        d_gb_bb_0_val = np.expand_dims(-np.sum(y_hat_bb_val*np.log(np.maximum(np.minimum(y_hat_gb_val,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1_val = np.expand_dims(np.sum((y_hat_bb_val-y_hat_gb_val)**2,1),-1) #[n,1] 
    else: 
        raise ValueError('train_allocator_sklearn: This dist_type is not implemented')  
    d_gb_bb_val = np.hstack([d_gb_bb_0_val,d_gb_bb_1_val])
    #------------------------------
    if feat_type == 'x' or feat_type == 'x_dist' or feat_type == 'x_yhats' or feat_type == 'all':
        x_original_val = ds_sklearn.get('ds_val').get('x_val').reset_index(drop=True)
    else:
        temp_x_val = ds_sklearn.get('ds_val').get('x_val').reset_index(drop=True)
        x_original_val = pd.DataFrame(np.zeros_like(temp_x_val),columns=temp_x_val.columns)
    #------------------------------
    if feat_type == 'x_yhats' or feat_type == 'all' or feat_type == 'yhats' or feat_type == 'yhats_dist':
        y_hat_gb_val_rescaled = scale_to_min_max(y_hat_gb_val,mn_gb,mx_gb)
        y_hat_bb_val_rescaled = scale_to_min_max(y_hat_bb_val,mn_bb,mx_bb)
    else:
        y_hat_gb_val_rescaled = np.zeros_like(y_hat_gb_val)
        y_hat_bb_val_rescaled = np.zeros_like(y_hat_bb_val)
    #------------------------------
    if feat_type == 'x_dist' or feat_type == 'all' or feat_type == 'dist' or feat_type == 'yhats_dist':
        d_gb_bb_val_rescaled = scale_to_min_max(d_gb_bb_val,mn_d,mx_d)
    else:
        d_gb_bb_val_rescaled = np.zeros_like(d_gb_bb_val)
    #------------------------------
    
    x_val = pd.concat([x_original_val,
                         pd.DataFrame(y_hat_gb_val_rescaled,columns=['y_hat_gb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(y_hat_bb_val_rescaled,columns=['y_hat_bb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(d_gb_bb_val_rescaled,columns=['d_gb_bb_ce','d_gb_bb_mse'])],axis=1)
    x_val = x_val.to_numpy() 
    #val-----------------------------------------------------------------------
    
    #test----------------------------------------------------------------------
    loss_gb_test = pd.read_csv(os.path.join(dir_model_gb,'losses_test_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
    loss_bb_test = pd.read_csv(os.path.join(dir_model_bb,'losses_test_nosplit_all.csv'),header=None).to_numpy().astype(np.float32) #[n,1]
        
    gb_c_test = (loss_gb_test<loss_cutoff).astype(np.int32)
    bb_c_test = (loss_bb_test<loss_cutoff).astype(np.int32)
    if rank_type == 'gb':
        gb_alloc_desireability_test = (2*gb_c_test-bb_c_test-1./(1.+np.exp(-loss_gb_test))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    elif rank_type == 'bb_minus_gb':
        gb_alloc_desireability_test = (2*gb_c_test-bb_c_test-1./(1.+np.exp(loss_bb_test-loss_gb_test))).astype(np.float32) #[n,1] small number indicates observation should be allocated to bb
    else:
        raise ValueError('train_allocator_tensorflow: This rank_type not implemented')
    
    r_test = get_percentiles_of_a_wrt_b(b=gb_alloc_desireability,a=gb_alloc_desireability_test,distr_min=-2.,distr_max=2.) #[0,1] gb_alloc_desireability percentile (big number means obs should be allocated to gb - it is in the pth percentile of gb desireability wrt distribution of training gb desireabilities)
    
    y_hat_gb_test = pd.read_csv(os.path.join(dir_model_gb,'y_hat_test_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    y_hat_bb_test = pd.read_csv(os.path.join(dir_model_bb,'y_hat_test_nosplit_all.csv'),index_col=0).to_numpy().astype(np.float32) #[n,p]
    #------------------------------
    if dist_type == 'ce':
        d_gb_bb_0_test = np.expand_dims(-np.sum(y_hat_bb_test*np.log(np.maximum(np.minimum(y_hat_gb_test,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1_test = np.zeros_like(d_gb_bb_0_test) #[n,1]
    elif dist_type == 'mse':
        d_gb_bb_1_test = np.expand_dims(np.sum((y_hat_bb_test-y_hat_gb_test)**2,1),-1) #[n,1] 
        d_gb_bb_0_test = np.zeros_like(d_gb_bb_1_test) #[n,1]
    elif dist_type == 'cemse':
        d_gb_bb_0_test = np.expand_dims(-np.sum(y_hat_bb_test*np.log(np.maximum(np.minimum(y_hat_gb_test,1-1e-7),1e-7)),1),-1) #[n,1]
        d_gb_bb_1_test = np.expand_dims(np.sum((y_hat_bb_test-y_hat_gb_test)**2,1),-1) #[n,1] 
    else: 
        raise ValueError('train_allocator_sklearn: This dist_type is not implemented')  
    d_gb_bb_test = np.hstack([d_gb_bb_0_test,d_gb_bb_1_test])
    #------------------------------
    if feat_type == 'x' or feat_type == 'x_dist' or feat_type == 'x_yhats' or feat_type == 'all':
        x_original_test = ds_sklearn.get('ds_test').get('x_test').reset_index(drop=True)
    else:
        temp_x_test = ds_sklearn.get('ds_test').get('x_test').reset_index(drop=True)
        x_original_test = pd.DataFrame(np.zeros_like(temp_x_test),columns=temp_x_test.columns)
    #------------------------------
    if feat_type == 'x_yhats' or feat_type == 'all' or feat_type == 'yhats' or feat_type == 'yhats_dist':
        y_hat_gb_test_rescaled = scale_to_min_max(y_hat_gb_test,mn_gb,mx_gb)
        y_hat_bb_test_rescaled = scale_to_min_max(y_hat_bb_test,mn_bb,mx_bb)
    else: 
        y_hat_gb_test_rescaled = np.zeros_like(y_hat_gb_test)
        y_hat_bb_test_rescaled = np.zeros_like(y_hat_bb_test)
    #------------------------------
    if feat_type == 'x_dist' or feat_type == 'all' or feat_type == 'dist' or feat_type == 'yhats_dist':
        d_gb_bb_test_rescaled = scale_to_min_max(d_gb_bb_test,mn_d,mx_d)
    else:
        d_gb_bb_test_rescaled = np.zeros_like(d_gb_bb_test)
    #------------------------------
    x_test = pd.concat([x_original_test,
                         pd.DataFrame(y_hat_gb_test_rescaled,columns=['y_hat_gb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(y_hat_bb_test_rescaled,columns=['y_hat_bb_'+str(i) for i in range(ds_dict.get('pred_dim'))]),
                         pd.DataFrame(d_gb_bb_test_rescaled,columns=['d_gb_bb_ce','d_gb_bb_mse'])],axis=1)
    x_test = x_test.to_numpy() 
    #test----------------------------------------------------------------------
    
    if hyper_set == 'med':
        param_grid = {'learning_rate':[10**i for i in range(-3,0,1)], 
                      'n_estimators':[2**i for i in range(4,10,1)], 
                      'max_depth':[2**i for i in range(3,7,1)], 
                      'subsample':[2e-1*i for i in range(1,6,1)]} 
    elif hyper_set == 'large':
        param_grid = {'learning_rate':[10**i for i in range(-2,0,1)], 
                      'n_estimators':[2**i for i in range(2,11,2)], 
                      'max_depth':[2**i for i in range(2,6,1)], 
                      'subsample':[25e-2*i for i in range(1,5,1)]} 
    elif hyper_set == 'xl':
        param_grid = {'learning_rate':[10**i for i in range(-3,0,1)], 
                      'n_estimators':[2**i for i in range(0,11,1)],
                      'max_depth':[2**i for i in range(0,6,1)], 
                      'subsample':[25e-2*i for i in range(1,5,1)]} 
    elif hyper_set == 'xl2':
        param_grid = {'learning_rate':[10**i for i in range(-3,0,1)], 
                      'n_estimators':[2**i for i in range(0,11,1)], 
                      'max_depth':[2**i for i in range(0,6,1)], 
                      'subsample':[.05,.1,.15,.2,]} 
    base_estimator = GradientBoostingRegressor(random_state=0)
    cv_folds = RepeatedKFold(n_splits=4,n_repeats=1,random_state=rep) 
    grid_search = GridSearchCV(estimator=base_estimator,
                               param_grid=param_grid,n_jobs=-1,
                               cv=cv_folds,scoring='neg_mean_squared_error',verbose=1)
    search_results_a = grid_search.fit(X=x_train,y=np.reshape(r_train,[-1]))
    a = search_results_a.best_estimator_
    
    print('----------\nTuning complete\nbest estimator:',a,'\n','cv best score:',np.round(search_results_a.best_score_,4),'\n','duration:',np.round((time()-start_time)/60,2),'min')
    pd.DataFrame.from_dict(search_results_a.cv_results_).to_csv(os.path.join(model_dir,'cv_results_allocator.csv'))
    
    ###########################################################################
    
    if problem_type == 'classification':
        y_train = np.eye(ds_dict.get('pred_dim'),dtype=np.float32)[ds_sklearn.get('ds_train').get('y_train').astype(np.int32)] #[n,p]
        y_val = np.eye(ds_dict.get('pred_dim'),dtype=np.float32)[ds_sklearn.get('ds_val').get('y_val').astype(np.int32)] #[n,p]
        y_test = np.eye(ds_dict.get('pred_dim'),dtype=np.float32)[ds_sklearn.get('ds_test').get('y_test').astype(np.int32)] #[n,p]
    elif problem_type == 'regression':
        y_train = np.expand_dims(ds_sklearn.get('ds_train').get('y_train'),-1) #[n,1]
        y_val = np.expand_dims(ds_sklearn.get('ds_val').get('y_val'),-1) #[n,1]
        y_test = np.expand_dims(ds_sklearn.get('ds_test').get('y_test'),-1) #[n,1]
    else:
        raise ValueError('train_allocator_sklearn: This problem_type is not implemented')
    
    r_hat_train = np.expand_dims(a.predict(x_train),-1)
    r_hat_val = np.expand_dims(a.predict(x_val),-1)
    r_hat_test = np.expand_dims(a.predict(x_test),-1)
    
    out_train = np.concatenate([r_train,r_hat_train,y_train,y_hat_gb,y_hat_bb],1)
    out_val = np.concatenate([r_val,r_hat_val,y_val,y_hat_gb_val,y_hat_bb_val],1)
    out_test = np.concatenate([r_test,r_hat_test,y_test,y_hat_gb_test,y_hat_bb_test],1)
        
    col_names_train = ['r_train']+['r_hat_train']+['y_train_'+str(i) for i in range(y_train.shape[1])]+['y_hat_wb_train_'+str(i) for i in range(y_train.shape[1])]+['y_hat_bb_train_'+str(i) for i in range(y_train.shape[1])]
    out_train = pd.DataFrame(out_train,columns=col_names_train)
    col_names_val = ['r_val']+['r_hat_val']+['y_val_'+str(i) for i in range(y_val.shape[1])]+['y_hat_wb_val_'+str(i) for i in range(y_val.shape[1])]+['y_hat_bb_val_'+str(i) for i in range(y_val.shape[1])]
    out_val = pd.DataFrame(out_val,columns=col_names_val)
    col_names_test = ['r_test']+['r_hat_test']+['y_test_'+str(i) for i in range(y_test.shape[1])]+['y_hat_wb_test_'+str(i) for i in range(y_test.shape[1])]+['y_hat_bb_test_'+str(i) for i in range(y_test.shape[1])]
    out_test = pd.DataFrame(out_test,columns=col_names_test)
    
    out_train.to_csv(os.path.join(model_dir,'out_train.csv'))
    out_val.to_csv(os.path.join(model_dir,'out_val.csv'))
    out_test.to_csv(os.path.join(model_dir,'out_test.csv'))
    
    stop_time = time()
    run_time = np.round((stop_time-start_time)/60,2)
    print('\ntrain_allocator_sklearn: total runtime (min)', run_time)
    
    pd.DataFrame([[run_time]],columns=['total_run_time_min']).to_csv(os.path.join(model_dir,'pipeline_step_4_total_run_time.csv'))

    ###########################################################################
    ###########################################################################
        
    error_random_train = np.sqrt(np.mean(np.sum((r_train[np.random.permutation(r_train.shape[0])]-r_train)**2,1),0)+1e-7)
    error_train = np.sqrt(np.mean(np.sum((r_train-r_hat_train)**2,1),0)+1e-7)
    
    error_random_val = np.sqrt(np.mean(np.sum((r_val[np.random.permutation(r_val.shape[0])]-r_val)**2,1),0)+1e-7)
    error_val = np.sqrt(np.mean(np.sum((r_val-r_hat_val)**2,1),0)+1e-7)
    
    error_random_test = np.sqrt(np.mean(np.sum((r_test[np.random.permutation(r_test.shape[0])]-r_test)**2,1),0)+1e-7)
    error_test = np.sqrt(np.mean(np.sum((r_test-r_hat_test)**2,1),0)+1e-7)
    
    print('\nerrors: train (random,wb)', np.round(error_random_train,2), np.round(error_train,2))
    print('errors: val (random,wb)', np.round(error_random_val,2), np.round(error_val,2))
    print('errors: test (random,wb)', np.round(error_random_test,2), np.round(error_test,2))
    
    pd_acc = pd.DataFrame([[error_random_train, error_train],
                           [error_random_val, error_val],
                           [error_random_test, error_test]],
                          index=['train_ds','val_ds','test_ds'],
                          columns=['random_underlying_rmse','model_underlying_rmse'])
    
    pd_acc.to_csv(os.path.join(model_dir,'errors_underlying_random_and_pred.csv'))












