# -*- coding: utf-8 -*-
"""

Description: This file contains functions related to data (loading, formatting, saving, etc.).

"""

###############################################################################

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from models import IdentityLayer

###############################################################################

def scale_to_minus_one_to_one(X):
    mn = X.min(0)
    X = X-mn
    mx = X.max(0)
    X = 2.*X/mx-1.
    return X,mn,mx

def scale_to_min_max(X,mn,mx):
    X = X-mn
    X = 2.*X/mx-1.
    return X

def sk_to_tf_ds(x_train,y_train,x_val,y_val,x_test,y_test,categorical_indicator,embeddings):
    
    x_train_tf = {key: np.expand_dims(value,-1) for key, value in x_train.items()} #{'feature_name':[N,1] dtype=any (e.g. string,float,int)}
    train_ds_tf = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    x_val_tf = {key: np.expand_dims(value,-1) for key, value in x_val.items()} #{'feature_name':[N,1] dtype=any (e.g. string,float,int)}
    val_ds_tf = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    x_test_tf = {key: np.expand_dims(value,-1) for key, value in x_test.items()} #{'feature_name':[N,1] dtype=any (e.g. string,float,int)}
    test_ds_tf = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    for feature_name,categorical_feature in categorical_indicator.items():
        x_train_tf.update({feature_name:embeddings.get(feature_name)(x_train_tf.get(feature_name))}) #does feature need to have the same batch size as used to train embedding?
    train_ds_tf = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    for feature_name,categorical_feature in categorical_indicator.items():
        x_val_tf.update({feature_name:embeddings.get(feature_name)(x_val_tf.get(feature_name))}) #does feature need to have the same batch size as used to train embedding?
    val_ds_tf = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    for feature_name,categorical_feature in categorical_indicator.items():
        x_test_tf.update({feature_name:embeddings.get(feature_name)(x_test_tf.get(feature_name))})
    test_ds_tf = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
        
    ds_tensorflow = {'ds_train':train_ds_tf,'ds_val':val_ds_tf,'ds_test':test_ds_tf}
    
    return ds_tensorflow

def get_batches_per_epoch_and_total_epochs(dataset_name,batch_size):
    if dataset_name == 'Wine':
        epochs = 800
        batches_per_epoch = int(2554*.7/batch_size)
    elif dataset_name == 'Phoneme':
        epochs = 800
        batches_per_epoch = int(3172*.7/batch_size)
    elif dataset_name == 'EyeMovements':
        epochs = 800
        batches_per_epoch = int(7608*.7/batch_size)
    elif dataset_name == 'Electricity':
        epochs = 400
        batches_per_epoch = int(38474*.7/batch_size)
    elif dataset_name == 'Jannis':
        epochs = 400
        batches_per_epoch = int(57580*.7/batch_size)
    elif dataset_name == 'MiniBooNE':
        epochs = 400
        batches_per_epoch = int(72998*.7/batch_size)
    elif dataset_name == 'Covertype':
        epochs = 100
        batches_per_epoch = int(566602*.7/batch_size)
    elif dataset_name == 'Pol':
        epochs = 400
        batches_per_epoch = int(10082*.7/batch_size)
    elif dataset_name == 'House16H':
        epochs = 400
        batches_per_epoch = int(13488*.7/batch_size)
    elif dataset_name == 'KDDIPUMS':
        epochs = 800
        batches_per_epoch = int(5188*.7/batch_size)
    elif dataset_name == 'MagicTelescope':
        epochs = 400
        batches_per_epoch = int(13376*.7/batch_size)
    elif dataset_name == 'Bank':
        epochs = 400
        batches_per_epoch = int(10578*.7/batch_size)
    elif dataset_name == 'Higgs':
        epochs = 100
        batches_per_epoch = int(940160*.7/batch_size)
    elif dataset_name == 'Credit':
        epochs = 400
        batches_per_epoch = int(16714*.7/batch_size)
    elif dataset_name == 'California':
        epochs = 400
        batches_per_epoch = int(20634*.7/batch_size)
    elif dataset_name == 'Sim1':
        epochs = 1000
        batches_per_epoch = 100 #TODO
    elif dataset_name == 'CPU_R':
        epochs = 800
        batches_per_epoch = int(8192*.7/batch_size)
    elif dataset_name == 'Pol_R':
        epochs = 400
        batches_per_epoch = int(15000*.7/batch_size)
    elif dataset_name == 'Elevators_R':
        epochs = 400
        batches_per_epoch = int(16599*.7/batch_size)
    elif dataset_name == 'Isolet_R':
        epochs = 800
        batches_per_epoch = int(7797*.7/batch_size)
    elif dataset_name == 'Wine_R':
        epochs = 800
        batches_per_epoch = int(6497*.7/batch_size)
    elif dataset_name == 'Ailerons_R':
        epochs = 400
        batches_per_epoch = int(13750*.7/batch_size)
    elif dataset_name == 'Houses_R':
        epochs = 400
        batches_per_epoch = int(20640*.7/batch_size)
    elif dataset_name == 'House16H_R':
        epochs = 400
        batches_per_epoch = int(22784*.7/batch_size)
    elif dataset_name == 'Diamonds_R':
        epochs = 400
        batches_per_epoch = int(53940*.7/batch_size)
    elif dataset_name == 'BrazilianHouses_R':
        epochs = 400
        batches_per_epoch = int(10692*.7/batch_size)
    elif dataset_name == 'BikeSharingDemand_R':
        epochs = 400
        batches_per_epoch = int(17379*.7/batch_size)
    elif dataset_name == 'NYCTaxi_R':
        epochs = 100
        batches_per_epoch = int(581835*.7/batch_size)
    elif dataset_name == 'HouseSales_R':
        epochs = 400
        batches_per_epoch = int(21613*.7/batch_size)
    elif dataset_name == 'Sulfur_R':
        epochs = 400
        batches_per_epoch = int(10081*.7/batch_size)
    elif dataset_name == 'MedicalCharges_R':
        epochs = 100
        batches_per_epoch = int(163065*.7/batch_size)
    elif dataset_name == 'MiamiHousing_R':
        epochs = 400
        batches_per_epoch = int(13932*.7/batch_size)
    elif dataset_name == 'Superconduct_R':
        epochs = 400
        batches_per_epoch = int(21263*.7/batch_size)
    elif dataset_name == 'California_R':
        epochs = 400
        batches_per_epoch = int(20640*.7/batch_size)
    elif dataset_name == 'Fifa_R':
        epochs = 400
        batches_per_epoch = int(18063*.7/batch_size)
    elif dataset_name == 'Year_R':
        epochs = 100
        batches_per_epoch = int(515345*.7/batch_size)
    else:
        raise ValueError('get_batches_per_epoch_and_total_epochs: dataset_name not implemented')
        
    return batches_per_epoch,epochs

def get_dataset(dataset_name):
    seed = 1 #used to shuffle to dataset before spliting into train and test
    
    if dataset_name == 'Wine':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getWine(seed=seed,prefix=None)
    elif dataset_name == 'EyeMovements':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getEyeMovements(seed=seed,prefix=None)
    elif dataset_name == 'Phoneme':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getPhoneme(seed=seed,prefix=None)
    elif dataset_name == 'Electricity':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getElectricity(seed=seed,prefix=None)
    elif dataset_name == 'Jannis':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getJannis(seed=seed,prefix=None)
    elif dataset_name == 'MiniBooNE':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getMiniBooNE(seed=seed,prefix=None)
    elif dataset_name == 'Covertype':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getCovertype(prefix=None,seed=seed)
    elif dataset_name == 'Pol':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getPol(prefix=None,seed=seed)
    elif dataset_name == 'House16H':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getHouse16H(prefix=None,seed=seed)
    elif dataset_name == 'KDDIPUMS':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getKDDIPUMS(prefix=None,seed=seed)
    elif dataset_name == 'MagicTelescope':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getMagicTelescope(prefix=None,seed=seed)
    elif dataset_name == 'Bank':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getBank(prefix=None,seed=seed)
    elif dataset_name == 'Higgs':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getHiggs(prefix=None,seed=seed)
    elif dataset_name == 'Credit':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getCredit(prefix=None,seed=seed)
    elif dataset_name == 'California':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getCalifornia(prefix=None,seed=seed)
    elif dataset_name == 'Sim1':
        problem_type = 'classification'
        pred_dim = 2
        ds_dict = getSim1(seed=seed,prefix=None)
    elif dataset_name == 'CPU_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getCPU_R(seed=seed,prefix=None)
    elif dataset_name == 'Pol_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getPol_R(seed=seed,prefix=None)
    elif dataset_name == 'Elevators_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getElevators_R(seed=seed,prefix=None)
    elif dataset_name == 'Isolet_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getIsolet_R(seed=seed,prefix=None)
    elif dataset_name == 'Wine_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getWine_R(seed=seed,prefix=None)
    elif dataset_name == 'Ailerons_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getAilerons_R(seed=seed,prefix=None)
    elif dataset_name == 'Houses_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getHouses_R(seed=seed,prefix=None)
    elif dataset_name == 'House16H_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getHouse16H_R(seed=seed,prefix=None)
    elif dataset_name == 'Diamonds_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getDiamonds_R(seed=seed,prefix=None)
    elif dataset_name == 'BrazilianHouses_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getBrazilianHouses_R(seed=seed,prefix=None)
    elif dataset_name == 'BikeSharingDemand_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getBikeSharingDemand_R(seed=seed,prefix=None)
    elif dataset_name == 'NYCTaxi_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getNYCTaxi_R(seed=seed,prefix=None)
    elif dataset_name == 'HouseSales_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getHouseSales_R(seed=seed,prefix=None)
    elif dataset_name == 'Sulfur_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getSulfur_R(seed=seed,prefix=None)
    elif dataset_name == 'MedicalCharges_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getMedicalCharges_R(seed=seed,prefix=None)
    elif dataset_name == 'MiamiHousing_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getMiamiHousing_R(seed=seed,prefix=None)
    elif dataset_name == 'Superconduct_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getSuperconduct_R(seed=seed,prefix=None)
    elif dataset_name == 'California_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getCalifornia_R(seed=seed,prefix=None)
    elif dataset_name == 'Fifa_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getFifa_R(seed=seed,prefix=None)
    elif dataset_name == 'Year_R':
        problem_type = 'regression'
        pred_dim = 1
        ds_dict = getYear_R(seed=seed,prefix=None)
    else:
        raise ValueError('get_datasets: This dataset_name is not implemented yet')
    
    x_train,x_test,y_train,y_test,categorical_indicator = ds_dict.get('x_train'),ds_dict.get('x_test'),ds_dict.get('y_train'),ds_dict.get('y_test'),ds_dict.get('categorical_indicator')
    n_val = int(y_train.shape[0]*0.09/0.79) #70% of total ds is train, 9% (30% of 30%) is val, and 21% (70% of 30%) is test
    
    if problem_type == 'classification':
        y_train_flat = np.argmax(y_train,1)
        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,stratify=y_train_flat,test_size=n_val,shuffle=True,random_state=seed)
    elif problem_type == 'regression':
        x_train, x_val, y_train, y_val = train_test_split(x_train,y_train,test_size=n_val,shuffle=True,random_state=seed)
    
    ##########################################
    
    x_train_tf = {key: np.expand_dims(value,-1) for key, value in x_train.items()} #{'feature_name':[N,1] dtype=any (e.g. string,float,int)}
    train_ds_tf = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    x_val_tf = {key: np.expand_dims(value,-1) for key, value in x_val.items()} #{'feature_name':[N,1] dtype=any (e.g. string,float,int)}
    val_ds_tf = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    x_test_tf = {key: np.expand_dims(value,-1) for key, value in x_test.items()} #{'feature_name':[N,1] dtype=any (e.g. string,float,int)}
    test_ds_tf = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    #-----
    embeddings = {}
    categorical_vocabulary_size_tf = {}
    for feature_name,categorical_feature in categorical_indicator.items():
        feature = train_ds_tf.map(lambda x,y:x.get(feature_name)).batch(4096)
        if categorical_feature:
            print('Warning (get_dataset): tf_dataset generation, categorical feature detected - untested map-adapt-apply embedding pipeline')
            print('Warning (get_dataset): tf_dataset generation, categorical feature detected - must be coded as a string as StringLookup embedding has been applied')
            embedding = StringLookup()
            embedding.adapt(feature)
            vocabulary_size = embedding.vocab_size()
        else:
            embedding = IdentityLayer()
            vocabulary_size = None
        embeddings.update({feature_name:embedding})
        categorical_vocabulary_size_tf.update({feature_name:vocabulary_size})
    
    for feature_name,categorical_feature in categorical_indicator.items():
        x_train_tf.update({feature_name:embeddings.get(feature_name)(x_train_tf.get(feature_name))})
    train_ds_tf = tf.data.Dataset.from_tensor_slices((x_train_tf, y_train)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    for feature_name,categorical_feature in categorical_indicator.items():
        x_val_tf.update({feature_name:embeddings.get(feature_name)(x_val_tf.get(feature_name))}) 
    val_ds_tf = tf.data.Dataset.from_tensor_slices((x_val_tf, y_val)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    
    for feature_name,categorical_feature in categorical_indicator.items():
        x_test_tf.update({feature_name:embeddings.get(feature_name)(x_test_tf.get(feature_name))})
    test_ds_tf = tf.data.Dataset.from_tensor_slices((x_test_tf, y_test)) #ds ({'feature_name':[1,] dtype=any (e.g. string,float,int)}, tf.Tensor [1,p], dtype=float32)
    #-----
    
    ds_tensorflow = {'ds_train':train_ds_tf,'ds_val':val_ds_tf,'ds_test':test_ds_tf}
    
    ###########################################
    
    if problem_type == 'classification':
        y_train_sk = np.argmax(y_train,1).astype(np.float32)
        y_val_sk = np.argmax(y_val,1).astype(np.float32)
        y_test_sk = np.argmax(y_test,1).astype(np.float32)
    elif problem_type == 'regression':
        y_train_sk = y_train.flatten().astype(np.float32)
        y_val_sk = y_val.flatten().astype(np.float32)
        y_test_sk = y_test.flatten().astype(np.float32)
    else:
        raise ValueError('get_dataset: This problem_type is not implemented')
        
    train_ds_sk = {'x_train':x_train,'y_train':y_train_sk}
    val_ds_sk = {'x_val':x_val,'y_val':y_val_sk}
    test_ds_sk = {'x_test':x_test,'y_test':y_test_sk}
    
    #----
    for feature_name,categorical_feature in categorical_indicator.items():
        if categorical_feature:
            print('Warning (get_dataset): sk_dataset generation, categorical feature detected - embedding pipeline not implemented')
    categorical_vocabulary_size_sk = dict(zip(categorical_indicator.keys(), [None]*len(list(categorical_indicator.keys()))))
    #----
    
    ds_sklearn = {'ds_train':train_ds_sk,'ds_val':val_ds_sk,'ds_test':test_ds_sk}
    
    return pred_dim,{'ds_tensorflow':ds_tensorflow,'categorical_vocabulary_size_tensorflow':categorical_vocabulary_size_tf,'embeddings_tf':embeddings,
                     'ds_sklearn':ds_sklearn,'categorical_vocabulary_size_sklearn':categorical_vocabulary_size_sk,
                     'pred_dim':pred_dim,'problem_type':problem_type,'categorical_indicator':categorical_indicator}

###############################################################################
###############################################################################

#spiral curve + blocky diamond
def getSim1(prefix,seed):    
    
    np.random.seed(seed)
    
    N = 20000
    n = int(N/2)
    
    x = np.random.uniform(-.5,.5,n)
    y = np.random.uniform(-1,1,n)
    
    r = np.sqrt(x**2+y**2)
        
    theta_predicted = np.sqrt(r*2000)
    
    x_center = -.5
    y_center = 0
    
    x_predicted = r*np.cos(theta_predicted)
    y_predicted = r*np.sin(theta_predicted)
    
    d = np.sqrt((x-x_predicted)**2+(y-y_predicted)**2)
    spiral_thickness = .3
    x_spiral = x+x_center
    y_spiral = y+y_center
    is_in_spiral = d<spiral_thickness
    
    px = np.random.uniform(0,1,n)
    py = np.random.uniform(-1,1,n)
    
    is_in_rect_0 = np.logical_and(np.logical_and(px>.4,px<.6),np.logical_and(py>.25,py<.35))
    is_in_rect_1 = np.logical_and(np.logical_and(px>.3,px<.7),np.logical_and(py>.15,py<.25))
    is_in_rect_2 = np.logical_and(np.logical_and(px>.2,px<.8),np.logical_and(py>.05,py<.15))
    is_in_rect_3 = np.logical_and(np.logical_and(px>.1,px<.9),np.logical_and(py>-.05,py<.05))
    is_in_rect_4 = np.logical_and(np.logical_and(px>.2,px<.8),np.logical_and(py>-.15,py<-.05))
    is_in_rect_5 = np.logical_and(np.logical_and(px>.3,px<.7),np.logical_and(py>-.25,py<-.15))
    is_in_rect_6 = np.logical_and(np.logical_and(px>.4,px<.6),np.logical_and(py>-.35,py<-.25))
    
    is_in_blocky_diamond = is_in_rect_0+is_in_rect_1+is_in_rect_2+is_in_rect_3+is_in_rect_4+is_in_rect_5+is_in_rect_6
    
    x_combined = np.concatenate([x+x_center,px])
    y_combined = np.concatenate([y+y_center,py])
    is_in_shapes = np.concatenate([is_in_spiral,is_in_blocky_diamond])
    
    X = pd.DataFrame(np.stack([x_combined,y_combined],1),columns=['x0','x1'])
    y_flat = is_in_shapes.astype(np.int32)
    
    n_classes = 2
    train_split_p = 0.5
    
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getMiniBooNE(prefix,seed):    
    X,y = fetch_openml(data_id=44128,return_X_y=True) #preprocessed_Grinsztajn version [72998,50], [72998,]
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'True':1,'False':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getJannis(prefix,seed):    
    X,y = fetch_openml(data_id=44131,return_X_y=True) #preprocessed_Grinsztajn version [57580,54], [57580,]
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'1':1,'0':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getWine(prefix,seed):    
    X,y = fetch_openml(data_id=44091,return_X_y=True) #preprocessed_Grinsztajn version [2554,11], [2554,]
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    X.columns = ['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
                 'pH', 'sulphates', 'alcohol'] #if column names contain spaces, error during model saving
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'True':1,'False':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getElectricity(prefix,seed):
    X,y = fetch_openml(data_id=44120,return_X_y=True) #preprocessed_Grinsztajn version [38474,7], [38474,]
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'UP':1,'DOWN':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getEyeMovements(prefix,seed):
    X,y = fetch_openml(data_id=44130,return_X_y=True) #preprocessed_Grinsztajn version [7608,20], [7608,]
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'1':1,'0':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getPhoneme(prefix,seed):
    X,y = fetch_openml(data_id=44127,return_X_y=True) #preprocessed_Grinsztajn version [3172,5], [3172,]
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'2':1,'1':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getCovertype(prefix,seed):
    X,y = fetch_openml(data_id=44121,return_X_y=True) #preprocessed_Grinsztajn version [566602,10], [566602,]
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'1':1,'0':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getPol(prefix,seed):
    X,y = fetch_openml(data_id=44122,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'P':1,'N':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getHouse16H(prefix,seed):
    X,y = fetch_openml(data_id=44123,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'P':1,'N':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getKDDIPUMS(prefix,seed):
    X,y = fetch_openml(data_id=44124,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'P':1,'N':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getMagicTelescope(prefix,seed):
    X,y = fetch_openml(data_id=44125,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'g':1,'h':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getBank(prefix,seed):
    X,y = fetch_openml(data_id=44126,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'2':1,'1':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getHiggs(prefix,seed):
    X,y = fetch_openml(data_id=44129,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'1':1,'0':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getCredit(prefix,seed):
    X,y = fetch_openml(data_id=44089,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'1':1,'0':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getCalifornia(prefix,seed):
    X,y = fetch_openml(data_id=44090,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    n_classes = 2
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y_flat = y.map({'True':1,'False':0})
    y = np.eye(n_classes)[y_flat].astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,stratify=y_flat,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getCPU_R(prefix,seed):
    X,y = fetch_openml(data_id=44132,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getPol_R(prefix,seed):
    X,y = fetch_openml(data_id=44133,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getElevators_R(prefix,seed):
    X,y = fetch_openml(data_id=44134,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getIsolet_R(prefix,seed):
    X,y = fetch_openml(data_id=44135,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getWine_R(prefix,seed):
    X,y = fetch_openml(data_id=44136,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getAilerons_R(prefix,seed):
    X,y = fetch_openml(data_id=44137,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getHouses_R(prefix,seed):
    X,y = fetch_openml(data_id=44138,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getHouse16H_R(prefix,seed):
    X,y = fetch_openml(data_id=44139,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getDiamonds_R(prefix,seed):
    X,y = fetch_openml(data_id=44140,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getBrazilianHouses_R(prefix,seed):
    X,y = fetch_openml(data_id=44141,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getBikeSharingDemand_R(prefix,seed):
    X,y = fetch_openml(data_id=44142,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getNYCTaxi_R(prefix,seed):
    X,y = fetch_openml(data_id=44143,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getHouseSales_R(prefix,seed):
    X,y = fetch_openml(data_id=44144,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getSulfur_R(prefix,seed):
    X,y = fetch_openml(data_id=44145,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getMedicalCharges_R(prefix,seed):
    X,y = fetch_openml(data_id=44146,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getMiamiHousing_R(prefix,seed):
    X,y = fetch_openml(data_id=44147,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getSuperconduct_R(prefix,seed):
    X,y = fetch_openml(data_id=44148,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getCalifornia_R(prefix,seed):
    X,y = fetch_openml(data_id=44025,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getFifa_R(prefix,seed):
    X,y = fetch_openml(data_id=44026,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

def getYear_R(prefix,seed):
    X,y = fetch_openml(data_id=44027,return_X_y=True) 
    idx = np.random.RandomState(seed=seed).permutation(X.index)
    X = X.reindex(idx)
    y = y.reindex(idx)
    
    y = y-y.min(0)
    y = 2.*y/y.max(0)-1.
    
    X = X-X.min(0)
    X = 2.*X/X.max(0)-1.
    
    train_split_p = 0.79 #70% train, 30% of 30% = 9% is val, rest (21%) is test per Grinsztajn paper
    
    y = np.reshape(y.values,[-1,1]).astype('float32')
    X = X.astype('float32')
    
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=1.-train_split_p,shuffle=True,random_state=seed)
    categorical_indicator = dict(zip(X.columns,list(np.logical_or(X.dtypes==object,X.dtypes=='category'))))
    return {'x_train':x_train,'x_test':x_test,'y_train':y_train,'y_test':y_test,'categorical_indicator':categorical_indicator}

