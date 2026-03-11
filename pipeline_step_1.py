# -*- coding: utf-8 -*-
"""

Description: This script constructs the underlying task (i.e. regression and classification) datasets.

Inputs:
    - run_machine: the machine used to run the code (e.g. 'local' or 'server' - governs the output directories etc.)
    - n_splits: the number of splits used for hyperparameter tuning (int)
    - split_type: which type of data split is used for training ('kfold', 'bootstrap')
        - 'kfold': the original training dataset is partitioned into n_splits parts and then grouped in kfolds manner
            - For classification tasks, the partitions are stratified by the response (regression tasks are not stratified)

Outputs:
    - A ‘data_splits’ directory in the ‘[experiment_name]’ directory
    - A pickled ‘splits_[split_type]’ file in ‘data_splits’ directory
        - This file contains a list of [n_splits] tuples, each tuple contains two numpy arrays, 
          the first array represents the indexes of the training set, and the second represents the indexes of the validation set

python3 pipeline_step_1.py --run_machine='local' --n_splits=4 --split_type='kfold' > pipeline_step_1_outputs.txt

"""

###############################################################################

import os
from pathlib import Path
import pickle
import numpy as np
from time import time
import argparse

from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold

from utils import get_superfix_prefix_and_make_dir
from data import get_dataset

###############################################################################

parser = argparse.ArgumentParser()

parser.add_argument('--run_machine', default='local', type=str, metavar='N', help="the machine used to run the code (e.g. 'local' or 'server' - governs the output directories etc)")
parser.add_argument('--n_splits', default=4, type=int, metavar='N', help="the number of splits used for hyperparameter tuning (int)")
parser.add_argument('--split_type', default='kfold', type=str, metavar='N', help="which type of data split is used for training ('kfold', 'bootstrap')")

args = parser.parse_args()
run_machine = args.run_machine
n_splits = args.n_splits
split_type = args.split_type

###############################################################################

dataset_names_list = ['Wine',
'Phoneme',
'EyeMovements',
'Electricity',
'Jannis',
'MiniBooNE',
'Covertype',
'Pol',
'House16H',
'KDDIPUMS',
'MagicTelescope',
'Bank',
'Higgs',
'Credit',
'California',
'CPU_R',
'Pol_R',
'Elevators_R',
'Isolet_R',
'Wine_R',
'Ailerons_R',
'Houses_R',
'House16H_R',
'Diamonds_R',
'BrazilianHouses_R',
'BikeSharingDemand_R',
'NYCTaxi_R',
'HouseSales_R',
'Sulfur_R',
'MedicalCharges_R',
'MiamiHousing_R',
'Superconduct_R',
'California_R',
'Fifa_R',
'Year_R']
    
###############################################################################

t1 = time()
for i,dataset_name in enumerate(dataset_names_list):
    print('('+str(i+1)+'/'+str(len(dataset_names_list))+') Processing: '+dataset_name)
    t0 = time()
    experiment_name = 'Exp_'+dataset_name
    
    dir_superfix,dir_prefix = get_superfix_prefix_and_make_dir(run_machine=run_machine,experiment_name=experiment_name,overwrite_check=False)
    dir_splits = os.path.join(dir_prefix,'data_splits')
    Path(dir_splits).mkdir(parents=True, exist_ok=True)
    
    pred_dim_underlying,ds_dict = get_dataset(dataset_name=dataset_name) 
    
    ds = ds_dict.get('ds_sklearn').get('ds_train')
    problem_type = ds_dict.get('problem_type')
    x = ds.get('x_train')
    y = ds.get('y_train')
    n = ds.get('y_train').shape[0]+ds_dict.get('ds_sklearn').get('ds_test').get('y_test').shape[0]+ds_dict.get('ds_sklearn').get('ds_val').get('y_val').shape[0]
    
    if split_type == 'kfold':
        if problem_type == 'classification':
            splits = RepeatedStratifiedKFold(n_splits=n_splits,n_repeats=1,random_state=0).split(X=x,y=y)
        elif problem_type == 'regression':
            splits = RepeatedKFold(n_splits=n_splits,n_repeats=1,random_state=0).split(X=x,y=y)
        else:
            raise ValueError('pipeline_step_1: This problem_type is not implemented')
        splits = list(splits) #shape: list[list[np.array(n_train,),np.array(n_test,)]]
        #Note: the first entry for each split is the "oos test data" the second is the "train data"
    elif split_type == 'bootstrap': 
        raise ValueError('pipeline_step_1: This split_type is not implemented')
    else:
        raise ValueError('pipeline_step_1: This split_type is not implemented')
    
    with open(os.path.join(dir_prefix,'data_splits','splits_'+split_type), 'wb') as f:
        pickle.dump(splits, f)
    
    print(' - n: '+str(n),'| mins:',np.round((time()-t0)/60,2))
    
print('Total time (min):',np.round((time()-t1)/60,2))


