# -*- coding: utf-8 -*-
"""

Description: This script constructs the allocator training dataset used to learn how to optimally allocate between the glass box and black box models.

Inputs:
    - run_machine: the machine used to run the code (e.g. 'local' or 'server' - governs the output directories etc.)

Outputs:
    - For each (dataset, ensemble member model) tuple, three 'losses_[ds_type]_nosplit_all.csv' files, each corresponding to the train, val, and test datasets used to train the allocator

python3 pipeline_step_3.py --run_machine='local' > pipeline_step_3_outputs.txt

"""

###############################################################################

import os
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

from utils import get_superfix_prefix_and_make_dir, get_superfix
from data import get_dataset

###############################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--run_machine', default='local', type=str, metavar='N', help="the machine used to run the code (e.g. 'local' or 'server' - governs the output directories etc)")

args = parser.parse_args()
run_machine = args.run_machine

###############################################################################

dir_superfix = get_superfix(run_machine=run_machine)

experiment_names = [f.name for f in os.scandir(dir_superfix) if f.is_dir()]

for i,experiment_name in enumerate(experiment_names):
    print('('+str(i+1)+'/'+str(len(experiment_names))+') '+experiment_name)
    dataset_name = experiment_name[4:]
    dir_superfix,dir_prefix = get_superfix_prefix_and_make_dir(run_machine=run_machine,experiment_name=experiment_name,overwrite_check=False)
    
    model_names = [f.name for f in os.scandir(dir_prefix) if f.is_dir() and f.name[0:9]!='allocator' and f.name!='data_splits']
    
    pred_dim_underlying,ds_dict = get_dataset(dataset_name=dataset_name)
    problem_type = ds_dict.get('problem_type')
    
    for model_name in model_names:
        
        dir_model = os.path.join(dir_prefix,model_name)
        
        files_are_missing = False
        for ds_type in ['train','val','test']:
            y = ds_dict.get('ds_sklearn').get('ds_'+ds_type).get('y_'+ds_type)
            y_hat_path = os.path.join(dir_model,'y_hat_'+str(ds_type)+'_nosplit_all.csv')
            if Path(y_hat_path).exists():
                y_hat = pd.read_csv(y_hat_path,index_col=0).to_numpy(dtype=np.float32)
                if problem_type == 'classification':
                    y = np.eye(pred_dim_underlying)[y.astype(np.int32)].astype(np.float32)
                    losses = np.sum(-y*np.log(np.maximum(np.minimum(y_hat,1-1e-7),1e-7)),-1) 
                elif problem_type == 'regression':
                    if len(y.shape) == 1:
                        y = np.expand_dims(y,-1)
                    else:
                        pass
                    losses = np.sum((y-y_hat)**2,1)
                else:
                    raise ValueError('pipeline_step_3: This problem_type is not implemented')
                
                np.savetxt(os.path.join(dir_model,'losses_'+ds_type+'_nosplit_all.csv'),losses,delimiter=',')
            else:
                files_are_missing = True
        if files_are_missing:
            print(' - '+model_name+' (x)')
        else:
            print(' - '+model_name)
            
            
    print('\n-----------------------------------------------------------\n')
