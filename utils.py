# -*- coding: utf-8 -*-
"""

Description: This file contains utility functions.

Required manual updates:
    - Update model results directory path in get_superfix_prefix_and_make_dir and get_superfix
    
"""

###############################################################################

import os
from pathlib import Path
import pandas as pd

###############################################################################

def get_superfix_prefix_and_make_dir(run_machine,experiment_name,overwrite_check):
    if run_machine == 'local':
        superfix = '/path/to/model/results/dir/' #TODO: Update
        prefix = os.path.join(superfix,experiment_name)
    else:
        raise ValueError('get_superfix_prefix_and_make_dir: This run_machine value is not implemented')
    
    if os.path.isdir(prefix):
        if overwrite_check:
            raise ValueError('get_superfix_prefix_and_make_dir: This experiment directory already exists - previous results would have been overwritten')
        else:
            pass
    else:
        os.mkdir(prefix)
    
    return superfix,prefix

def get_superfix(run_machine):
    if run_machine == 'local':
        superfix = '/path/to/model/results/dir/' #TODO: Update
    else:
        raise ValueError('get_superfix: This run_machine value is not implemented')
    
    return superfix

def get_best_val_setting_dnn_model_name(experiment_dir_path,problem_type):
    dnn_model_dirs = [os.path.join(experiment_dir_path,f.name) for f in os.scandir(experiment_dir_path) if f.is_dir() and f.name[0:3]=='dnn']
    dnn_model_dirs.sort()
    
    if len(dnn_model_dirs)==0:
        raise ValueError('get_best_val_setting_dnn_model_name: There are no underlying task dnn models associated with this experiment')
    
    dnns_with_val_metric = []
    for dnn_model_dir in dnn_model_dirs:
        if problem_type == 'classification':
            accs_dir = os.path.join(dnn_model_dir,'accs_underlying_random_dnn_nosplit_all.csv')
            if Path(accs_dir).exists():
                accs = pd.read_csv(accs_dir,index_col=0)
                val_metric = accs._get_value('val_ds','model_underlying_acc')
            else:
                raise ValueError('get_best_val_setting_dnn_model_name: accs_dir does not exist for dnn model at',dnn_model_dir)
        elif problem_type == 'regression':
            errors_dir = os.path.join(dnn_model_dir,'errors_underlying_random_dnn_nosplit_all.csv')
            if Path(errors_dir).exists():
                errors = pd.read_csv(errors_dir,index_col=0)
                val_metric = errors._get_value('val_ds','model_underlying_rmse')
            else:
                raise ValueError('get_best_val_setting_dnn_model_name: errors_dir does not exist for dnn model at',dnn_model_dir)
        else:
            raise ValueError('get_best_val_setting_dnn_model_name: This problem_type is not implemented')
    
        dnns_with_val_metric.append([dnn_model_dir,val_metric])
    
    dnns_with_val_metric_pd = pd.DataFrame(dnns_with_val_metric,columns=['dnn_model_dir','val_metric'])
    dnns_with_val_metric_pd = dnns_with_val_metric_pd.sort_values('val_metric',ascending=True)
    if problem_type == 'classification':
        best_dnn_dir = dnns_with_val_metric_pd.iloc[[-1]]['dnn_model_dir'].values[0]
    elif problem_type == 'regression':
        best_dnn_dir = dnns_with_val_metric_pd.iloc[[0]]['dnn_model_dir'].values[0]
    else:
        raise ValueError('get_best_val_setting_dnn_model_name: This problem_type is not implemented')
    
    best_dnn_name = best_dnn_dir.split(os.sep)[-1]
    
    return best_dnn_name

