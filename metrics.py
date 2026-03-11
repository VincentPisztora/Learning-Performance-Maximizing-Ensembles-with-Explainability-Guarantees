# -*- coding: utf-8 -*-
"""

Description: This file contains functions related to training metric tracking.

"""

import os
import pandas as pd
import numpy as np

def init_metrics():
    metrics_dict = {'regularization_loss':[],
                    'primary_loss':[],
                    'allocator_loss':[],
                    'allocation_penalty_loss':[],
                    'c_x_primary_loss':[],
                    'c_x_allocator_loss':[],
                    'c_x_allocator_loss_val':[],
                    'c_x_allocator_loss_test':[],
                    'underlying_task_loss_train':[],
                    'underlying_task_loss_val':[],
                    'underlying_task_loss_test':[],
                    'wb_estimation_loss_train':[],
                    'wb_estimation_loss_val':[],
                    'wb_estimation_loss_test':[],
                    'bb_estimation_loss_train':[],
                    'bb_estimation_loss_val':[],
                    'bb_estimation_loss_test':[],
                    'p_hat_entropy':[],
                    'd_hat_entropy':[],
                    'avg_d_hat':[],
                    'avg_p_hat':[],
                    'allocation_prop':[],
                    'delta_loss':[],
                    'delta_loss_val':[],
                    'delta_loss_test':[]}
    return metrics_dict

def update_metrics(metrics_dict,new_metrics_dict):
    for k,v in metrics_dict.items():
        if new_metrics_dict.get(k) is None:
            v.append(0.0)
        else:
            v.append(new_metrics_dict.get(k).numpy())
    return metrics_dict

def save_metrics(metrics_dict,prefix):
    pd.DataFrame(metrics_dict).to_csv(os.path.join(prefix,'training_metrics.csv'))

def summarize_metrics(metrics_dict):
    for k,v in metrics_dict.items():
        metrics_dict.update({k:np.mean(v)})
    return metrics_dict

def update_metrics2(metrics_dict,new_metrics_dict):
    for k,v in metrics_dict.items():
        if new_metrics_dict.get(k) is None:
            v.append(0.0)
        else:
            v.append(new_metrics_dict.get(k))
    return metrics_dict

