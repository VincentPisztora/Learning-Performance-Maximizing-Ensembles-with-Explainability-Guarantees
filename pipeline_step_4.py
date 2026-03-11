# -*- coding: utf-8 -*-
"""

Description: This script fits all allocator models for each dataset described in the paper. 
For each (dataset, glass box type, black box type, allocator type) 4-tuple, an allocator
(of the specified type) is learned to optimally allocate prediction between the best glass box
and black box of the specified type available for the given dataset. 

Inputs:
    - run_machine: the machine used to run the code (e.g. 'local' or 'server' - governs the output directories etc.)
    - r_cutoff_type: the dataset partition that is used determine the sufficiency cutoff value for regression tasks ('val', 'train')
    - hyper_set: the size of the hyperparameter search space used to tune sklearn allocators ('med', 'large', 'xl', 'xl2')
    - rank_type: the rule used to break sufficiency ties for allocation ranking - 'bb_minus_gb' corresponds to the ranking used in the paper ('bb_minus_gb', 'gb')
    - feat_type: the set of features used by the allocator for prediction
    - dist_type: the distance type(s) used to measure differences between the glass box and black box losses [applicable only for those feat_types that include measures of how the glass box and black box losses differ]
    - rep: the index of this allocator replicate

Outputs:
    - Three inference files per (dataset, allocator type) tuple - one for each data partition ('out_train.csv', 'out_val.csv', 'out_test.csv') containing:
        - r_hat_[partition]: the final allocator's prediction of the optimal ranking for each observation
        - r_[partition]: the true optimal ranking for each observation
        - y_[partition]: the true underlying task label for each observation
        - y_hat_wb_[partition]: the underlying task perdiction of the glass box for each observation
        - y_hat_bb_[partition]: the underlying task perdiction of the black box for each observation
    - One errors file per (dataset, allocator type) tuple ('errors_underlying_random_and_pred.csv') containing the 
      optimal allocation ranking RMSE for each data partition (train,val,test) for the learned allocator and for the random baseline

python3 pipeline_step_4.py --run_machine='local' --r_cutoff_type='val' --hyper_set='large' --rank_type='bb_minus_gb' --feat_type='all' --dist_type='cemse' --rep=0 > pipeline_step_4_outputs.txt

"""

###############################################################################

import argparse
from utils import get_superfix_prefix_and_make_dir, get_best_val_setting_dnn_model_name
from data import get_dataset, get_batches_per_epoch_and_total_epochs
from utils_2 import get_optimizer, train_allocator_tensorflow, train_allocator_sklearn
from models import TabWRN, IdentityLayer

###############################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--run_machine', default='local', type=str, metavar='N', help="the machine used to run the code (e.g. 'local' or 'server' - governs the output directories etc)")
parser.add_argument('--r_cutoff_type', default='val', type=str, metavar='N', help="the dataset partition that is used determine the sufficiency cutoff value for regression tasks ('val', 'train')")
parser.add_argument('--hyper_set', default='large', type=str, metavar='N', help="the size of the hyperparameter search space used to tune sklearn allocators ('med', 'large', 'xl', 'xl2')")
parser.add_argument('--rank_type', default='bb_minus_gb', type=str, metavar='N', help="the rule used to break sufficiency ties for allocation ranking - 'bb_minus_gb' corresponds to the ranking used in the paper ('bb_minus_gb', 'gb')")
parser.add_argument('--feat_type', default='all', type=str, metavar='N', help="the set of features used by the allocator for prediction")
parser.add_argument('--dist_type', default='cemse', type=str, metavar='N', help="the distance type(s) used to measure differences between the glass box and black box losses [applicable only for those feat_types that include measures of how the glass box and black box losses differ]")
parser.add_argument('--rep', default=0, type=int, metavar='N', help="the index of this allocator replicate")

args = parser.parse_args()
run_machine = args.run_machine
r_cutoff_type = args.r_cutoff_type
hyper_set = args.hyper_set
rank_type = args.rank_type
feat_type = args.feat_type
dist_type = args.dist_type
rep = args.rep

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

allocator_types_list = ['gradient_boosting_trees_regressor', 'dnn']

###############################################################################

def pipeline_step_4(run_machine,dataset_name,experiment_name,gb_model_type,bb_model_type,allocator_type,batch_size,wd,optimizer_type,momentum,init_lr,lr_schedule,rep,r_cutoff_type,hyper_set,rank_type,feat_type,dist_type):
    _,dir_prefix = get_superfix_prefix_and_make_dir(run_machine=run_machine,experiment_name=experiment_name,overwrite_check=False)
    pred_dim_underlying,ds_dict = get_dataset(dataset_name=dataset_name)
    problem_type = ds_dict.get('problem_type')
    
    if bb_model_type == 'dnn':
        bb_model_type = get_best_val_setting_dnn_model_name(experiment_dir_path=dir_prefix,problem_type=problem_type)
    
    if allocator_type == 'gradient_boosting_trees_regressor':
        a_model_type = 'gradient_boosting_trees_regressor_gb_'+gb_model_type+'_bb_'+bb_model_type+'_rep_'+str(rep)+'_f_'+feat_type+'_d_'+dist_type
        train_allocator_sklearn(dir_prefix=dir_prefix,model_type=a_model_type,gb_model_type=gb_model_type,
                                bb_model_type=bb_model_type,ds_dict=ds_dict,rep=rep,r_cutoff_type=r_cutoff_type,
                                hyper_set=hyper_set,rank_type=rank_type,
                                feat_type=feat_type,dist_type=dist_type) 
    elif allocator_type == 'dnn':
        
        batches_per_epoch,epochs = get_batches_per_epoch_and_total_epochs(dataset_name=dataset_name,batch_size=batch_size)
        decay_steps = epochs*batches_per_epoch
        
        categorical_features_indicator_expanded_dict = ds_dict.get('categorical_indicator').copy()
        categorical_features_indicator_expanded_dict.update([('y_hat_gb_'+str(i),False) for i in range(ds_dict.get('pred_dim'))])
        categorical_features_indicator_expanded_dict.update([('y_hat_bb_'+str(i),False) for i in range(ds_dict.get('pred_dim'))])
        categorical_features_indicator_expanded_dict.update({'d_gb_bb_ce':False,'d_gb_bb_mse':False})
        
        categorical_features_vocabulary_size_expanded_dict = ds_dict.get('categorical_vocabulary_size_tensorflow').copy()
        categorical_features_vocabulary_size_expanded_dict.update([('y_hat_gb_'+str(i),None) for i in range(ds_dict.get('pred_dim'))])
        categorical_features_vocabulary_size_expanded_dict.update([('y_hat_bb_'+str(i),None) for i in range(ds_dict.get('pred_dim'))])
        categorical_features_vocabulary_size_expanded_dict.update({'d_gb_bb_ce':None,'d_gb_bb_mse':None})
        
        embeddings_expanded = ds_dict.get('embeddings_tf').copy()
        embeddings_expanded.update([('y_hat_gb_'+str(i),IdentityLayer()) for i in range(ds_dict.get('pred_dim'))])
        embeddings_expanded.update([('y_hat_bb_'+str(i),IdentityLayer()) for i in range(ds_dict.get('pred_dim'))])
        embeddings_expanded.update({'d_gb_bb_ce':IdentityLayer(),'d_gb_bb_mse':IdentityLayer()})
        
        arch_hyper_dict={'pred_dim':pred_dim_underlying,'n_base_nodes':16,'N':4,'k':2,'drop_p':0.0,'weight_decay':wd,
                         'categorical_features_indicator_dict':categorical_features_indicator_expanded_dict,
                         'categorical_features_vocabulary_size_dict':categorical_features_vocabulary_size_expanded_dict,
                         'embedding_output_dim':1}
        
        model = TabWRN(output_size=1,
                       with_top=True,
                       num_base_nodes=arch_hyper_dict.get('n_base_nodes'),
                       N=arch_hyper_dict.get('N'),
                       k=arch_hyper_dict.get('k'),
                       drop_p=arch_hyper_dict.get('drop_p'),
                       weight_decay=arch_hyper_dict.get('weight_decay'),
                       categorical_features_indicator_dict=arch_hyper_dict.get('categorical_features_indicator_dict'),
                       categorical_features_vocabulary_size_dict=arch_hyper_dict.get('categorical_features_vocabulary_size_dict'),
                       embedding_output_dim=1)
        
        opt = get_optimizer(optimizer_type=optimizer_type,momentum=momentum,init_lr=init_lr,lr_schedule=lr_schedule,decay_steps=decay_steps) 
        
        a_model_type = 'allocator_dnn_gb_'+gb_model_type+'_bb_'+bb_model_type+'_opt_'+optimizer_type+'_sch_'+lr_schedule+'_lr_'+str(init_lr)+'_wd_'+str(wd)+'_rep_'+str(rep)+'_f_'+feat_type+'_d_'+dist_type
        train_allocator_tensorflow(dir_prefix=dir_prefix,model_type=a_model_type,model=model,
                                   gb_model_type=gb_model_type,bb_model_type=bb_model_type,
                                   ds_dict=ds_dict,epochs=epochs,opt=opt,batch_size=batch_size,
                                   categorical_features_indicator_dict=categorical_features_indicator_expanded_dict,
                                   embeddings=embeddings_expanded,r_cutoff_type=r_cutoff_type,rank_type=rank_type,
                                   feat_type=feat_type,dist_type=dist_type)
    else:
        raise ValueError('pipeline_step_4: This allocator_type is not implemented')


###############################################################################


for dataset_name in dataset_names_list:
    experiment_name = 'Exp_'+dataset_name
    if dataset_name[-2:] == '_R':
        gb_model_types_list = ['regression', 'regression_tree']
        bb_model_types_list = ['dnn', 'gradient_boosting_trees_regressor']
    else:
        gb_model_types_list = ['logistic_regression', 'classification_tree']
        bb_model_types_list = ['dnn', 'gradient_boosting_trees_classifier']
        
    _,dir_prefix = get_superfix_prefix_and_make_dir(run_machine=run_machine,experiment_name=experiment_name,overwrite_check=False)
    pred_dim_underlying,ds_dict = get_dataset(dataset_name=dataset_name)
    
    for gb_model_type in gb_model_types_list:
        for bb_model_type in bb_model_types_list:
            print(dataset_name,gb_model_type,bb_model_type)
            for allocator_type in allocator_types_list:
                if allocator_type == 'dnn':
                    dnn_specific_params = get_best_val_setting_dnn_model_name(experiment_dir_path=dir_prefix,problem_type=ds_dict.get('problem_type')).split('_')
                    print(dnn_specific_params)
                    wd = float(dnn_specific_params[8])
                    optimizer_type = dnn_specific_params[2]
                    init_lr = float(dnn_specific_params[6])
                    lr_schedule = dnn_specific_params[4]
                    momentum = 0.9
                    batch_size = 64
                    pipeline_step_4(run_machine=run_machine,dataset_name=dataset_name,
                                    experiment_name=experiment_name,gb_model_type=gb_model_type,
                                    bb_model_type=bb_model_type,allocator_type=allocator_type,
                                    batch_size=batch_size,wd=wd,optimizer_type=optimizer_type,
                                    momentum=momentum,init_lr=init_lr,lr_schedule=lr_schedule,rep=rep,
                                    r_cutoff_type=r_cutoff_type,
                                    hyper_set=hyper_set,
                                    rank_type=rank_type,
                                    feat_type=feat_type,dist_type=dist_type)
                else:
                    pipeline_step_4(run_machine=run_machine,dataset_name=dataset_name,
                                        experiment_name=experiment_name,gb_model_type=gb_model_type,
                                        bb_model_type=bb_model_type,allocator_type=allocator_type,
                                        batch_size=None,wd=None,optimizer_type=None,
                                        momentum=None,init_lr=None,lr_schedule=None,rep=rep,
                                        r_cutoff_type=r_cutoff_type,
                                        hyper_set=hyper_set,
                                        rank_type=rank_type,
                                        feat_type=feat_type,dist_type=dist_type)