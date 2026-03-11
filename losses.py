# -*- coding: utf-8 -*-
"""

Description: This file contains custom tensorflow losses.

"""

import tensorflow as tf

class IndividualMSELoss():
    def __call__(self, y_true, y_pred): #y_true: [n,?3?], y_pred: [n,r,?3?]
        r = tf.squeeze(tf.slice(tf.shape(y_pred),[1],[1])) #int
        y_true = tf.tile(tf.expand_dims(y_true,1),[1,r,1]) #[n,r,?3?]
        i_mse = tf.reduce_mean((y_true-y_pred)**2,-1) #[n,r]
        return i_mse #[n,r]

class MSEDrawLoss():
    def __call__(self, y_true, y_pred, member_weights): #[n,?3?],[n,r,?3?],[n,r]
        if member_weights is None:
            member_weights = 1.
        r = tf.squeeze(tf.slice(tf.shape(y_pred),[1],[1])) #int
        y_true = tf.tile(tf.expand_dims(y_true,1),[1,r,1]) #[n,r,?3?]
        region_draw_weighted_mse = tf.reduce_mean(tf.reduce_sum(member_weights*tf.reduce_mean(tf.math.square(y_pred - y_true),-1),-1)) #float
        return region_draw_weighted_mse #float

class IndividualCrossEntropyLoss():
    def __call__(self, y_true, y_pred, y_pred_is_logit=True): #y_true: [n,?2?], y_pred: [n,r,?2?]
        #y_true: is a probability, y_pred is a logit by default but can also be probability
        r = tf.squeeze(tf.slice(tf.shape(y_pred),[1],[1])) #int
        y_true = tf.tile(tf.expand_dims(y_true,1),[1,r,1]) #[n,r,?2?]
        if y_pred_is_logit:
            y_pred_max = tf.reduce_max(y_pred,axis=2,keepdims=True) #[n,r,1]
            y_pred_stable_log_sum_exp = tf.math.log(tf.reduce_sum(tf.math.exp(y_pred-y_pred_max),axis=2,keepdims=True)) + y_pred_max #[n,r,1]
            i_ce = tf.reduce_sum(y_true*(y_pred_stable_log_sum_exp-y_pred),axis=2) #[n,r]
        else:
            epsilon = 1e-10
            i_ce = tf.reduce_sum(-y_true*tf.math.log(tf.math.maximum(y_pred,epsilon)),-1) #[n,r]
        
        return i_ce #[n,r]

class CrossEntropyDrawLoss():
    def __call__(self, y_true, y_pred, member_weights, y_pred_is_logit=True): #y_true: [n,?3?], y_pred: [n,r,?3?], region_weights: [n,r]
        #y_true: is a probability, y_pred is a logit by default but can also be probability
        if member_weights is None:
            member_weights = 1.
        r = tf.squeeze(tf.slice(tf.shape(y_pred),[1],[1])) #int
        y_true = tf.tile(tf.expand_dims(y_true,1),[1,r,1]) #[n,r,?3?]
        if y_pred_is_logit:
            y_pred_max = tf.reduce_max(y_pred,axis=2,keepdims=True) #[n,r,1]
            y_pred_stable_log_sum_exp = tf.math.log(tf.reduce_sum(tf.math.exp(y_pred-y_pred_max),axis=2,keepdims=True)) + y_pred_max #[n,r,1]
            region_draw_weighted_ce = tf.reduce_mean(tf.reduce_sum(member_weights*tf.reduce_sum(y_true*(y_pred_stable_log_sum_exp-y_pred),axis=2),-1)) #float
        else:
            epsilon = 1e-10
            i_ce = tf.reduce_sum(-y_true*tf.math.log(tf.math.maximum(y_pred,epsilon)),-1) #[n,r]
            region_draw_weighted_ce = tf.reduce_mean(tf.reduce_sum(member_weights*i_ce,-1)) #float
        
        return region_draw_weighted_ce #float
