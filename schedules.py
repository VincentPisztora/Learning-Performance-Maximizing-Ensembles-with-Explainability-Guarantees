# -*- coding: utf-8 -*-
"""

Description: This file contains subclassed tensorflow learning rate classes.

"""

import tensorflow as tf
import math

class CosineDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self,initial_learning_rate,decay_steps,alpha=0.0):
        super(CosineDecaySchedule, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.alpha = alpha
        
    def __call__(self,step):
        initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate)
        dtype = initial_learning_rate.dtype
        decay_steps = tf.cast(self.decay_steps,dtype)
        step = tf.cast(step,dtype)
        pi = tf.constant(math.pi, dtype=dtype)
        alpha = tf.cast(self.alpha,dtype)
        
        adj_step = tf.math.minimum(step,decay_steps)
        cosine_decay = 0.5 * (1. + tf.math.cos(pi*adj_step/decay_steps))
        decayed = (1. - alpha) * cosine_decay + alpha
        return initial_learning_rate * decayed
    
    def get_config(self):
        config = {}
        config.update({'initial_learning_rate':self.initial_learning_rate,'decay_steps':self.decay_steps,'alpha':self.alpha})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
