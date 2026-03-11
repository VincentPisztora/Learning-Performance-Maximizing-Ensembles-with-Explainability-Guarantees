# -*- coding: utf-8 -*-
"""

Description: This file contains wrapper classes for TF optimizers using custom learning rate schedules.

"""

import tensorflow as tf
from schedules import CosineDecaySchedule


class SGDOptimizerCosineDecayScheduleWrapper(tf.keras.optimizers.SGD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
    
    @classmethod
    def from_config(cls, config):
        custom_objects = {'CosineDecaySchedule':CosineDecaySchedule}
        return super().from_config(config, custom_objects=custom_objects)
    
class AdamOptimizerCosineDecayScheduleWrapper(tf.keras.optimizers.Adam):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
    
    @classmethod
    def from_config(cls, config):
        custom_objects = {'CosineDecaySchedule':CosineDecaySchedule}
        return super().from_config(config, custom_objects=custom_objects)

class RMSpropOptimizerCosineDecayScheduleWrapper(tf.keras.optimizers.RMSprop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @classmethod
    def from_config(cls, config):
        custom_objects = {'CosineDecaySchedule':CosineDecaySchedule}
        return super().from_config(config, custom_objects=custom_objects)