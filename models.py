# -*- coding: utf-8 -*-
"""

Description: This file contains implementations of tabular wideresnet and gradient boosting trees regressor models.

"""


###############################################################################
#References:
#https://rpmarchildon.com/wp-content/uploads/2018/09/RM-W-Keras-VGG-WRN-vF1.html#section_3
#https://arxiv.org/pdf/1605.07146.pdf
#https://github.com/keras-team/keras/blob/v2.8.0/keras/layers/normalization/batch_normalization.py#L1125-L1265

###############################################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Lambda, ReLU, Dropout
from tensorflow.keras.regularizers import L2
from tensorflow.keras.layers import Embedding

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize


class MSELoss():
    
    def __init__(self):
        pass
    
    def loss(self, y_true, y_pred): #(y_true): [n,], (y_pred): [n,]
        l = (y_true-y_pred)**2
        return np.mean(l)
    
    def negative_gradient(self, y_true, y_pred): #y_true: [n,], y_pred: [n,]
        return -2*(y_true-y_pred)

class GradientBoostingTreesRegressor():
    def __init__(self, n_trees, learning_rate, max_depth, loss=MSELoss(), p_obs=1., p_feat=1., eo=False, n_constructed_feat=3):
        self.n_trees=n_trees
        self.learning_rate=learning_rate
        self.max_depth=max_depth
        self.loss = loss
        self.p_obs = p_obs #proportion dropped
        self.p_feat = p_feat #proportion dropped
        self.eo = eo #'either_or'
        self.n_constructed_feat = n_constructed_feat
        self.feat_subsamples = []
        self.obs_subsamples = []
    
    def fit(self, X, y): #x: [n,?1?], y: [n,]
        if self.eo:
            drop_is = [np.random.binomial(n=1,p=self.p_feat,size=[1,]) for _ in range(self.n_trees)]
            self.feat_subsamples = [np.array([x<1e-5 for x in [1-drop_i]*(X.shape[1]-self.n_constructed_feat)+[drop_i]*self.n_constructed_feat]).flatten() for drop_i in drop_is]
        else:
            self.feat_subsamples = [(np.random.binomial(n=1,p=self.p_feat,size=[X.shape[1],])<1e-5) for _ in range(self.n_trees)]
        
        self.obs_subsamples = [(np.random.binomial(n=1,p=self.p_obs,size=[X.shape[0],])<1e-5) for _ in range(self.n_trees)]
        
        self.trees = []
        self.base_prediction = self._get_optimal_base_value(y=y, loss=self.loss.loss)
        current_predictions = self.base_prediction * np.ones(shape=[y.shape[0],])
        for i in range(self.n_trees):
            feat_subsample = self.feat_subsamples[i]
            obs_subsample = self.obs_subsamples[i]
            pseudo_residuals = self.loss.negative_gradient(y_true=y, y_pred=current_predictions)
                        
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X[obs_subsample][:,feat_subsample], pseudo_residuals[obs_subsample])
            self._update_terminal_nodes(tree=tree, X=X[:,feat_subsample], y=y, current_predictions=current_predictions, loss=self.loss.loss)
            current_predictions += self.learning_rate * tree.predict(X[:,feat_subsample])
            self.trees.append(tree)
    
    def _get_optimal_base_value(self, y, loss):
        fun = lambda c: loss(y_true=y, y_pred=c)
        c0 = y.mean()
        o = minimize(fun=fun, x0=c0).x[0]        
        return o
    
    def _update_terminal_nodes(self, tree, X, y, current_predictions, loss):
        leaf_nodes = np.nonzero(tree.tree_.children_left == -1)[0]
        leaf_node_for_each_sample = tree.apply(X)
        for leaf in leaf_nodes:
            samples_in_this_leaf = np.where(leaf_node_for_each_sample == leaf)[0]
            y_in_leaf = y.take(samples_in_this_leaf, axis=0)
            preds_in_leaf = current_predictions.take(samples_in_this_leaf, axis=0)
            val = self._get_optimal_leaf_value(y=y_in_leaf, 
                                               current_predictions=preds_in_leaf,
                                               loss=loss)
            tree.tree_.value[leaf, 0, 0] = val
    
    def _get_optimal_leaf_value(self, y, current_predictions, loss):
        fun = lambda c: loss(y_true=y, y_pred=current_predictions + c)
        c0 = y.mean()
        o = minimize(fun=fun, x0=c0).x[0]
        return o
    
    def predict(self, X):
        return (self.base_prediction 
                + self.learning_rate 
                * np.sum([tree.predict(X[:,self.feat_subsamples[i]]) for i,tree in enumerate(self.trees)], axis=0))
    
    def get_params(self,deep=True):
        return {'learning_rate':self.learning_rate,'loss':self.loss,'max_depth':self.max_depth,'n_trees':self.n_trees,'p_obs':self.p_obs,'p_feat':self.p_feat,'eo':self.eo, 'n_constructed_feat':self.n_constructed_feat}
        
    def set_params(self,**params):
        self.learning_rate = params.get('learning_rate')
        self.loss = params.get('loss')
        self.max_depth = params.get('max_depth')
        self.n_trees = params.get('n_trees')
        self.p_obs = params.get('p_obs')
        self.p_feat = params.get('p_feat')
        self.eo = params.get('eo')
        self.n_constructed_feat = params.get('n_constructed_feat')
        return self


###############################################################################

class IdentityLayer(keras.layers.Layer):
    def __init__(self):
        super(IdentityLayer, self).__init__()

    def call(self, inputs):
        return inputs

###############################################################################

class CatEmbed(keras.Model):
    def __init__(self,categorical_features_indicator_dict,categorical_features_vocabulary_size_dict,output_dim=1,**kwargs):
        super(CatEmbed,self).__init__(**kwargs)
        self.categorical_features_indicator_dict = categorical_features_indicator_dict
        self.categorical_features_vocabulary_size_dict = categorical_features_vocabulary_size_dict
        self.output_dim = output_dim
        
        def getEmbedding(embed,vocabulary_size):
            if embed == True:
                embedding = Embedding(input_dim=vocabulary_size,output_dim=self.output_dim) #[n,input_length,output_dim]
            else:
                embedding = IdentityLayer() #[n,1]
            return embedding
        
        self.feature_embeddings_dict = {}
        for feature_name,is_categorical_feature in self.categorical_features_indicator_dict.items():
            vocabulary_size = self.categorical_features_vocabulary_size_dict.get(feature_name)
            self.feature_embeddings_dict.update({feature_name:getEmbedding(embed=is_categorical_feature,vocabulary_size=vocabulary_size)})
        
    #inputs: {'feature_name':tensor([n,1]) dtype float32 or int64}
    #output: tensor([n,p*output_dim] dtype float32)
    def call(self, inputs, training=None):
        z = inputs
        z_list = []
        for feature_name,is_categorical_feature in self.categorical_features_indicator_dict.items():
            z_embedded = self.feature_embeddings_dict.get(feature_name)(z.get(feature_name))
            n = tf.squeeze(tf.slice(tf.shape(z_embedded),[0],[1]))
            z_list.append(tf.reshape(z_embedded,[n,-1]))
        z = tf.concat(z_list,1)
        
        return z
    
    def get_config(self):
        config = {}
        config.update({'categorical_features_indicator_dict':self.categorical_features_indicator_dict})
        config.update({'categorical_features_vocabulary_size_dict':self.categorical_features_vocabulary_size_dict})
        config.update({'output_dim':self.output_dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


#####################################
#####################################
#####################################
#Tabular WRN
#####################################
#####################################
#####################################

class TWRND1Block(keras.Model):
    def __init__(self,nodes,weight_decay,**kwargs):
        super(TWRND1Block,self).__init__(**kwargs)
        self.nodes = nodes
        self.weight_decay = weight_decay
        self.d = Dense(units=self.nodes,kernel_regularizer=L2(self.weight_decay))
        
    def call(self,inputs):
        z = self.d(inputs)
        
        return z
    
    def get_config(self):
        config = {}
        config.update({'nodes':self.nodes,'weight_decay':self.weight_decay})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

#l 'deepening factor': num convolutions per block, best performance found in paper with l=2
#N: num blocks within each group [N = (n-4)/(3*l)]
class TWRNGroup(keras.Model):
    def __init__(self,nodes,N,drop_p,weight_decay,**kwargs):
        super(TWRNGroup,self).__init__(**kwargs)        
        self.nodes = nodes
        self.N = N
        self.drop_p = drop_p
        self.weight_decay = weight_decay
        
        self.blocks_dict = {}
        for i in range(self.N):
            self.blocks_dict.update({'TWRN_Block_'+str(i):TWRNMidBlock(block_i=i,
                                                                     nodes=nodes,drop_p=self.drop_p,
                                                                     weight_decay=self.weight_decay)})
        
    def call(self,inputs):
        for key,block in self.blocks_dict.items():
            if block.block_i == 0:
                z = block(inputs)
            else:
                z = block(z)
        return z
    
    def get_config(self):
        config = {}
        config.update({'nodes':self.nodes,'N':self.N,'drop_p':self.drop_p,'weight_decay':self.weight_decay})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TWRNMidBlock(keras.Model):
    def __init__(self,block_i,nodes,drop_p,weight_decay,**kwargs):
        super(TWRNMidBlock,self).__init__(**kwargs)
        self.block_i = block_i
        self.nodes = nodes
        self.drop_p = drop_p
        self.weight_decay = weight_decay
        
        if self.block_i == 0:
            self.res_d = Dense(units=nodes,kernel_regularizer=L2(self.weight_decay))
            self.d0 = Dense(units=nodes,kernel_regularizer=L2(self.weight_decay))
        else:
            self.res_d = Lambda(lambda x: x)
            self.d0 = Dense(units=nodes,kernel_regularizer=L2(self.weight_decay))
        
        self.bn0 = BatchNormalization()
        self.act0 = ReLU()
        self.d1 = Dense(units=nodes,kernel_regularizer=L2(self.weight_decay))
        self.bn1 = BatchNormalization()
        self.act1 = ReLU()
        self.dropout = Dropout(rate=self.drop_p)
        self.add = tf.keras.layers.Add()
        
        self.inputL = tf.keras.layers.InputLayer()
        
    def call(self,inputs):        
        z0 = self.res_d(inputs)
        
        z1 = self.bn0(inputs)
        z1 = self.act0(z1)
        
        z1 = self.d0(z1)
        z1 = self.bn1(z1)
        z1 = self.act1(z1)
        z1 = self.dropout(z1)
        
        z1 = self.d1(z1)
        
        z = self.add([z0, z1])
        
        return z
    
    def get_config(self):
        config = {}
        config.update({'block_i':self.block_i,'nodes':self.nodes,
                       'drop_p':self.drop_p})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


#num_base_filters: num conv filters in conv1_block (that gets multiplied later by k)
#k 'widening factor': multiplies the num of features in conv layers
#l 'deepening factor': num convolutions per block, best performance found in paper with l=2 which is what i use here
#N: num blocks within each group [N = (n-4)/(3*l)]
#n: num of convolutions in whole model [n = 1 + (1+l*N) + (1+l*N) + (1+l*N) -> n = 4 + 3*l*N]
#WRN naming convention is WRN-n-k
#So e.g. (n:28,k:10) == (N:4,k:10)

class TabWRN(keras.Model):
    def __init__(self,output_size,with_top,num_base_nodes,N,k,drop_p,weight_decay,categorical_features_indicator_dict,categorical_features_vocabulary_size_dict,embedding_output_dim,**kwargs):
        super(TabWRN,self).__init__(**kwargs)
        self.inputs_shape = None
        self.output_size = output_size
        self.with_top = with_top
        self.num_base_nodes = num_base_nodes
        self.N = N
        self.k = k
        self.drop_p = drop_p
        self.weight_decay = weight_decay
        self.categorical_features_indicator_dict=categorical_features_indicator_dict
        self.categorical_features_vocabulary_size_dict=categorical_features_vocabulary_size_dict
        self.embedding_output_dim = embedding_output_dim
        self.categorical_embedding = CatEmbed(categorical_features_indicator_dict=self.categorical_features_indicator_dict,
                                              categorical_features_vocabulary_size_dict=self.categorical_features_vocabulary_size_dict,
                                              output_dim=self.embedding_output_dim)
        
        
        self.d1_block = TWRND1Block(nodes=self.num_base_nodes,weight_decay=self.weight_decay)
        self.groups_dict = {}
        for i in range(3):
            if i == 0:
                self.groups_dict.update({'TWRN_Group_'+str(i):TWRNGroup(nodes=self.num_base_nodes*(2**i)*self.k,
                                                                      N=self.N,drop_p=self.drop_p,
                                                                      weight_decay=self.weight_decay)})
            else:
                self.groups_dict.update({'TWRN_Group_'+str(i):TWRNGroup(nodes=self.num_base_nodes*(2**i)*self.k,
                                                                      N=self.N,drop_p=self.drop_p,
                                                                      weight_decay=self.weight_decay)})
        self.bn = BatchNormalization()
        self.act = ReLU()
        #self.avg_pool = GlobalAveragePooling2D()
        self.flat = Flatten()
        if self.with_top:
            self.top_layer = Dense(self.output_size,kernel_regularizer=L2(self.weight_decay))
        else:
            self.top_layer = Lambda(lambda x: x)
        
    def call(self,inputs):
        z = self.categorical_embedding(inputs)
        z = self.d1_block(z)
        for key,group in self.groups_dict.items():
            z = group(z)
        z = self.bn(z)
        z = self.act(z)
        
        #z = self.avg_pool(z)
        z = self.flat(z)
        z = self.top_layer(z)
        
        return z
    
    def get_config(self):
        config = {}
        config.update({'output_size':self.output_size,'with_top':self.with_top,
                       'num_base_nodes':self.num_base_nodes,'N':self.N,
                       'k':self.k,'drop_p':self.drop_p,'weight_decay':self.weight_decay})
        config.update({'categorical_features_indicator_dict':self.categorical_features_indicator_dict})
        config.update({'categorical_features_vocabulary_size_dict':self.categorical_features_vocabulary_size_dict})
        config.update({'embedding_output_dim':self.embedding_output_dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
