from keras import initializers, regularizers, activations
from keras.layers import Dense, Add, BatchNormalization, PReLU
#from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np


class GraphEmbed(Layer):
    def __init__(self, **kwargs):
        super(GraphEmbed, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GraphEmbed, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # init_feats = (samples, max_atoms, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        init_feats, distances = inputs

        # Get parameters
        max_atoms = int(init_feats.shape[1])
        atom_feat = int(init_feats.shape[-1])
        coor_dims = int(distances.shape[-1])

        # Generate vector features filled with zeros
        vector_features = tf.zeros_like(init_feats)
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, coor_dims, 1])  # (samples, max_atoms, coor_dims, atom_feat)

        return [init_feats, vector_features]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], (input_shape[0][0], input_shape[0][1], input_shape[-1][-1], input_shape[0][-1])]


class GraphSToS(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
            
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GraphSToS, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphSToS, self).get_config()
        base_config['filters'] = self.filters
        return base_config

    def build(self, input_shape):
        atom_feat = input_shape[-1]
        self.w_ss = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_ss')

        if self.bias_initializer is not None:
            self.b_ss = self.add_weight(shape=(self.filters,),
                                        name='b_ss',
                                        initializer=self.bias_initializer)

        super(GraphSToS, self).build(input_shape)

    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        scalar_features = inputs

        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])

        # Expand scalar features to 4D
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
        scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])  # (samples, max_atoms, max_atoms, atom_feat)

        # Combine between atoms
        scalar_features_t = tf.transpose(scalar_features, perm=[0, 2, 1, 3])  # (samples, max_atoms, max_atoms, atom_feat)
        scalar_features = tf.concat([scalar_features, scalar_features_t], -1)  # (samples, max_atoms, max_atoms, atom_feat*2)

        # Linear combination
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat * 2])
        # scalar_features = tf.matmul(scalar_features, self.w_ss) + self.b_ss
        if self.bias_initializer is None:
            scalar_features = tf.matmul(scalar_features, self.w_ss)
        else:
            scalar_features = tf.matmul(scalar_features, self.w_ss) + self.b_ss
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, self.filters])
        
        # multiply bond feature
        if bond is not None:
            scalar_features = tf.linalg.einsum('aijk,aijk->aijk', scalar_features, bond)
        
        # masking
        if mask is not None:
            mask = tf.reshape(mask, [-1, max_atoms, 1])
            mask = tf.tile(mask, [1, 1, max_atoms])
            scalar_features = tf.linalg.einsum('aijk,aij->aijk', scalar_features, mask)

        # Activation
        scalar_features = self.activation(scalar_features)

        return scalar_features

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1], self.filters


class GraphSToV(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GraphSToV, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphSToV, self).get_config()
        base_config['filters'] = self.filters
        return base_config

    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        self.w_sv = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_sv')

        if self.bias_initializer is not None:
            self.b_sv = self.add_weight(shape=(self.filters,),
                                        name='b_sv',
                                        initializer=self.bias_initializer)

        super(GraphSToV, self).build(input_shape)

    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        scalar_features, distances = inputs

        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])
        coor_dims = int(distances.shape[-1])

        # Expand scalar features to 4D
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
        scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])

        # Combine between atoms
        scalar_features_t = tf.transpose(scalar_features, perm=[0, 2, 1, 3])
        scalar_features = tf.concat([scalar_features, scalar_features_t], -1)

        # Apply weights
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat * 2])
        # scalar_features = tf.matmul(scalar_features, self.w_sv) + self.b_sv
        if self.bias_initializer is None:
            scalar_features = tf.matmul(scalar_features, self.w_sv)
        else:
            scalar_features = tf.matmul(scalar_features, self.w_sv) + self.b_sv
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, 1, self.filters])
        scalar_features = tf.tile(scalar_features, [1, 1, 1, coor_dims, 1])

        # multiply bond feature
        if bond is not None:
            scalar_features = tf.linalg.einsum('aijkl,aijl->aijkl', scalar_features, bond)

        # masking
        if mask is not None:
            mask = tf.reshape(mask, [-1, max_atoms, 1])
            mask = tf.tile(mask, [1, 1, max_atoms])
            scalar_features = tf.linalg.einsum('aijkl,aij->aijkl', scalar_features, mask)

        # Expand distances to 5D
        distances = tf.reshape(distances, [-1, max_atoms, max_atoms, coor_dims, 1])
        distances = tf.tile(distances, [1, 1, 1, 1, self.filters])

        # Tensor product
        vector_features = tf.multiply(scalar_features, distances)

        # Activation
        vector_features = self.activation(vector_features)

        return vector_features

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], input_shape[1][-1], self.filters


class GraphVToV(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GraphVToV, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphVToV, self).get_config()
        base_config['filters'] = self.filters
        return base_config

    def build(self, input_shape):
        atom_feat = input_shape[-1]
        self.w_vv = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_vv')

        if self.bias_initializer is not None:
            self.b_vv = self.add_weight(shape=(self.filters,),
                                        name='b_vv',
                                        initializer=self.bias_initializer)

        super(GraphVToV, self).build(input_shape)

    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        vector_features = inputs

        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])

        # Expand vector features to 5D
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, coor_dims, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, max_atoms, 1, 1])

        # Combine between atoms
        vector_features_t = tf.transpose(vector_features, perm=[0, 2, 1, 3, 4])
        vector_features = tf.concat([vector_features, vector_features_t], -1)

        # Apply weights
        vector_features = tf.reshape(vector_features, [-1, atom_feat * 2])
        # vector_features = tf.matmul(vector_features, self.w_vv) + self.b_vv
        if self.bias_initializer is None:
            vector_features = tf.matmul(vector_features, self.w_vv)
        else:
            vector_features = tf.matmul(vector_features, self.w_vv) + self.b_vv
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])

        # multiply bond feature
        if bond is not None:
            vector_features = tf.linalg.einsum('aijkl,aijl->aijkl', vector_features, bond)

        # masking
        if mask is not None:
            mask = tf.reshape(mask, [-1, max_atoms, 1])
            mask = tf.tile(mask, [1, 1, max_atoms])
            vector_features = tf.linalg.einsum('aijkl,aij->aijkl', vector_features, mask)

        # Activation
        vector_features = self.activation(vector_features)

        return vector_features

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[1], input_shape[-2], self.filters


class GraphVToS(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters

        super(GraphVToS, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphVToS, self).get_config()
        base_config['filters'] = self.filters
        return base_config

    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        self.w_vs = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_vs')

        if self.bias_initializer is not None:
            self.b_vs = self.add_weight(shape=(self.filters,),
                                        name='b_vs',
                                        initializer=self.bias_initializer)

        super(GraphVToS, self).build(input_shape)

    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        vector_features, distances = inputs

        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])

        # Expand vector features to 5D
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, coor_dims, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, max_atoms, 1, 1])

        # Combine between atoms
        vector_features_t = tf.transpose(vector_features, perm=[0, 2, 1, 3, 4])
        vector_features = tf.concat([vector_features, vector_features_t], -1)

        # Apply weights
        vector_features = tf.reshape(vector_features, [-1, atom_feat * 2])
        # vector_features = tf.matmul(vector_features, self.w_vs) + self.b_vs
        if self.bias_initializer is None:
            vector_features = tf.matmul(vector_features, self.w_vs)
        else:
            vector_features = tf.matmul(vector_features, self.w_vs) + self.b_vs
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])

        # multiply bond feature
        if bond is not None:
            vector_features = tf.linalg.einsum('aijkl,aijl->aijkl', vector_features, bond)

        # masking
        if mask is not None:
            mask = tf.reshape(mask, [-1, max_atoms, 1])
            mask = tf.tile(mask, [1, 1, max_atoms])
            vector_features = tf.linalg.einsum('aijkl,aij->aijkl', vector_features, mask)

        # # Calculate r^ = r / |r| and expand it to 5D
        # distances_hat = tf.sqrt(tf.reduce_sum(tf.square(distances), axis=-1, keepdims=True))
        # distances_hat = distances_hat + tf.cast(tf.equal(distances_hat, 0), tf.float32)
        # distances_hat = tf.divide(distances, distances_hat)
        # distances_hat = tf.reshape(distances_hat, [-1, max_atoms, max_atoms, coor_dims, 1])
        # distances_hat = tf.tile(distances_hat, [1, 1, 1, 1, self.filters])

        distances_hat = tf.reshape(distances, [-1, max_atoms, max_atoms, coor_dims, 1])
        distances_hat = tf.tile(distances_hat, [1, 1, 1, 1, self.filters])

        # Projection of v onto r = v (dot) r^
        scalar_features = tf.multiply(vector_features, distances_hat)
        scalar_features = tf.reduce_sum(scalar_features, axis=-2)

        # Activation
        scalar_features = self.activation(scalar_features)

        return scalar_features

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], self.filters


class GraphConvS(Layer):
    def __init__(self,
                 filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.pooling = pooling

        super(GraphConvS, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphConvS, self).get_config()
        base_config['filters'] = self.filters
        base_config['pooling'] = self.pooling
        return base_config

    def build(self, input_shape):
        atom_feat_1 = input_shape[0][-1]
        atom_feat_2 = input_shape[1][-1]
        self.w_conv_scalar = self.add_weight(shape=(atom_feat_1 + atom_feat_2, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_scalar')

        if self.bias_initializer is not None:
            self.b_conv_scalar = self.add_weight(shape=(self.filters,),
                                                 name='b_conv_scalar',
                                                 initializer=self.bias_initializer)
        super(GraphConvS, self).build(input_shape)

    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # scalar_features_1 = (samples, max_atoms, max_atoms, atom_feat)
        # scalar_features_2 = (samples, max_atoms, max_atoms, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        scalar_features_1, scalar_features_2, adjacency = inputs

        # Get parameters
        max_atoms = int(scalar_features_1.shape[1])
        atom_feat_1 = int(scalar_features_1.shape[-1])
        atom_feat_2 = int(scalar_features_2.shape[-1])

        # Concatenate two features
        scalar_features = tf.concat([scalar_features_1, scalar_features_2], axis=-1)

        # Linear combination
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat_1 + atom_feat_2])
        # scalar_features = tf.matmul(scalar_features, self.w_conv_scalar) + self.b_conv_scalar
        if self.bias_initializer is None:
            scalar_features = tf.matmul(scalar_features, self.w_conv_scalar)
        else:
            scalar_features = tf.matmul(scalar_features, self.w_conv_scalar) + self.b_conv_scalar
        
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, self.filters])

        # Adjacency masking
        adjacency = tf.reshape(adjacency, [-1, max_atoms, max_atoms, 1])
        adjacency = tf.tile(adjacency, [1, 1, 1, self.filters])
        scalar_features = tf.multiply(scalar_features, adjacency)

        # Integrate over second atom axis
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=2)
        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=2)
        elif self.pooling == "mean":
            scalar_features = tf.reduce_mean(scalar_features, axis=2)
        elif self.pooling == "all":
            scalar_features = tf.stack([tf.reduce_sum(scalar_features, axis=2),
                                       tf.reduce_max(scalar_features, axis=2),
                                       tf.reduce_mean(scalar_features, axis=2)],
                                       axis=-1)
            scalar_features = tf.reshape(scalar_features, [-1,  max_atoms, self.filters * 3])

        # Activation
        if self.activation == "prelu":
            self.activation = PReLU(input_shape=(max_atoms, self.filters), shared_axes=[1, 2])
        scalar_features = self.activation(scalar_features)

        return scalar_features

    def compute_output_shape(self, input_shape):
        if self.pooling == "all":
            return input_shape[0][0], input_shape[0][1], self.filters * 3
        else:
            return input_shape[0][0], input_shape[0][1], self.filters
        

class GraphConvV(Layer):
    def __init__(self,
                 filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.pooling = pooling

        super(GraphConvV, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(GraphConvV, self).get_config()
        base_config['filters'] = self.filters
        base_config['pooling'] = self.pooling
        return base_config

    def build(self, input_shape):
        atom_feat_1 = input_shape[0][-1]
        atom_feat_2 = input_shape[1][-1]
        self.w_conv_vector = self.add_weight(shape=(atom_feat_1 + atom_feat_2, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_vector')

        if self.bias_initializer is not None:
            self.b_conv_vector = self.add_weight(shape=(self.filters,),
                                                 initializer=self.bias_initializer,
                                                 name='b_conv_vector')
        super(GraphConvV, self).build(input_shape)

    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # vector_features_1 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # vector_features_2 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        vector_features_1, vector_features_2, adjacency = inputs

        # Get parameters
        max_atoms = int(vector_features_1.shape[1])
        atom_feat_1 = int(vector_features_1.shape[-1])
        atom_feat_2 = int(vector_features_2.shape[-1])
        coor_dims = int(vector_features_1.shape[-2])

        # Concatenate two features
        vector_features = tf.concat([vector_features_1, vector_features_2], axis=-1)

        # Linear combination
        vector_features = tf.reshape(vector_features, [-1, atom_feat_1 + atom_feat_2])
        # vector_features = tf.matmul(vector_features, self.w_conv_vector) + self.b_conv_vector
        if self.bias_initializer is None:
            vector_features = tf.matmul(vector_features, self.w_conv_vector)
        else:
            vector_features = tf.matmul(vector_features, self.w_conv_vector) + self.b_conv_vector
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])

        # Adjacency masking
        adjacency = tf.reshape(adjacency, [-1, max_atoms, max_atoms, 1, 1])
        adjacency = tf.tile(adjacency, [1, 1, 1, coor_dims, self.filters])
        vector_features = tf.multiply(vector_features, adjacency)

        # Integrate over second atom axis
        if self.pooling == "sum":
            vector_features = tf.reduce_sum(vector_features, axis=2)
        elif self.pooling == "max":
            vector_features = tf.reduce_max(vector_features, axis=2)
        elif self.pooling == "avg":
            vector_features = tf.reduce_mean(vector_features, axis=2)
        elif self.pooling == "all":
            vector_features = tf.stack([tf.reduce_sum(vector_features, axis=2),
                                       tf.reduce_max(vector_features, axis=2),
                                       tf.reduce_mean(vector_features, axis=2)],
                                       axis=-1)
            vector_features = tf.reshape(vector_features, [-1,  max_atoms, coor_dims, self.filters * 3])

        # Activation
        vector_features = self.activation(vector_features)

        return vector_features

    def compute_output_shape(self, input_shape):
        if self.pooling == "all":
            return input_shape[0][0], input_shape[0][1], input_shape[0][-2], self.filters * 3
        else:
            return input_shape[0][0], input_shape[0][1], input_shape[0][-2], self.filters


class GraphGather(Layer):
    def __init__(self,
                 pooling="sum",
                 system="cartesian",
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
            self.activation_n = activation
        self.pooling = pooling
        self.system = system

        super(GraphGather, self).__init__(**kwargs)

    def build(self, inputs_shape):
        super(GraphGather, self).build(inputs_shape)

    def get_config(self):
        base_config = super(GraphGather, self).get_config()
        base_config['pooling'] = self.pooling
        base_config['system'] = self.system
        return base_config

    def call(self, inputs, adjms=None, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        scalar_features, vector_features = inputs

        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])

        # Integrate over atom axis
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=1)
            vector_features = tf.reduce_sum(vector_features, axis=1)

        elif self.pooling == "avg":
            scalar_features = tf.reduce_mean(scalar_features, axis=1)
            vector_features = tf.reduce_mean(vector_features, axis=1)

        elif self.pooling == "avg_adv_1":
            #mask = tf.reduce_max(adjms, axis=-1)
            #mask = tf.where(mask > tf.zeros_like(mask), tf.ones_like(mask), tf.zeros_like(mask))  # (batch, num_atoms)

            mask_s = tf.reshape(mask, [-1, max_atoms, 1])
            mask_s = tf.tile(mask_s, [1, 1, atom_feat])  # (batch, max_atoms, atom_feat)

            mask_v = tf.reshape(mask, [-1, max_atoms, 1, 1])
            mask_v = tf.tile(mask_v, [1, 1, coor_dims, atom_feat])  # (batch, max_atoms, 3, atom_feat)

            scalar_features = tf.ragged.boolean_mask(scalar_features, mask_s)  # (batch, ?, ?)
            vector_features = tf.ragged.boolean_mask(vector_features, mask_v)  # (batch, ?, ?, ?)

            scalar_features = tf.reduce_mean(scalar_features, axis=1)
            #scalar_features = tf.cast(scalar_features, tf.float32)
            vector_features = tf.reduce_mean(vector_features, axis=1)

        elif self.pooling == "avg_adv_2":
            # mask = tf.reduce_max(adjms, axis=-1)
            # mask = tf.where(mask > tf.zeros_like(mask), tf.ones_like(mask), tf.zeros_like(mask))  # (batch, num_atoms)
            num_atoms = tf.reduce_sum(mask, axis=1)  # (batch,)
    
            num_atoms_s = tf.reshape(num_atoms, [-1, 1])
            num_atoms_s = tf.tile(num_atoms_s, [1, atom_feat])  # (batch, atom_feat)
    
            num_atoms_v = tf.reshape(num_atoms, [-1, 1, 1])
            num_atoms_v = tf.tile(num_atoms_v, [1, coor_dims, atom_feat])  # (batch, 3, atom_feat)

            # scalar_features = tf.linalg.einsum('aij,ai->aij', scalar_features, mask)
            scalar_features = tf.reduce_sum(scalar_features, axis=1)  # (batch, atom_feat)
            scalar_features = tf.truediv(scalar_features, num_atoms_s)  # (batch, atom_feat)

            # vector_features = tf.linalg.einsum('aijk,ai->aijk', vector_features, mask)
            vector_features = tf.reduce_sum(vector_features, axis=1)  # (batch, 3, atom_feat)
            vector_features = tf.truediv(vector_features, num_atoms_v)  # (batch, 3, atom_feat)

        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=1)

            vector_features = tf.transpose(vector_features, perm=[0, 2, 3, 1])
            size = tf.sqrt(tf.reduce_sum(tf.square(vector_features), axis=1))
            idx = tf.reshape(tf.argmax(size, axis=-1, output_type=tf.int32), [-1, 1, atom_feat, 1])
            idx = tf.tile(idx, [1, coor_dims, 1, 1])
            vector_features = tf.reshape(tf.batch_gather(vector_features, idx), [-1, coor_dims, atom_feat])
            #vector_features = tf.reshape(tf.gather(vector_features, idx, batch_dims=-1), [-1, coor_dims, atom_feat])
            
        elif self.pooling == "min":
            scalar_features = tf.reduce_min(scalar_features, axis=1)

            vector_features = tf.transpose(vector_features, perm=[0, 2, 3, 1])
            size = tf.sqrt(tf.reduce_sum(tf.square(vector_features), axis=1))
            idx = tf.reshape(tf.argmin(size, axis=-1, output_type=tf.int32), [-1, 1, atom_feat, 1])
            idx = tf.tile(idx, [1, coor_dims, 1, 1])
            vector_features = tf.reshape(tf.batch_gather(vector_features, idx), [-1, coor_dims, atom_feat])
            #vector_features = tf.reshape(tf.gather(vector_features, idx, batch_dims=-1), [-1, coor_dims, atom_feat])
            
        elif self.pooling == "max_adv":
            scalar_features = tf.reduce_max(scalar_features, axis=1)

            vector_features = tf.transpose(vector_features, perm=[0, 2, 3, 1])
            size = tf.sqrt(tf.reduce_sum(tf.square(vector_features), axis=1))
            idx = tf.reshape(tf.argmax(size, axis=-1, output_type=tf.int32), [-1, 1, atom_feat, 1])
            idx = tf.tile(idx, [1, coor_dims, 1, 1])
            vector_features = tf.reshape(tf.gather(vector_features, idx, batch_dims=-1), [-1, coor_dims, atom_feat])

        # Activation
        scalar_features = self.activation(scalar_features)
        vector_features = self.activation(vector_features)

        if self.system == "spherical":
            x, y, z = tf.unstack(vector_features, axis=1)
            r = tf.sqrt(tf.square(x) + tf.square(y) + tf.square(z))
            t = tf.acos(tf.divide(z, r + tf.cast(tf.equal(r, 0), dtype=float)))
            p = tf.atan(tf.divide(y, x + tf.cast(tf.equal(x, 0), dtype=float)))
            vector_features = tf.stack([r, t, p], axis=1)

        return [scalar_features, vector_features]

    def compute_output_shape(self, inputs_shape):
        #return [(inputs_shape[0][0], inputs_shape[0][2]), (inputs_shape[1][0], inputs_shape[1][2], inputs_shape[1][3])]
        if self.pooling in ["maxmin", "maxavg", "maxsum"]:  # multipooling
            return [(inputs_shape[0][0], 2 * inputs_shape[0][2]),
                    (inputs_shape[1][0], inputs_shape[1][2], 2 * inputs_shape[1][3])]

        else:  # one pooling
            return [(inputs_shape[0][0], inputs_shape[0][2]), (inputs_shape[1][0], inputs_shape[1][2], inputs_shape[1][3])]


class GraphGatherS(Layer):
    def __init__(self,
                 pooling="sum",
                 system="cartesian",
                 activation=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.pooling = pooling
        self.system = system

        super(GraphGatherS, self).__init__(**kwargs)

    def build(self, inputs_shape):
        super(GraphGatherS, self).build(inputs_shape)

    def get_config(self):
        base_config = super(GraphGatherS, self).get_config()
        base_config['pooling'] = self.pooling
        base_config['system'] = self.system
        return base_config

    def call(self, inputs, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        scalar_features = inputs

        # Integrate over atom axis
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=1)

        elif self.pooling == "avg":
            scalar_features = tf.reduce_mean(scalar_features, axis=1)

        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=1)

        # Activation
        scalar_features = self.activation(scalar_features)

        return scalar_features

    def compute_output_shape(self, inputs_shape):
        return (inputs_shape[0][0], inputs_shape[0][2])


class Set2Set(Layer):
    def __init__(self,
                 output_dim,
                 step=3,
                 activation_lstm='tanh',
                 activation_recurrent='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.step = step
        self.activation_lstm = activations.get(activation_lstm)
        self.activation_recurrent = activations.get(activation_recurrent)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)

        self.w_linear, self.b_linear, self.w_recurrent = None, None, None
        self.b_recurrent_a, self.b_recurrent_b, self.b_recurrent_c = None, None, None

        super(Set2Set, self).__init__(**kwargs)

    def build(self, inputs_shape):

        self.w_recurrent = self.add_weight(name='s2s_w_recurrent',
                                           shape=(self.output_dim * 2, self.output_dim * 4),
                                           initializer=self.recurrent_initializer)
        self.b_recurrent_a = self.add_weight(name='s2s_b_recurrent_a',
                                             shape=(self.output_dim * 1,),
                                             initializer=initializers.Zeros())
        self.b_recurrent_b = self.add_weight(name='s2s_b_recurrent_b',
                                             shape=(self.output_dim * 1,),
                                             initializer=initializers.Ones())
        self.b_recurrent_c = self.add_weight(name='s2s_b_recurrent_c',
                                             shape=(self.output_dim * 2,),
                                             initializer=initializers.Zeros())
        super(Set2Set, self).build(inputs_shape)

    def get_config(self):
        config = {
            'pooling': "s2s",
            'output_dim': self.output_dim,
            'step': self.step,
            'activation_lstm': activations.serialize(self.activation_lstm),
            'activation_recurrent': activations.serialize(self.activation_recurrent),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer)}
        base_config = super(Set2Set, self).get_config()
        return {**base_config, **config}

    def call(self, inputs, mask=None, **kwargs):
        # Import graph scalar tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)

        features, adjacency = inputs

        # Get parameters
        num_features = int(features.shape[-1])

        # Linear combination
        #features = tf.matmul(features, self.w_linear) + self.b_linear

        # Set2Set embedding
        c = features
        q_star = tf.reduce_sum(tf.zeros_like(features), axis=1, keepdims=True)  # (batch, 1, num_features)
        c = tf.zeros_like(q_star)  # (batch, 1, num_features)
        q_star = tf.concat([q_star, q_star], -1)  # (batch, 1, 2*num_features)
        
        for i in range(self.step):
            q, c = self._lstm(q_star, c)  # (batch, 1, outdims), (batch, 1, outdims)
            e = tf.linalg.einsum('aij,akj->aik', features, q)  # (batch, num_atoms, 1)
            a = tf.nn.softmax(e, axis=1)  # (batch, num_atoms, 1)

            a = tf.tile(a, [1, 1, num_features])  # (batch, num_atoms, 1*num_features)
            r = tf.reduce_sum(tf.multiply(a, features), axis=1, keepdims=True)  # (batch, 1, 1*num_features)
            q_star = tf.concat([q, r], -1)  # (batch, 1, 2*num_features)

        return tf.reshape(q_star, [-1, self.output_dim * 2])  # (batch, 2*num_features)

    def _lstm(self, h, c):
        z = tf.matmul(h, self.w_recurrent) + tf.concat([self.b_recurrent_a, self.b_recurrent_b, self.b_recurrent_c], -1)
        i = self.activation_recurrent(z[:, :, :self.output_dim])
        f = self.activation_recurrent(z[:, :, self.output_dim:self.output_dim * 2])
        o = self.activation_recurrent(z[:, :, self.output_dim * 2:self.output_dim * 3])
        c_out = f * c + i * self.activation_lstm(z[:, :, self.output_dim * 3:])
        h_out = o * self.activation_lstm(c_out)
        return h_out, c_out

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, inputs_shape):
        return inputs_shape[0], 2 * self.output_dim


class Set2SetS(Layer):
    def __init__(self,
                 output_dim,
                 step=3,
                 activation_lstm='tanh',
                 activation_recurrent='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.step = step
        self.activation_lstm = activations.get(activation_lstm)
        self.activation_recurrent = activations.get(activation_recurrent)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)

        self.w_linear, self.b_linear, self.w_recurrent = None, None, None
        self.b_recurrent_a, self.b_recurrent_b, self.b_recurrent_c = None, None, None

        super(Set2SetS, self).__init__(**kwargs)

    def build(self, inputs_shape):
        self.w_recurrent = self.add_weight(name='s2s_w_recurrent',
                                           shape=(self.output_dim * 2, self.output_dim * 4),
                                           initializer=self.recurrent_initializer)
        self.b_recurrent_a = self.add_weight(name='s2s_b_recurrent_a',
                                             shape=(self.output_dim * 1,),
                                             initializer=initializers.Zeros())
        self.b_recurrent_b = self.add_weight(name='s2s_b_recurrent_b',
                                             shape=(self.output_dim * 1,),
                                             initializer=initializers.Ones())
        self.b_recurrent_c = self.add_weight(name='s2s_b_recurrent_c',
                                             shape=(self.output_dim * 2,),
                                             initializer=initializers.Zeros())

        super(Set2SetS, self).build(inputs_shape)

    def get_config(self):
        config = {
            'pooling': "s2s",
            'output_dim': self.output_dim,
            'step': self.step,
            'activation_lstm': activations.serialize(self.activation_lstm),
            'activation_recurrent': activations.serialize(self.activation_recurrent),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer)}
        base_config = super(Set2SetS, self).get_config()
        return {**base_config, **config}

    def call(self, inputs, mask=None, **kwargs):
        # Import graph scalar tensors
        features = inputs  # scalar_features = (samples, max_atoms, atom_feat)

        # Get parameters
        num_features = int(features.shape[-1])

        # Linear combination
        #features = tf.matmul(features, self.w_linear) + self.b_linear

        # Set2Set embedding
        q_star = tf.reduce_sum(tf.zeros_like(features), axis=1, keepdims=True)  # (batch, 1, num_features)
        c = tf.zeros_like(q_star)  # (batch, 1, num_features)
        q_star = tf.concat([q_star, q_star], -1)  # (batch, 1, 2*num_features)

        for i in range(self.step):
            q, c = self._lstm(q_star, c)  # (batch, 1, outdims), (batch, 1, outdims)
            e = tf.linalg.einsum('aij,akj->aik', features, q)  # (batch, num_atoms, 1)
            a = tf.nn.softmax(e, axis=1)  # (batch, num_atoms, 1)

            a = tf.tile(a, [1, 1, num_features])  # (batch, num_atoms, 1*num_features)
            r = tf.reduce_sum(tf.multiply(a, features), axis=1, keepdims=True)  # (batch, 1, 1*num_features)
            q_star = tf.concat([q, r], -1)  # (batch, 1, 2*num_features)

        return tf.reshape(q_star, [-1, self.output_dim * 2])  # (batch, 2*num_features)

    def _lstm(self, h, c):
        z = tf.matmul(h, self.w_recurrent) + tf.concat([self.b_recurrent_a, self.b_recurrent_b, self.b_recurrent_c], -1)
        i = self.activation_recurrent(z[:, :, :self.output_dim])
        f = self.activation_recurrent(z[:, :, self.output_dim:self.output_dim * 2])
        o = self.activation_recurrent(z[:, :, self.output_dim * 2:self.output_dim * 3])
        c_out = f * c + i * self.activation_lstm(z[:, :, self.output_dim * 3:])
        h_out = o * self.activation_lstm(c_out)
        return h_out, c_out

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, inputs_shape):
        return inputs_shape[0], 2 * self.output_dim


class Set2SetV(Layer):
    def __init__(self,
                 output_dim,
                 step=3,
                 activation_lstm='tanh',
                 activation_recurrent='hard_sigmoid',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.step = step
        self.activation_lstm = activations.get(activation_lstm)
        self.activation_recurrent = activations.get(activation_recurrent)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)

        self.w_linear, self.b_linear, self.w_recurrent = None, None, None
        self.b_recurrent_a, self.b_recurrent_b, self.b_recurrent_c = None, None, None

        super(Set2SetV, self).__init__(**kwargs)

    def build(self, inputs_shape):
        self.w_recurrent = self.add_weight(name='s2s_w_recurrent',
                                           shape=(self.output_dim * 2, self.output_dim * 4),
                                           initializer=self.recurrent_initializer)
        self.b_recurrent_a = self.add_weight(name='s2s_b_recurrent_a',
                                             shape=(self.output_dim * 1,),
                                             initializer=initializers.Zeros())
        self.b_recurrent_b = self.add_weight(name='s2s_b_recurrent_b',
                                             shape=(self.output_dim * 1,),
                                             initializer=initializers.Ones())
        self.b_recurrent_c = self.add_weight(name='s2s_b_recurrent_c',
                                             shape=(self.output_dim * 2,),
                                             initializer=initializers.Zeros())

        super(Set2SetV, self).build(inputs_shape)

    def get_config(self):
        config = {
            'pooling': "s2s",
            'output_dim': self.output_dim,
            'step': self.step,
            'activation_lstm': activations.serialize(self.activation_lstm),
            'activation_recurrent': activations.serialize(self.activation_recurrent),
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'recurrent_initializer': initializers.serialize(self.recurrent_initializer)}
        base_config = super(Set2SetV, self).get_config()
        return {**base_config, **config}

    def call(self, inputs, mask=None, **kwargs):
        # Import graph scalar tensors
        features = inputs  # vector_features = (samples, max_atoms, coor_dims, atom_feat)

        # Get parameters
        num_features = int(features.shape[-1])

        # Linear combination

        # Set2Set embedding
        q_star = tf.reduce_sum(tf.zeros_like(features), axis=1, keepdims=True)  # (batch, 1, 3, num_features)
        c = tf.zeros_like(q_star)  # (batch, 1, 3, num_features)
        q_star = tf.concat([q_star, q_star], -1)  # (batch, 1, 3, 2*num_features)

        for i in range(self.step):
            q, c = self._lstm(q_star, c)  # (batch, 1, outdims), (batch, 1, 3, outdims)
            e = tf.linalg.einsum('aijk,aljk->ailj', features, q)  # (batch, num_atoms, 1, 3)
            e = tf.transpose(e, perm=[0, 1, 3, 2])
            a = tf.nn.softmax(e, axis=1)  # (batch, num_atoms, 1, 3)

            a = tf.tile(a, [1, 1, 1, num_features])  # (batch, num_atoms, 3, 1*num_features)
            r = tf.reduce_sum(tf.multiply(a, features), axis=1, keepdims=True)  # (batch, 1, 3, 1*num_features)
            q_star = tf.concat([q, r], -1)  # (batch, 1, 3, 2*num_features)

        return tf.reshape(q_star, [-1, 3, self.output_dim * 2])  # (batch, 3, 2*num_features)

    def _lstm(self, h, c):
        z = tf.matmul(h, self.w_recurrent) + tf.concat([self.b_recurrent_a, self.b_recurrent_b, self.b_recurrent_c], -1)
        i = self.activation_recurrent(z[:, :, :, :self.output_dim])
        f = self.activation_recurrent(z[:, :, :, self.output_dim:self.output_dim * 2])
        o = self.activation_recurrent(z[:, :, :, self.output_dim * 2:self.output_dim * 3])

        c_out = f * c + i * self.activation_lstm(z[:, :, :, self.output_dim * 3:])
        h_out = o * self.activation_lstm(c_out)
        return h_out, c_out

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, inputs_shape):
        return inputs_shape[0], 3, 2 * self.output_dim


# for edge feature conv
class GraphEmbed_edge(Layer):
    def __init__(self,
                 filters=0,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.filters = filters
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        super(GraphEmbed_edge, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GraphEmbed_edge, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # init_feats = (samples, max_atoms, atom_feat)
        # bonds = (samples, max_atoms, max_atoms, bond_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        init_feats, distances, bonds = inputs
    
        # Get parameters
        max_atoms = int(init_feats.shape[1])
        atom_feat = int(init_feats.shape[-1])
        coor_dims = int(distances.shape[-1])
    
        # Generate vector features filled with zeros
        vector_features = tf.zeros_like(init_feats)
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, atom_feat])
        vector_features = tf.tile(vector_features,
                                  [1, 1, coor_dims, 1])  # (samples, max_atoms, coor_dims, atom_feat)
    
        return [init_feats, vector_features, bonds]

    def compute_output_shape(self, input_shape):
        return [input_shape[0], (input_shape[0][0], input_shape[0][1], input_shape[1][-1], input_shape[0][-1]),
                input_shape[-1]]


class GraphSToS_edge(Layer):
    def __init__(self,
                 filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        #self.bond_filters = bond_filters
        self.pooling = pooling
        
        super(GraphSToS_edge, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(GraphSToS_edge, self).get_config()
        base_config['filters'] = self.filters
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        bond_feat = input_shape[1][-1]
        
        self.w_ss = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_ss')
        
        self.w_edge_ss = self.add_weight(shape=(bond_feat, self.filters),
                                               initializer=self.kernel_initializer,
                                               regularizer=self.kernel_regularizer,
                                               name='w_edge_ss')

        self.w_mp_ss = self.add_weight(shape=(self.filters, self.filters),
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           name='w_mp_ss')
        
        if self.bias_initializer is not None:
            self.b_ss = self.add_weight(shape=(self.filters,),
                                        name='b_ss',
                                        initializer=self.bias_initializer)
            self.b_edge_ss = self.add_weight(shape=(self.filters,),
                                             name='b_edge_ss',
                                             initializer=self.bias_initializer)
            self.b_mp_ss = self.add_weight(shape=(self.filters,),
                                           initializer=self.bias_initializer,
                                           name='b_mp_ss')

        super(GraphSToS_edge, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # bonds = (samples, max_atoms, max_atoms, bond_feat)
        scalar_features, bonds = inputs
        # _scalar_features = scalar_features
        
        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])
        
        # Expand scalar features to 4D
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
        scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])  # (samples, max_atoms, max_atoms, atom_feat)
        
        # Combine between atoms
        scalar_features_t = tf.transpose(scalar_features, perm=[0, 2, 1, 3])
        scalar_features = tf.concat([scalar_features, scalar_features_t], -1)
        
        # Linear combination
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat * 2])
        # scalar_features = tf.matmul(scalar_features, self.w_ss) + self.b_ss
        if self.bias_initializer is None:
            scalar_features = tf.matmul(scalar_features, self.w_ss)
        else:
            scalar_features = tf.matmul(scalar_features, self.w_ss) + self.b_ss
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, self.filters])
        
        if bonds is not None:
            # Linear combination of bond features
            # bonds = tf.matmul(bonds, self.w_edge_ss) + self.b_edge_ss
            if self.bias_initializer is None:
                bonds = tf.matmul(bonds, self.w_edge_ss)
            else:
                bonds = tf.matmul(bonds, self.w_edge_ss) + self.b_edge_ss
            # multiply bond feature
            scalar_features = tf.linalg.einsum('aijk,aijk->aijk', scalar_features, bonds)
            # scalar_features = tf.matmul(scalar_features, self.w_mp_ss) + self.b_mp_ss
            if self.bias_initializer is None:
                scalar_features = tf.matmul(scalar_features, self.w_mp_ss)
            else:
                scalar_features = tf.matmul(scalar_features, self.w_mp_ss) + self.b_mp_ss

        # masking
        if mask is not None:
            mask = tf.reshape(mask, [-1, max_atoms, 1])
            mask = tf.tile(mask, [1, 1, max_atoms])
            scalar_features = tf.linalg.einsum('aijk,aij->aijk', scalar_features, mask)
        
        # Activation
        scalar_features = self.activation(scalar_features)

        return scalar_features
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], self.filters


class GraphSToV_edge(Layer):
    def __init__(self,
                 filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.pooling = pooling
        
        super(GraphSToV_edge, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(GraphSToV_edge, self).get_config()
        base_config['filters'] = self.filters
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        bond_feat = input_shape[2][-1]
        self.w_sv = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_sv')
        
        self.w_edge_sv = self.add_weight(shape=(bond_feat, self.filters),
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         name='w_edge_sv')
        
        self.w_mp_sv = self.add_weight(shape=(self.filters, self.filters),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       name='w_mp_sv')

        if self.bias_initializer is not None:
            self.b_sv = self.add_weight(shape=(self.filters,),
                                        name='b_sv',
                                        initializer=self.bias_initializer)
            self.b_edge_sv = self.add_weight(shape=(self.filters,),
                                             name='b_edge_sv',
                                             initializer=self.bias_initializer)
            self.b_mp_sv = self.add_weight(shape=(self.filters,),
                                           initializer=self.bias_initializer,
                                           name='b_mp_sv')
            
        super(GraphSToV_edge, self).build(input_shape)
    
    def call(self, inputs, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        scalar_features, distances, bonds = inputs
        # _scalar_features = scalar_features

        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])
        coor_dims = int(distances.shape[-1])
        
        # Expand scalar features to 4D
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
        scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])
        
        # Combine between atoms
        scalar_features_t = tf.transpose(scalar_features, perm=[0, 2, 1, 3])
        scalar_features = tf.concat([scalar_features, scalar_features_t], -1)
        
        # Apply weights
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat * 2])
        # scalar_features = tf.matmul(scalar_features, self.w_sv) + self.b_sv
        if self.bias_initializer is None:
            scalar_features = tf.matmul(scalar_features, self.w_sv)
        else:
            scalar_features = tf.matmul(scalar_features, self.w_sv) + self.b_sv
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, max_atoms, 1, self.filters])
        scalar_features = tf.tile(scalar_features, [1, 1, 1, coor_dims, 1])
        
        if bonds is not None:
            # Linear combination of bond features
            # bonds = tf.matmul(bonds, self.w_edge_sv) + self.b_edge_sv
            if self.bias_initializer is None:
                bonds = tf.matmul(bonds, self.w_edge_sv)
            else:
                bonds = tf.matmul(bonds, self.w_edge_sv) + self.b_edge_sv
            # multiply bond feature
            scalar_features = tf.linalg.einsum('aijkl,aijl->aijkl', scalar_features, bonds)
            # scalar_features = tf.matmul(scalar_features, self.w_mp_sv) + self.b_mp_sv
            if self.bias_initializer is None:
                scalar_features = tf.matmul(scalar_features, self.w_mp_sv)
            else:
                scalar_features = tf.matmul(scalar_features, self.w_mp_sv) + self.b_mp_sv
            
        # masking
        if mask is not None:
            mask = tf.reshape(mask, [-1, max_atoms, 1])
            mask = tf.tile(mask, [1, 1, max_atoms])
            scalar_features = tf.linalg.einsum('aijkl,aij->aijkl', scalar_features, mask)
        
        # Expand distances to 5D
        distances = tf.reshape(distances, [-1, max_atoms, max_atoms, coor_dims, 1])
        distances = tf.tile(distances, [1, 1, 1, 1, self.filters])
        
        # Tensor product
        vector_features = tf.multiply(scalar_features, distances)
        
        # Activation
        vector_features = self.activation(vector_features)

        '''# Integrate over second atom axis
        if self.pooling == "sum":
            vector_features = tf.reduce_sum(vector_features, axis=2)
        elif self.pooling == "max":
            vector_features = tf.reduce_max(vector_features, axis=2)
        elif self.pooling == "mean":
            vector_features = tf.reduce_mean(vector_features, axis=2)'''

        return vector_features
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], input_shape[1][-1], self.filters


class GraphVToV_edge(Layer):
    def __init__(self,
                 filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.pooling = pooling
        
        super(GraphVToV_edge, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(GraphVToV_edge, self).get_config()
        base_config['filters'] = self.filters
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        bond_feat = input_shape[1][-1]

        self.w_vv = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_vv')
        
        
        self.w_edge_vv = self.add_weight(shape=(bond_feat, self.filters),
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         name='w_edge_vv')
        
        self.w_mp_vv = self.add_weight(shape=(self.filters, self.filters),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       name='w_mp_vv')
        
        if self.bias_initializer is not None:
            self.b_vv = self.add_weight(shape=(self.filters,),
                                        name='b_vv',
                                        initializer=self.bias_initializer)
            self.b_edge_vv = self.add_weight(shape=(self.filters,),
                                             name='b_edge_vv',
                                             initializer=self.bias_initializer)
            self.b_mp_vv = self.add_weight(shape=(self.filters,),
                                           initializer=self.bias_initializer,
                                           name='b_mp_vv')
            
        super(GraphVToV_edge, self).build(input_shape)
    
    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        vector_features, bonds = inputs
        
        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])
        
        # Expand vector features to 5D
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, coor_dims, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, max_atoms, 1, 1])
        
        # Combine between atoms
        vector_features_t = tf.transpose(vector_features, perm=[0, 2, 1, 3, 4])
        vector_features = tf.concat([vector_features, vector_features_t], -1)
        
        # Apply weights
        vector_features = tf.reshape(vector_features, [-1, atom_feat * 2])
        # vector_features = tf.matmul(vector_features, self.w_vv) + self.b_vv
        if self.bias_initializer is None:
            vector_features = tf.matmul(vector_features, self.w_vv)
        else:
            vector_features = tf.matmul(vector_features, self.w_vv) + self.b_vv
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])
        
        # multiply bond feature
        if bonds is not None:
            # Linear combination of bond features
            # bonds = tf.matmul(bonds, self.w_edge_vv) + self.b_edge_vv
            if self.bias_initializer is None:
                bonds = tf.matmul(bonds, self.w_edge_vv)
            else:
                bonds = tf.matmul(bonds, self.w_edge_vv) + self.b_edge_vv
            # multiply bond feature
            vector_features = tf.linalg.einsum('aijkl,aijl->aijkl', vector_features, bonds)
            # vector_features = tf.matmul(vector_features, self.w_mp_vv) + self.b_mp_vv
            if self.bias_initializer is None:
                vector_features = tf.matmul(vector_features, self.w_mp_vv)
            else:
                vector_features = tf.matmul(vector_features, self.w_mp_vv) + self.b_mp_vv

        # masking
        if mask is not None:
            mask = tf.reshape(mask, [-1, max_atoms, 1])
            mask = tf.tile(mask, [1, 1, max_atoms])
            vector_features = tf.linalg.einsum('aijkl,aij->aijkl', vector_features, mask)
        
        # Activation
        vector_features = self.activation(vector_features)

        '''# Integrate over second atom axis
        if self.pooling == "sum":
            vector_features = tf.reduce_sum(vector_features, axis=2)
        elif self.pooling == "max":
            vector_features = tf.reduce_max(vector_features, axis=2)
        elif self.pooling == "mean":
            vector_features = tf.reduce_mean(vector_features, axis=2)'''

        return vector_features
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], input_shape[0][-2], self.filters


class GraphVToS_edge(Layer):
    def __init__(self,
                 filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.pooling = pooling
        
        super(GraphVToS_edge, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(GraphVToS_edge, self).get_config()
        base_config['filters'] = self.filters
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        bond_feat = input_shape[2][-1]

        self.w_vs = self.add_weight(shape=(atom_feat * 2, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_vs')
        
        self.w_edge_vs = self.add_weight(shape=(bond_feat, self.filters),
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         name='w_edge_vs')
        
        self.w_mp_vs = self.add_weight(shape=(self.filters, self.filters),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       name='w_mp_vs')

        if self.bias_initializer is not None:
            self.b_vs = self.add_weight(shape=(self.filters,),
                                        name='b_vs',
                                        initializer=self.bias_initializer)
            self.b_edge_vs = self.add_weight(shape=(self.filters,),
                                             name='b_edge_vs',
                                             initializer=self.bias_initializer)
            self.b_mp_vs = self.add_weight(shape=(self.filters,),
                                           initializer=self.bias_initializer,
                                           name='b_mp_vs')
            
        super(GraphVToS_edge, self).build(input_shape)
    
    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # vector_features = (samples, max_atoms, coor_dims, atom_feat)
        # distances = (samples, max_atoms, max_atoms, coor_dims)
        vector_features, distances, bonds = inputs
        
        # Get parameters
        max_atoms = int(vector_features.shape[1])
        atom_feat = int(vector_features.shape[-1])
        coor_dims = int(vector_features.shape[-2])
        
        # Expand vector features to 5D
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 1, coor_dims, atom_feat])
        vector_features = tf.tile(vector_features, [1, 1, max_atoms, 1, 1])
        
        # Combine between atoms
        vector_features_t = tf.transpose(vector_features, perm=[0, 2, 1, 3, 4])
        vector_features = tf.concat([vector_features, vector_features_t], -1)
        
        # Apply weights
        vector_features = tf.reshape(vector_features, [-1, atom_feat * 2])
        # vector_features = tf.matmul(vector_features, self.w_vs) + self.b_vs
        if self.bias_initializer is None:
            vector_features = tf.matmul(vector_features, self.w_vs)
        else:
            vector_features = tf.matmul(vector_features, self.w_vs) + self.b_vs
        vector_features = tf.reshape(vector_features, [-1, max_atoms, max_atoms, coor_dims, self.filters])
        
        # multiply bond feature
        if bonds is not None:
            # Linear combination of bond features
            # bonds = tf.matmul(bonds, self.w_edge_vs) + self.b_edge_vs
            if self.bias_initializer is None:
                bonds = tf.matmul(bonds, self.w_edge_vs)
            else:
                bonds = tf.matmul(bonds, self.w_edge_vs) + self.b_edge_vs
            # multiply bond feature
            vector_features = tf.linalg.einsum('aijkl,aijl->aijkl', vector_features, bonds)
            # vector_features = tf.matmul(vector_features, self.w_mp_vs) + self.b_mp_vs
            if self.bias_initializer is None:
                vector_features = tf.matmul(vector_features, self.w_mp_vs)
            else:
                vector_features = tf.matmul(vector_features, self.w_mp_vs) + self.b_mp_vs
        
        # masking
        if mask is not None:
            mask = tf.reshape(mask, [-1, max_atoms, 1])
            mask = tf.tile(mask, [1, 1, max_atoms])
            vector_features = tf.linalg.einsum('aijkl,aij->aijkl', vector_features, mask)
        
        # # Calculate r^ = r / |r| and expand it to 5D
        # distances_hat = tf.sqrt(tf.reduce_sum(tf.square(distances), axis=-1, keepdims=True))
        # distances_hat = distances_hat + tf.cast(tf.equal(distances_hat, 0), tf.float32)
        # distances_hat = tf.divide(distances, distances_hat)
        # distances_hat = tf.reshape(distances_hat, [-1, max_atoms, max_atoms, coor_dims, 1])
        # distances_hat = tf.tile(distances_hat, [1, 1, 1, 1, self.filters])
        
        distances_hat = tf.reshape(distances, [-1, max_atoms, max_atoms, coor_dims, 1])
        distances_hat = tf.tile(distances_hat, [1, 1, 1, 1, self.filters])
        
        # Projection of v onto r = v (dot) r^
        scalar_features = tf.multiply(vector_features, distances_hat)
        scalar_features = tf.reduce_sum(scalar_features, axis=-2)
        
        # Activation
        scalar_features = self.activation(scalar_features)

        '''# Integrate over second atom axis
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=2)
        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=2)
        elif self.pooling == "mean":
            scalar_features = tf.reduce_mean(scalar_features, axis=2)'''

        return scalar_features
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][1], self.filters


class GraphConv_edge(Layer):
    def __init__(self,
                 filters,
                 bond_filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.bond_filters = bond_filters
        self.pooling = pooling
        
        super(GraphConv_edge, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(GraphConv_edge, self).get_config()
        base_config['filters'] = self.filters
        base_config['bond_filters'] = self.bond_filters
        base_config['pooling'] = self.pooling
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        bond_feat = input_shape[-1][-1]
        self.w_conv_scalar = self.add_weight(shape=(atom_feat, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_scalar')
        self.b_conv_scalar = self.add_weight(shape=(self.filters,),
                                             name='b_conv_scalar',
                                             initializer=self.bias_initializer)
        
        self.w_mp_scalar = self.add_weight(shape=(atom_feat + bond_feat, self.bond_filters),
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           name='w_mp_scalar')
        self.b_mp_scalar = self.add_weight(shape=(self.bond_filters,),
                                           initializer=self.bias_initializer,
                                           name='b_mp_scalar')
        
        self.w_update_scalar = self.add_weight(shape=(self.filters + self.bond_filters, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_update_scalar')
        self.b_update_scalar = self.add_weight(shape=(self.filters,),
                                             name='b_update_scalar',
                                             initializer=self.bias_initializer)
        super(GraphConv_edge, self).build(input_shape)
    
    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # bonds = (samples, max_atoms, max_atoms, bond_feat)
        scalar_features, bonds = inputs
        
        # Get parameters
        max_atoms = int(scalar_features.shape[1])
        atom_feat = int(scalar_features.shape[-1])

        # Message passing for edge
        if len(scalar_features.shape) == 3:
            scalar_features = tf.reshape(scalar_features, [-1, max_atoms, 1, atom_feat])
            scalar_features = tf.tile(scalar_features, [1, 1, max_atoms, 1])  # (samples, max_atoms, max_atoms, atom_feat)

        m = tf.concat([scalar_features, bonds], axis=-1)
        m = tf.matmul(m, self.w_mp_scalar) + self.b_mp_scalar
        m = self.activation(m)
        m = tf.reduce_sum(m, axis=2)

        # Integrate over second atom axis
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=2)
        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=2)
        elif self.pooling == "mean":
            scalar_features = tf.reduce_mean(scalar_features, axis=2)

        # Linear combination of scalar
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat])
        scalar_features = tf.matmul(scalar_features, self.w_conv_scalar) + self.b_conv_scalar
        scalar_features = self.activation(scalar_features)
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, self.filters])
        
        # Update
        scalar_features = tf.concat([scalar_features, m], axis=-1)
        scalar_features = tf.reshape(scalar_features, [-1, self.filters + self.bond_filters])
        scalar_features = tf.matmul(scalar_features, self.w_update_scalar) + self.b_update_scalar
        scalar_features = self.activation(scalar_features)
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, self.filters])

        return scalar_features
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.filters


class GraphConvS_edge(Layer):
    def __init__(self,
                 filters,
                 bond_filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.bond_filters = bond_filters
        self.pooling = pooling
        
        super(GraphConvS_edge, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(GraphConvS_edge, self).get_config()
        base_config['filters'] = self.filters
        base_config['bond_filters'] = self.bond_filters
        base_config['pooling'] = self.pooling
        return base_config
    
    def build(self, input_shape):
        atom_feat_1 = input_shape[0][-1]
        atom_feat_2 = input_shape[1][-1]
        bond_feat = input_shape[2][-1]
        self.w_conv_scalar = self.add_weight(shape=(atom_feat_1 + atom_feat_2, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_scalar')
        self.b_conv_scalar = self.add_weight(shape=(self.filters,),
                                             name='b_conv_scalar',
                                             initializer=self.bias_initializer)
        
        self.w_mp_scalar = self.add_weight(shape=(atom_feat_1 + atom_feat_2 + bond_feat, self.bond_filters),
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           name='w_mp_scalar')
        self.b_mp_scalar = self.add_weight(shape=(self.bond_filters,),
                                           initializer=self.bias_initializer,
                                           name='b_mp_scalar')
        
        self.w_update_scalar = self.add_weight(shape=(self.filters + self.bond_filters, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_update_scalar')
        self.b_update_scalar = self.add_weight(shape=(self.filters,),
                                             name='b_update_scalar',
                                             initializer=self.bias_initializer)
        super(GraphConvS_edge, self).build(input_shape)
    
    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # scalar_features_1 = (samples, max_atoms, max_atoms, atom_feat)
        # scalar_features_2 = (samples, max_atoms, max_atoms, atom_feat)
        # bonds = (samples, max_atoms, max_atoms, bond_feat)
        scalar_features_1, scalar_features_2, bonds = inputs
        
        # Get parameters
        max_atoms = int(scalar_features_1.shape[1])
        atom_feat_1 = int(scalar_features_1.shape[-1])
        atom_feat_2 = int(scalar_features_2.shape[-1])
        
        # Concatenate two features
        scalar_features = tf.concat([scalar_features_1, scalar_features_2], axis=-1)

        # Message passing for edge
        m = tf.concat([scalar_features, bonds], axis=-1)
        m = tf.matmul(m, self.w_mp_scalar) + self.b_mp_scalar
        m = self.activation(m)
        m = tf.reduce_sum(m, axis=2)

        # Integrate over second atom axis
        if self.pooling == "sum":
            scalar_features = tf.reduce_sum(scalar_features, axis=2)
        elif self.pooling == "max":
            scalar_features = tf.reduce_max(scalar_features, axis=2)
        elif self.pooling == "mean":
            scalar_features = tf.reduce_mean(scalar_features, axis=2)

        # Linear combination of scalar
        scalar_features = tf.reshape(scalar_features, [-1, atom_feat_1 + atom_feat_2])
        scalar_features = tf.matmul(scalar_features, self.w_conv_scalar) + self.b_conv_scalar
        scalar_features = self.activation(scalar_features)
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, self.filters])
        
        # Update
        scalar_features = tf.concat([scalar_features, m], axis=-1)
        scalar_features = tf.reshape(scalar_features, [-1, self.filters + self.bond_filters])
        scalar_features = tf.matmul(scalar_features, self.w_update_scalar) + self.b_update_scalar
        scalar_features = self.activation(scalar_features)
        scalar_features = tf.reshape(scalar_features, [-1, max_atoms, self.filters])

        return scalar_features
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.filters


class GraphConvV_edge(Layer):
    def __init__(self,
                 filters,
                 bond_filters,
                 pooling='sum',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.bond_filters = bond_filters
        self.pooling = pooling
        
        super(GraphConvV_edge, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(GraphConvV_edge, self).get_config()
        base_config['filters'] = self.filters
        base_config['bond_filters'] = self.bond_filters
        base_config['pooling'] = self.pooling
        return base_config
    
    def build(self, input_shape):
        atom_feat_1 = input_shape[0][-1]
        atom_feat_2 = input_shape[1][-1]
        bond_feat = input_shape[2][-1]
        self.w_conv_vector = self.add_weight(shape=(atom_feat_1 + atom_feat_2, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_conv_vector')
        
        self.b_conv_vector = self.add_weight(shape=(self.filters,),
                                             initializer=self.bias_initializer,
                                             name='b_conv_vector')

        self.w_mp_vector = self.add_weight(shape=(atom_feat_1 + atom_feat_2 + bond_feat, self.bond_filters),
                                           initializer=self.kernel_initializer,
                                           regularizer=self.kernel_regularizer,
                                           name='w_mp_vector')
        self.b_mp_vector = self.add_weight(shape=(self.bond_filters,),
                                           initializer=self.bias_initializer,
                                           name='b_mp_vector')
        
        self.w_update_vector = self.add_weight(shape=(self.filters + self.bond_filters, self.filters),
                                             initializer=self.kernel_initializer,
                                             regularizer=self.kernel_regularizer,
                                             name='w_update_vector')
        self.b_update_vector = self.add_weight(shape=(self.filters,),
                                             name='b_update_vector',
                                             initializer=self.bias_initializer)
        super(GraphConvV_edge, self).build(input_shape)
    
    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # vector_features_1 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # vector_features_2 = (samples, max_atoms, max_atoms, coor_dims, atom_feat)
        # bonds = (samples, max_atoms, max_atoms, bond_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        vector_features_1, vector_features_2, bonds = inputs

        # Get parameters
        max_atoms = int(vector_features_1.shape[1])
        atom_feat_1 = int(vector_features_1.shape[-1])
        atom_feat_2 = int(vector_features_2.shape[-1])
        bond_feat = int(bonds.shape[-1])
        coor_dims = int(vector_features_1.shape[-2])
        
        # Concatenate two features
        vector_features = tf.concat([vector_features_1, vector_features_2], axis=-1)

        # Message passing for edge
        m = tf.reshape(bonds, [-1, max_atoms, max_atoms, 1, bond_feat])
        m = tf.tile(m, [1, 1, 1, coor_dims, 1])
        
        m = tf.concat([vector_features, m], axis=-1)
        m = tf.matmul(m, self.w_mp_vector) + self.b_mp_vector
        m = self.activation(m)
        m = tf.reduce_sum(m, axis=2)


        # Integrate over second atom axis
        if self.pooling == "sum":
            vector_features = tf.reduce_sum(vector_features, axis=2)
        elif self.pooling == "max":
            vector_features = tf.reduce_max(vector_features, axis=2)
        elif self.pooling == "avg":
            vector_features = tf.reduce_mean(vector_features, axis=2)

        # Linear combination
        vector_features = tf.reshape(vector_features, [-1, atom_feat_1 + atom_feat_2])
        vector_features = tf.matmul(vector_features, self.w_conv_vector) + self.b_conv_vector
        vector_features = self.activation(vector_features)
        vector_features = tf.reshape(vector_features, [-1, max_atoms, coor_dims, self.filters])
        
        # Update
        vector_features = tf.concat([vector_features, m], axis=-1)
        vector_features = tf.reshape(vector_features, [-1, self.filters + self.bond_filters])
        vector_features = tf.matmul(vector_features, self.w_update_vector) + self.b_update_vector
        vector_features = self.activation(vector_features)
        vector_features = tf.reshape(vector_features, [-1, max_atoms, 3, self.filters])
        
        return vector_features
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][-2], self.filters



# baseline
class GraphConv(Layer):
    def __init__(self,
                 filters,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        
        super(GraphConv, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(GraphConv, self).get_config()
        base_config['filters'] = self.filters
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        self.w_conv = self.add_weight(shape=(atom_feat, self.filters),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name='w_conv')
        
        if self.bias_initializer is not None:
            self.b_conv = self.add_weight(shape=(self.filters,),
                                          name='b_conv',
                                          initializer=self.bias_initializer)
        super(GraphConv, self).build(input_shape)
    
    def call(self, inputs, bond=None, mask=None):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        scalar_features, adjacency = inputs
        
        # Matrix multiplication X'=AXW
        scalar_features = tf.linalg.einsum('aij,ajk->aik', adjacency, scalar_features)
        
        if self.bias_initializer is None:
            scalar_features = tf.linalg.einsum('aij,jk->aik', scalar_features, self.w_conv)
        else:
            scalar_features = tf.linalg.einsum('aij,jk->aik', scalar_features, self.w_conv) + self.b_conv
        
        # Activation
        scalar_features = self.activation(scalar_features)
        
        return scalar_features
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.filters


class WeaveLayer(Layer):
    def __init__(self,
                 filters=50,
                 atom_filters=50,
                 pair_filters=50,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation="relu",
                 update_pair=True,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.filters = filters
        self.atom_filters = atom_filters
        self.pair_filters = pair_filters
        self.update_pair = update_pair
        
        super(WeaveLayer, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(WeaveLayer, self).get_config()
        base_config['filters'] = self.filters
        base_config['atom_filters'] = self.atom_filters
        base_config['pair_filters'] = self.pair_filters
        base_config['update_pair'] = self.update_pair
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        pair_feat = input_shape[1][-1]
        
        self.w_aa = self.add_weight(shape=(atom_feat, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_aa')
        self.w_pa = self.add_weight(shape=(pair_feat, self.filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_pa')
        self.w_a = self.add_weight(shape=(self.filters * 2, self.atom_filters),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_a')
        
        if self.bias_initializer is not None:
            self.b_aa = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='b_aa')
            self.b_pa = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='b_pa')
            self.b_a = self.add_weight(shape=(self.atom_filters,),
                                        initializer=self.bias_initializer,
                                        name='b_a')
            
        if self.update_pair:
            self.w_pp = self.add_weight(shape=(pair_feat, self.filters),
                                        initializer=self.kernel_initializer,
                                        regularizer=self.kernel_regularizer,
                                        name='w_pp')
            self.w_ap1 = self.add_weight(shape=(atom_feat*2, self.filters),
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         name='w_ap1')
            self.w_ap2 = self.add_weight(shape=(atom_feat*2, self.filters),
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         name='w_ap2')
            self.w_p = self.add_weight(shape=(self.filters + self.filters, self.pair_filters),
                                       initializer=self.kernel_initializer,
                                       regularizer=self.kernel_regularizer,
                                       name='w_p')

            if self.bias_initializer is not None:
                self.b_pp = self.add_weight(shape=(self.filters,),
                                            initializer=self.bias_initializer,
                                            name='b_pp')
                self.b_ap1 = self.add_weight(shape=(self.filters,),
                                             initializer=self.bias_initializer,
                                             name='b_ap1')
                self.b_ap2 = self.add_weight(shape=(self.filters,),
                                             initializer=self.bias_initializer,
                                             name='b_ap2')
                self.b_p = self.add_weight(shape=(self.pair_filters,),
                                           initializer=self.bias_initializer,
                                           name='b_p')

        super(WeaveLayer, self).build(input_shape)
    
    def call(self, inputs):
        # Import graph tensors
        # a_in = (samples, max_atoms, atom_feat)
        # p_in = (samples, max_atoms, max_atoms, bond_feat)
        a_in, p_in, adjacency = inputs
        
        # Get parameters
        max_atoms = int(a_in.shape[1])
        atom_feat = int(a_in.shape[-1])
        bond_feat = int(p_in.shape[-1])
        
        # Update node features
        if self.bias_initializer is None:
            a_to_a = tf.linalg.einsum('aij,jk->aik', a_in, self.w_aa)
            p_to_a = tf.linalg.einsum('aijk,kl->ail', p_in, self.w_pa)
        else:
            a_to_a = tf.linalg.einsum('aij,jk->aik', a_in, self.w_aa) + self.b_aa
            p_to_a = tf.linalg.einsum('aijk,kl->aijl', p_in, self.w_pa) + self.b_pa
        a_to_a = self.activation(a_to_a)
        p_to_a = self.activation(p_to_a)
        p_to_a = tf.linalg.einsum('aijk,aij->aijk', p_to_a, adjacency)
        p_to_a = tf.linalg.einsum('aijk->aik', p_to_a)
        
        new_a = tf.concat([a_to_a, p_to_a], axis=-1)
        if self.bias_initializer is None:
            new_a = tf.linalg.einsum('aij,jk->aik', new_a, self.w_a)
        else:
            new_a = tf.linalg.einsum('aij,jk->aik', new_a, self.w_a) + self.b_a
        new_a = self.activation(new_a)

        if self.update_pair:
            # Update edge features
            # Expand scalar features to 4D
            a_to_p = tf.reshape(a_in, [-1, max_atoms, 1, atom_feat])
            a_to_p = tf.tile(a_to_p, [1, 1, max_atoms, 1])  # (samples, max_atoms, max_atoms, atom_feat)
            a_to_p = tf.linalg.einsum('aijk,aij->aijk', a_to_p, adjacency)
            a_to_p_t = tf.transpose(a_to_p, perm=[0, 2, 1, 3])  # (samples, max_atoms, max_atoms, atom_feat)
            a_to_p = tf.concat([a_to_p, a_to_p_t], -1)

            if self.bias_initializer is None:
                a_to_p_1 = tf.linalg.einsum('aijk,kl->aijl', a_to_p, self.w_ap1)
                a_to_p_2 = tf.linalg.einsum('aijk,kl->ajil', a_to_p, self.w_ap2)
                p_to_p = tf.linalg.einsum('aijk,kl->aijl', p_in, self.w_pp)
            else:
                a_to_p_1 = tf.linalg.einsum('aijk,kl->aijl', a_to_p, self.w_ap1) + self.b_ap1
                a_to_p_2 = tf.linalg.einsum('aijk,kl->ajil', a_to_p, self.w_ap2) + self.b_ap2
                p_to_p = tf.linalg.einsum('aijk,kl->aijl', p_in, self.w_pp) + self.b_pp
            # a_to_p_1, a_to_p_2 : (b, max_atoms, self.filters)

            new_a_to_p = Add()([a_to_p_1, a_to_p_2])
            new_a_to_p = self.activation(new_a_to_p)
            # a_to_p : (b, max_atoms, max_atoms, self.filters)
            p_to_p = self.activation(p_to_p)

            new_p = tf.concat([new_a_to_p, p_to_p], axis=-1)
            new_p = tf.linalg.einsum('aijk,aij->aijk', new_p, adjacency)
            
            if self.bias_initializer is None:
                new_p = tf.linalg.einsum('aijk,kl->aijl', new_p, self.w_p)
            else:
                new_p = tf.linalg.einsum('aijk,kl->aijl', new_p, self.w_p) + self.b_p
            new_p = self.activation(new_p)
            
            return [new_a, new_p]
        else:
            return [new_a, p_in]
    
    def compute_output_shape(self, input_shape):
        return [(input_shape[0][0], input_shape[0][1], self.atom_filters),
                (input_shape[1][0], input_shape[1][1], input_shape[1][1], self.pair_filters)]


class WeaveGather(Layer):
    def __init__(self,
                 n_feat=128,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 gaussian_expand=True,
                 compress_post_gaussian_expansion=False,
                 activation="tanh",
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

        self.gaussian_expand = gaussian_expand
        self.compress_post_gaussian_expansion = compress_post_gaussian_expansion
        self.n_feat = n_feat

        super(WeaveGather, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(WeaveGather, self).get_config()
        base_config['n_feat'] = self.n_feat
        base_config['gaussian_expand'] = self.gaussian_expand
        base_config['compress_post_gaussian_expansion'] = self.compress_post_gaussian_expansion
        base_config['activation'] = self.activation
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[-1]
        if self.compress_post_gaussian_expansion:
            self.w_compress = self.add_weight(shape=(atom_feat * 11, self.n_feat),
                                              initializer=self.kernel_initializer,
                                              regularizer=self.kernel_regularizer,
                                              name='w_compress')
            
            if self.bias_initializer is not None:
                self.b_compress = self.add_weight(shape=(self.n_feat,),
                                                  initializer=self.bias_initializer,
                                                  name='b_compress')
            
        super(WeaveGather, self).build(input_shape)

    def call(self, inputs, mask=None):
        # Import graph tensors
        # a_in = (samples, max_atoms, atom_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        a_in = inputs

        max_atoms = a_in.shape[1]
        atom_feat = a_in.shape[-1]
        
        if self.gaussian_expand:
            outputs = self.gaussian_histogram(a_in)
            # Integrate over second atom axis
            output_molecules = tf.reduce_sum(outputs, axis=1)  # (b, atom_feat * 11)

            if self.compress_post_gaussian_expansion:
                output_molecules = tf.matmul(output_molecules, self.w_compress) + self.b_compress
                output_molecules = self.activation(output_molecules)

        else:
            # Integrate over second atom axis
            output_molecules = tf.reduce_sum(a_in, axis=1)  # (b, atom_feat * 11)

        return output_molecules

    def compute_output_shape(self, input_shape):
        if self.gaussian_expand and self.compress_post_gaussian_expansion:
            return input_shape[0], input_shape[-1]
        elif self.gaussian_expand:
            return input_shape[0], input_shape[-1] * 11
        elif not self.gaussian_expand:
            return input_shape[0], input_shape[-1]
        
    def gaussian_histogram(self, x):
        """Expands input into a set of gaussian histogram bins.
        Parameters
        ----------
        x: tf.Tensor
          Of shape `(N, n_feat)`
        Examples
        --------
        This method uses 11 bins spanning portions of a Gaussian with zero mean
        and unit standard deviation.
        >>> gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170),
        ...                         (-0.739, 0.134), (-0.468, 0.118),
        ...                         (-0.228, 0.114), (0., 0.114),
        ...                         (0.228, 0.114), (0.468, 0.118),
        ...                         (0.739, 0.134), (1.080, 0.170),
        ...                         (1.645, 0.283)]
        We construct a Gaussian at `gaussian_memberships[i][0]` with standard
        deviation `gaussian_memberships[i][1]`. Each feature in `x` is assigned
        the probability of falling in each Gaussian, and probabilities are
        normalized across the 11 different Gaussians.
        Returns
        -------
        outputs: tf.Tensor
          Of shape `(N, 11*n_feat)`
        """
        import tensorflow_probability as tfp
        gaussian_memberships = [(-1.645, 0.283), (-1.080, 0.170), (-0.739, 0.134),
                                (-0.468, 0.118), (-0.228, 0.114), (0., 0.114),
                                (0.228, 0.114), (0.468, 0.118), (0.739, 0.134),
                                (1.080, 0.170), (1.645, 0.283)]
        dist = [tfp.distributions.Normal(p[0], p[1]) for p in gaussian_memberships]
        dist_max = [dist[i].prob(gaussian_memberships[i][0]) for i in range(11)]
        outputs = [dist[i].prob(x) / dist_max[i] for i in range(11)]
        outputs = tf.stack(outputs, axis=3)
        outputs = outputs / tf.reduce_sum(outputs, axis=3, keepdims=True)
        outputs = tf.reshape(outputs, [-1, x.shape[1], x.shape[2] * 11])
        return outputs


class MessagePassing(Layer):
    def __init__(self,
                 t=5,
                 n_hidden=100,
                 message_fn='enn',
                 update_fn='gru',
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        if activation in ["relu", "tanh", "sigmoid", None]:
            self.activation = activations.get(activation)
        else:
            self.activation = activation
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.t = t
        self.message_fn = message_fn
        self.update_fn = update_fn
        self.n_hidden = n_hidden
        
        super(MessagePassing, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(MessagePassing, self).get_config()
        base_config['t'] = self.t
        base_config['n_hidden'] = self.n_hidden
        base_config['message_fn'] = self.message_fn
        base_config['update_fn'] = self.update_fn
        return base_config
    
    def build(self, input_shape):
        atom_feat = input_shape[0][-1]
        bond_feat = input_shape[1][-1]

        if self.message_fn == 'enn':
            # Default message function: edge network, update function: GRU
            # more options to be implemented
            self.message_fn = EdgeNetwork(self.n_hidden)
        if self.update_fn == 'gru':
            self.update_fn = GatedRecurrentUnit(self.n_hidden)
        self.built = True
        
        super(MessagePassing, self).build(input_shape)
    
    def call(self, inputs):
        # Import graph tensors
        # scalar_features = (samples, max_atoms, atom_feat)
        # bonds = (samples, max_atoms, max_atoms, bond_feat)
        # adjacency = (samples, max_atoms, max_atoms)
        """ Perform T steps of message passing """
        atoms, bonds = inputs

        # Get parameters
        max_atoms = int(atoms.shape[1])
        atom_feat = int(atoms.shape[-1])
        
        if atom_feat < self.n_hidden:
            pad_length = self.n_hidden - atom_feat
            out = tf.pad(atoms, ((0, 0), (0, 0), (0, pad_length)), mode='CONSTANT')
        elif atom_feat > self.n_hidden:
            raise ValueError("Too large initial feature vector")
        else:
            out = atoms

        for i in range(self.t):
            # message = self.message_fn([out, bonds, adjacency])
            message = self.message_fn([out, bonds])
            out = self.update_fn([out, message])
        return out
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.n_hidden


class EdgeNetwork(Layer):
    def __init__(self,
                 n_hidden=100,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 activation=None,
                 **kwargs):
        self.n_hidden = n_hidden
        self.kernel_initializer = initializers.get(kernel_initializer)
        # self.bias_initializer = initializers.get(bias_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.n_hidden = n_hidden
        
        super(EdgeNetwork, self).__init__(**kwargs)
    
    def get_config(self):
        base_config = super(EdgeNetwork, self).get_config()
        base_config['n_hidden'] = self.n_hidden
        return base_config
    
    def build(self, input_shape):
        bond_feat = input_shape[1][-1]
        
        self.w_edge = self.add_weight(shape=(bond_feat, self.n_hidden * self.n_hidden),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer,
                                    name='w_edge')
        if self.bias_initializer is not None:
            self.b_edge = self.add_weight(shape=(self.n_hidden * self.n_hidden,),
                                        initializer=self.bias_initializer,
                                        name='b_edge')
        super(EdgeNetwork, self).build(input_shape)
    
    def call(self, inputs):
        # Import graph tensors
        # atoms = (samples, max_atoms, atom_feat)
        # bonds = (samples, max_atoms, max_atoms, bond_feat)
        # adjacency = (samples, max_atoms, max_atoms)

        atoms, bonds = inputs
        
        # Get parameters
        max_atoms = int(atoms.shape[1])
        
        if self.bias_initializer is None:
            adj_from_bond = tf.matmul(bonds, self.w_edge)
        else:
            adj_from_bond = tf.matmul(bonds, self.w_edge) + self.b_edge
        adj_from_bond = tf.reshape(adj_from_bond, (-1, max_atoms, max_atoms, self.n_hidden, self.n_hidden))
        # (samples, max_atoms, max_atoms, self.n_hidden, n_hidden)
        
        atoms_expand = tf.reshape(atoms, [-1, max_atoms, 1, self.n_hidden])
        atoms_expand = tf.tile(atoms_expand, [1, 1, max_atoms, 1])  # (samples, max_atoms, max_atoms, atom_feat)
        #atoms_expand = tf.linalg.einsum('aijk,aij->aijk', atoms_expand, adjacency)
        atoms_out = tf.linalg.einsum('aijkl,aijk->aijk', adj_from_bond, atoms_expand)  # (samples, max_atoms, max_atoms, n_hidden)
        #atoms_out = tf.linalg.einsum('aijk,aij->aik', atoms_out, adjacency)  # (samples, max_atoms, n_hidden)
        #atoms_out = tf.linalg.einsum('aijk,aij->aijk', atoms_out, adjacency)
        atoms_out = tf.reduce_sum(atoms_out, axis=2)

        return atoms_out
    
    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], self.n_hidden


class GatedRecurrentUnit(Layer):
    """ Submodule for Message Passing """
    
    def __init__(self, n_hidden=100,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 **kwargs):
        super(GatedRecurrentUnit, self).__init__(**kwargs)
        self.n_hidden = n_hidden
        self.kernel_initializer = initializers.get(kernel_initializer)
        if bias_initializer is None:
            self.bias_initializer = None
        else:
            self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
    
    def get_config(self):
        config = super(GatedRecurrentUnit, self).get_config()
        config['n_hidden'] = self.n_hidden
        return config
    
    def build(self, input_shape):
        n_hidden = self.n_hidden

        self.Wz = self.add_weight(shape=(n_hidden, n_hidden),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  name='Wz')
        self.Wr = self.add_weight(shape=(n_hidden, n_hidden),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  name='Wr')
        self.Wh = self.add_weight(shape=(n_hidden, n_hidden),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  name='Wh')
        self.Uz = self.add_weight(shape=(n_hidden, n_hidden),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  name='Uz')
        self.Ur = self.add_weight(shape=(n_hidden, n_hidden),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  name='Ur')
        self.Uh = self.add_weight(shape=(n_hidden, n_hidden),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  name='Uh')
        
        if self.bias_initializer is not None:
            self.bz = self.add_weight(shape=(n_hidden,),
                                      initializer=self.bias_initializer,
                                      name='bz')
            self.br = self.add_weight(shape=(n_hidden,),
                                      initializer=self.bias_initializer,
                                      name='br')
            self.bh = self.add_weight(shape=(n_hidden,),
                                      initializer=self.bias_initializer,
                                      name='bh')

    def call(self, inputs):
        # inputs[0] = out = atom_feat, inputs[1] = message
        z = tf.nn.sigmoid(
            tf.matmul(inputs[1], self.Wz) + tf.matmul(inputs[0], self.Uz) + self.bz)
        r = tf.nn.sigmoid(
            tf.matmul(inputs[1], self.Wr) + tf.matmul(inputs[0], self.Ur) + self.br)
        h = (1 - z) * tf.nn.tanh(
            tf.matmul(inputs[1], self.Wh) + tf.matmul(inputs[0] * r, self.Uh) +
            self.bh) + z * inputs[0]
        return h
