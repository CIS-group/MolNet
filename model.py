# for resnet
from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)
import six
from math import ceil
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers.merge import add
from keras.regularizers import l2
#from keras import backend as K


#from tensorflow.keras.models import Model
from keras.layers import Input, Add, Dense, Flatten, TimeDistributed, Concatenate, Activation, Lambda, LeakyReLU, PReLU, Dropout, Add
from keras.layers import Conv3D, MaxPool3D, AvgPool3D, ZeroPadding3D, BatchNormalization, ReLU, Reshape
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.optimizers import Adam, Adagrad
from keras.regularizers import l2
from keras.models import Model
import numpy as np

import tensorflow as tf

# import modeuls from current directory
from layer import *
from loss import std_mae, std_rmse, std_r2, metric_wsr, metric_r2


def model_molnet(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]

    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    act = hyper["act_s"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]

    units_edge = 128

    if "lrelu" in act:
        # act = tf.nn.leaky_relu
        try:
            rate = float(act.split("lrelu")[-1])
        except:
            rate = 0.2
        act = LeakyReLU(alpha=rate)

    elif act == "prelu":
        # act = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        act = PReLU(shared_axes=[1, 2])

    elif act == "swish":
        # act = swish
        act = tf.nn.swish

    elif act == "mish":
        act = mish

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    d_cutoff = 5.0
    near = Lambda(lambda x: tf.where(tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)) < d_cutoff,
                                     tf.ones_like(tf.reduce_sum(x, axis=-1)),
                                     tf.zeros_like(tf.reduce_sum(x, axis=-1))))(dists)
    near_not_bond = Lambda(
        lambda x: tf.subtract(x[0], tf.where(x[1] > tf.zeros_like(x[1]), tf.ones_like(x[1]), tf.zeros_like(x[1]))))(
        [near, adjms])

    # Normalize adjacency matrix by D^(-1/2) * A_hat * D^(-1/2), Kipf et al. 2016
    near_not_bond = Lambda(lambda x: x + tf.eye(num_atoms))(near_not_bond)
    near_not_bond_sqrt = Lambda(lambda x: tf.pow(tf.reduce_sum(x, axis=1), -0.5))(near_not_bond)

    near_not_bond_sqrt = Lambda(
        lambda x: tf.where(tf.is_inf(x), x, tf.zeros_like(x)))(near_not_bond_sqrt)
    near_not_bond_sqrt = Lambda(lambda x: tf.matrix_diag(x))(near_not_bond_sqrt)
    near_not_bond_norm = Lambda(lambda x: tf.matmul(tf.matmul(x[1], x[0]), x[1]))([near_not_bond, near_not_bond_sqrt])

    sc = Dense(units_conv, activation=act, kernel_regularizer=l2(0.005))(atoms)
    sc, vc = GraphEmbed()([sc, dists])
    sc_near, vc_near = GraphEmbed()([sc, dists])

    for _ in range(num_layers):
        sc_s = GraphSToS(units_conv, activation=act)(sc)
        sc_v = GraphVToS(units_conv, activation=act)([vc, dists])

        vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
        vc_v = GraphVToV(units_conv, activation='tanh')(vc)

        _sc = GraphConvS(units_conv, pooling='sum', activation=act)([sc_s, sc_v, adjms])
        _vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

        sc = Add()([sc, _sc])
        vc = Add()([vc, _vc])

        sc_s_near = GraphSToS(units_conv, activation=act)(sc_near)
        sc_v_near = GraphVToS(units_conv, activation=act)([vc_near, dists])
        _sc_near = GraphConvS(units_conv, pooling='sum', activation=act)([sc_s_near, sc_v_near, near_not_bond_norm])

        vc_s_near = GraphSToV(units_conv, activation='tanh')([sc_near, dists])
        vc_v_near = GraphVToV(units_conv, activation='tanh')(vc_near)
        _vc_near = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s_near, vc_v_near, near_not_bond_norm])

        sc_near = Add()([sc_near, _sc_near])
        vc_near = Add()([vc_near, _vc_near])

    sc = Concatenate(axis=-1)([sc, sc_near])

    if pooling == 's2s':
        sc = Set2SetS(units_dense)(sc)

    elif ',' in pooling:
        sc_pooling = []
        for pool in pooling.split(","):
            if pool == "sum":
                _sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
            elif pool == "avg":
                _sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
            elif pool == "max":
                _sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)
            sc_pooling.append(_sc)
        sc = Concatenate(axis=-1)(sc_pooling)

    elif pooling == "sum":
        sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
    elif pooling == "avg":
        sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
    elif pooling == "max":
        sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)

    if hyper["act_s"] == "prelu":
        _act = LeakyReLU(alpha=0.2)
        sc_out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc)
        out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc_out)
    else:
        sc_out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc)
        out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc_out)

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def model_molnet_woNC(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    act = hyper["act_s"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]

    if "lrelu" in act:
        # act = tf.nn.leaky_relu
        try:
            rate = float(act.split("lrelu")[-1])
        except:
            rate = 0.2
        act = LeakyReLU(alpha=rate)

    elif act == "prelu":
        # act = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        act = PReLU(shared_axes=[1, 2])

    elif act == "swish":
        # act = swish
        act = tf.nn.swish

    elif act == "mish":
        act = mish

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    sc = Dense(units_conv, activation=act, kernel_regularizer=l2(0.005))(atoms)
    sc, vc = GraphEmbed()([sc, dists])

    for _ in range(num_layers):
        sc_s = GraphSToS(units_conv, activation=act)(sc)
        sc_v = GraphVToS(units_conv, activation=act)([vc, dists])

        vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
        vc_v = GraphVToV(units_conv, activation='tanh')(vc)

        _sc = GraphConvS(units_conv, pooling='sum', activation=act)([sc_s, sc_v, adjms])
        _vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

        sc = Add()([sc, _sc])
        vc = Add()([vc, _vc])

    if pooling == 's2s':
        sc = Set2SetS(units_dense)(sc)
        # vc = Set2SetV(units_dense)(vc)

    elif ',' in pooling:
        sc_pooling = []
        for pool in pooling.split(","):
            if pool == "sum":
                _sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
            elif pool == "avg":
                _sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
            elif pool == "max":
                _sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)
            sc_pooling.append(_sc)
        sc = Concatenate(axis=-1)(sc_pooling)

    elif pooling == "sum":
        sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
    elif pooling == "avg":
        sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
    elif pooling == "max":
        sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)

    if hyper["act_s"] == "prelu":
        _act = LeakyReLU(alpha=0.2)
        sc_out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc)
        out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc_out)
    else:
        sc_out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc)
        out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc_out)

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def model_molnet_edge(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    num_bond_features = hyper["num_bond_features"]

    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    act = hyper["act_s"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]

    units_edge = 128

    if "lrelu" in act:
        # act = tf.nn.leaky_relu
        try:
            rate = float(act.split("lrelu")[-1])
        except:
            rate = 0.2
        act = LeakyReLU(alpha=rate)

        # output = tf.keras.layers.Dense(n_units, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        # output = tf.layers.dense(input, n_units, activation=lambda x : tf.nn.leaky_relu(x, alpha=0.01)) in tf2.0

    elif act == "prelu":
        # act = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        act = PReLU(shared_axes=[1, 2])
        # PReLU(alpha_initializer=tf.initializers.constant(0.25))

    elif act == "swish":
        # act = swish
        act = tf.nn.swish

    elif act == "mish":
        act = mish

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    bonds = Input(name='bond_inputs', shape=(num_atoms, num_atoms, num_bond_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    d_cutoff = 5.0
    near = Lambda(lambda x: tf.where(tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)) < d_cutoff,
                                     tf.ones_like(tf.reduce_sum(x, axis=-1)),
                                     tf.zeros_like(tf.reduce_sum(x, axis=-1))))(dists)

    near_not_bond = Lambda(
        lambda x: tf.subtract(x[0], tf.where(x[1] > tf.zeros_like(x[1]), tf.ones_like(x[1]), tf.zeros_like(x[1]))))(
        [near, adjms])
    near_not_bond_sqrt = Lambda(lambda x: tf.pow(tf.reduce_sum(x, axis=1), -0.5))(near_not_bond)

    near_not_bond_sqrt = Lambda(
        lambda x: tf.where(tf.is_inf(x), x, tf.zeros_like(x)))(near_not_bond_sqrt)
    near_not_bond_sqrt = Lambda(lambda x: tf.matrix_diag(x))(near_not_bond_sqrt)
    near_not_bond_norm = Lambda(lambda x: tf.matmul(tf.matmul(x[1], x[0]), x[1]))([near_not_bond, near_not_bond_sqrt])

    sc = Dense(units_conv, activation=LeakyReLU(alpha=0.2), kernel_regularizer=l2(0.005))(atoms)
    sc, vc, bf = GraphEmbed_edge()([sc, dists, bonds])

    sc_near, vc_near, bf = GraphEmbed_edge()([sc, dists, bonds])

    for _ in range(num_layers):
        sc_s = GraphSToS_edge(units_conv, pooling='sum', activation=act)([sc, bf])
        sc_v = GraphVToS_edge(units_conv, pooling='sum', activation=act)([vc, dists, bf])

        vc_s = GraphSToV_edge(units_conv, pooling='sum', activation='tanh')([sc, dists, bf])
        vc_v = GraphVToV_edge(units_conv, pooling='sum', activation='tanh')([vc, bf])

        _sc = GraphConvS(units_conv, pooling='sum', activation=act)([sc_s, sc_v, adjms])
        _vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

        sc = Add()([sc, _sc])
        vc = Add()([vc, _vc])

        sc_s_near = GraphSToS(units_conv, activation=act)(sc_near)
        sc_v_near = GraphVToS(units_conv, activation=act)([vc_near, dists])
        _sc_near = GraphConvS(units_conv, pooling='sum', activation=act)([sc_s_near, sc_v_near, near_not_bond_norm])

        vc_s_near = GraphSToV(units_conv, activation='tanh')([sc_near, dists])
        vc_v_near = GraphVToV(units_conv, activation='tanh')(vc_near)
        _vc_near = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s_near, vc_v_near, near_not_bond_norm])

        sc_near = Add()([sc_near, _sc_near])
        vc_near = Add()([vc_near, _vc_near])

    sc = Concatenate(axis=-1)([sc, sc_near])

    if pooling == 's2s':
        sc = Set2SetS(units_dense * 2)(sc)

    elif ',' in pooling:
        sc_pooling = []
        for pool in pooling.split(","):
            if pool == "sum":
                _sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
            elif pool == "avg":
                _sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
            elif pool == "max":
                _sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)
            sc_pooling.append(_sc)
        sc = Concatenate(axis=-1)(sc_pooling)

    elif pooling == "sum":
        sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
    elif pooling == "avg":
        sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
    elif pooling == "max":
        sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)

    if hyper["act_s"] == "prelu":
        _act = LeakyReLU(alpha=0.2)
        sc_out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc)
        out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc_out)
    else:
        sc_out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc)
        out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc_out)

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def model_molnet_edge_woNC(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    num_bond_features = hyper["num_bond_features"]

    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    act = hyper["act_s"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]

    units_edge = 128

    if "lrelu" in act:
        # act = tf.nn.leaky_relu
        try:
            rate = float(act.split("lrelu")[-1])
        except:
            rate = 0.2
        act = LeakyReLU(alpha=rate)

        # output = tf.keras.layers.Dense(n_units, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        # output = tf.layers.dense(input, n_units, activation=lambda x : tf.nn.leaky_relu(x, alpha=0.01)) in tf2.0

    elif act == "prelu":
        # act = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        act = PReLU(shared_axes=[1, 2])
        # PReLU(alpha_initializer=tf.initializers.constant(0.25))

    elif act == "swish":
        # act = swish
        act = tf.nn.swish

    elif act == "mish":
        act = mish

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    bonds = Input(name='bond_inputs', shape=(num_atoms, num_atoms, num_bond_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    # sc = Dense(units_conv, activation=LeakyReLU(alpha=0.2), kernel_regularizer=l2(0.005))(atoms)
    sc = Dense(units_conv, activation=act, kernel_regularizer=l2(0.005))(atoms)
    sc, vc, bf = GraphEmbed_edge()([sc, dists, bonds])

    for _ in range(num_layers):
        sc_s = GraphSToS_edge(units_conv, pooling='sum', activation=act)([sc, bf])
        sc_v = GraphVToS_edge(units_conv, pooling='sum', activation=act)([vc, dists, bf])

        vc_s = GraphSToV_edge(units_conv, pooling='sum', activation='tanh')([sc, dists, bf])
        vc_v = GraphVToV_edge(units_conv, pooling='sum', activation='tanh')([vc, bf])

        # sc = GraphConvS_edge(units_conv, units_edge, pooling='sum', activation=act)([sc_s, sc_v, bf])
        # vc = GraphConvV_edge(units_conv, units_edge, pooling='sum', activation='tanh')([vc_s, vc_v, bf])

        _sc = GraphConvS(units_conv, pooling='sum', activation=act)([sc_s, sc_v, adjms])
        _vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

        sc = Add()([sc, _sc])
        vc = Add()([vc, _vc])

    if pooling == 's2s':
        sc = Set2SetS(units_dense)(sc)
        # vc = Set2SetV(units_dense)(vc)

    elif ',' in pooling:
        sc_pooling = []
        for pool in pooling.split(","):
            if pool == "sum":
                _sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
            elif pool == "avg":
                _sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
            elif pool == "max":
                _sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)
            sc_pooling.append(_sc)
        sc = Concatenate(axis=-1)(sc_pooling)

    elif pooling == "sum":
        sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
    elif pooling == "avg":
        sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
    elif pooling == "max":
        sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)

    if hyper["act_s"] == "prelu":
        _act = LeakyReLU(alpha=0.2)
        sc_out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc)
        out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc_out)
        # vc_out = TimeDistributed(Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005)))(vc)
        # vc_out = TimeDistributed(Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005)))(vc_out)
    else:
        sc_out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc)
        out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc_out)

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def gcn(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    num_bond_features = hyper["num_bond_features"]
    units_conv = hyper["units_conv"]
    units_conv = 32
    units_dense = hyper["units_dense"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]
    
    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    bonds = Input(name='bond_inputs', shape=(num_atoms, num_atoms, 0))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))
    
    sc = GraphConv(units_conv, bias_initializer=None, activation='relu')([atoms, adjms])
    sc = GraphConv(units_conv, bias_initializer=None, activation='relu')([sc, adjms])

    if pooling == 's2s':
        sc = Set2SetS(units_dense)(sc)
    elif pooling == "sum":
        sc = Lambda(lambda x: tf.reduce_sum(x, axis=1))(sc)
    elif pooling == "avg":
        sc = Lambda(lambda x: tf.reduce_mean(x, axis=1))(sc)
    elif pooling == "max":
        sc = Lambda(lambda x: tf.reduce_max(x, axis=1))(sc)

    out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc)
    out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(out)
    
    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def mpnn(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    num_bond_features = hyper["num_bond_features"]

    num_layers = hyper["num_layers"]  # 2
    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    outputs = hyper["outputs"]
    
    t = 5
    step = 10
    n_hidden = 100
    
    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    bonds = Input(name='bond_inputs', shape=(num_atoms, num_atoms, num_bond_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))
    
    m = MessagePassing(t=t, n_hidden=n_hidden, activation='relu')([atoms, bonds])
    embed_atom = Dense(n_hidden, activation='relu')(m)
    
    embed_mol = Set2SetS(n_hidden, step=step)(embed_atom)
    embed_mol = Dense(n_hidden * 2, activation='relu')(embed_mol)
    
    out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(embed_mol)
    out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(out)

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def weave(hyper):

    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    num_bond_features = hyper["num_bond_features"]

    units_dense = hyper["units_dense"]

    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    outputs = hyper["outputs"]

    # W2N2: num_layers = 2, atom_pair_graph_distance = 2
    units_weave = 128
    n_hidden = 50
    units_fc1 = 2000
    units_fc2 = 100
    drop_rate = 0.25
    gaussian_expand = False
    compress_post_gaussian_expansion = False

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    bonds = Input(name='bond_inputs', shape=(num_atoms, num_atoms, num_bond_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))
    
    weave_a, weave_p = WeaveLayer(atom_filters=num_features, pair_filters=num_bond_features, update_pair=True)([atoms, bonds, adjms])
    weave_a, _ = WeaveLayer(atom_filters=n_hidden, pair_filters=n_hidden, update_pair=False)([weave_a, weave_p, adjms])  # (b, max_atoms, n_hidden)

    out = WeaveGather(units_weave, gaussian_expand=gaussian_expand,
                      compress_post_gaussian_expansion=compress_post_gaussian_expansion)(weave_a)
    # (b, units_weave * 11)
    out = Dense(units_fc1, activation='relu', kernel_regularizer=l2(0.005))(out)
    out = Dense(units_fc2, activation='relu', kernel_regularizer=l2(0.005))(out)

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def model_3DGCN(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    sc, vc = GraphEmbed()([atoms, dists])

    for _ in range(num_layers):
        sc_s = GraphSToS(units_conv, activation='relu')(sc)
        sc_v = GraphVToS(units_conv, activation='relu')([vc, dists])
        sc = GraphConvS(units_conv, pooling='sum', activation='relu')([sc_s, sc_v, adjms])

        vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
        vc_v = GraphVToV(units_conv, activation='tanh')(vc)
        vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

    if pooling == 's2s':
        sc = Set2SetS(units_dense)(sc)
        vc = Set2SetV(units_dense)(vc)
    elif pooling == "avg_adv_2":
        sc, vc = GraphGather(pooling=pooling)([sc, vc], adjms=adjms)
    elif ',' in pooling:  # multipooling
        sc_pooling = []
        vc_pooling = []
        #for pool in poolings:
        for pool in pooling.split(","):
            _sc, _vc = GraphGather(pooling=pool)([sc, vc])
            sc_pooling.append(_sc)
            vc_pooling.append(_vc)
        sc = Concatenate(axis=-1)(sc_pooling)
        vc = Concatenate(axis=-1)(vc_pooling)
        
    else:
        sc, vc = GraphGather(pooling=pooling)([sc, vc])

    sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc)
    sc_out = Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005))(sc_out)

    vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc)
    vc_out = TimeDistributed(Dense(units_dense, activation='relu', kernel_regularizer=l2(0.005)))(vc_out)
    vc_out = Flatten()(vc_out)

    out = Concatenate(axis=-1)([sc_out, vc_out])

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def swish(x):
    return x * tf.keras.backend.sigmoid(x)


def mish(x):
    return x * tf.keras.backend.tanh(tf.keras.backend.softplus(x))


def model_3DGCN_act(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    act = hyper["act_s"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]
    
    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))

    sc, vc = GraphEmbed()([atoms, dists])

    if "lrelu" in act:
        # act = tf.nn.leaky_relu
        try:
            rate = float(act.split("lrelu")[-1])
        except:
            rate = 0.2
        act = LeakyReLU(alpha=rate)

    elif act == "prelu":
        #act = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        act = PReLU(shared_axes=[1, 2])
        # PReLU(alpha_initializer=tf.initializers.constant(0.25))
    
    elif act == "swish":
        #act = swish
        act = tf.nn.swish
        
    elif act == "mish":
        act = mish
  
    for _ in range(num_layers):
        sc_s = GraphSToS(units_conv, activation=act)(sc)
        sc_v = GraphVToS(units_conv, activation=act)([vc, dists])
        if hyper["act_s"] == "prelu":
            sc = GraphConvS(units_conv, pooling='sum', activation="prelu")([sc_s, sc_v, adjms])
        else:
            sc = GraphConvS(units_conv, pooling='sum', activation=act)([sc_s, sc_v, adjms])
        #sc = GraphConvS(units_conv, pooling='sum', activation=act)([sc_s, sc_v, adjms])

        vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
        vc_v = GraphVToV(units_conv, activation='tanh')(vc)
        vc = GraphConvV(units_conv, pooling='sum', activation='tanh')([vc_s, vc_v, adjms])

    if pooling == 's2s':
        sc = Set2SetS(units_dense)(sc)
        vc = Set2SetV(units_dense)(vc)
        
    elif pooling == "avg_adv_2":
        sc, vc = GraphGather(pooling=pooling)([sc, vc], adjms=adjms)
        
    elif ',' in pooling:  # multipooling
        sc_pooling = []
        vc_pooling = []
        # for pool in poolings:
        for pool in pooling.split(","):
            _sc, _vc = GraphGather(pooling=pool)([sc, vc])
            sc_pooling.append(_sc)
            vc_pooling.append(_vc)
        sc = Concatenate(axis=-1)(sc_pooling)
        vc = Concatenate(axis=-1)(vc_pooling)
        
    else:
        sc, vc = GraphGather(pooling=pooling)([sc, vc])

    if hyper["act_s"] == "prelu":
        _act = LeakyReLU(alpha=0.2)
        sc_out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc)
        sc_out = Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005))(sc_out)
        vc_out = TimeDistributed(Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005)))(vc)
        vc_out = TimeDistributed(Dense(units_dense, activation=_act, kernel_regularizer=l2(0.005)))(vc_out)
    else:
        sc_out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc)
        sc_out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc_out)
        vc_out = TimeDistributed(Dense(units_dense, activation=act, kernel_regularizer=l2(0.005)))(vc)
        vc_out = TimeDistributed(Dense(units_dense, activation=act, kernel_regularizer=l2(0.005)))(vc_out)

    vc_out = Flatten()(vc_out)
    out = Concatenate(axis=-1)([sc_out, vc_out])

    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")

    return model


def model_3DGCN_edge(hyper):
    # Kipf adjacency, neighborhood mixing
    num_atoms = hyper["num_atoms"]
    num_features = hyper["num_features"]
    num_bond_features = hyper["num_bond_features"]

    units_conv = hyper["units_conv"]
    units_dense = hyper["units_dense"]
    num_layers = hyper["num_layers"]
    std = hyper["data_std"]
    loss = hyper["loss"]
    task = hyper["task"]
    act = hyper["act_s"]
    pooling = hyper["pooling"]
    outputs = hyper["outputs"]

    units_edge = 128

    if "lrelu" in act:
        # act = tf.nn.leaky_relu
        try:
            rate = float(act.split("lrelu")[-1])
        except:
            rate = 0.2
        act = LeakyReLU(alpha=rate)
    
        # output = tf.keras.layers.Dense(n_units, activation=tf.keras.layers.LeakyReLU(alpha=0.01))(x)
        # output = tf.layers.dense(input, n_units, activation=lambda x : tf.nn.leaky_relu(x, alpha=0.01)) in tf2.0

    elif act == "prelu":
        # act = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])
        act = PReLU(shared_axes=[1, 2])
        # PReLU(alpha_initializer=tf.initializers.constant(0.25))

    elif act == "swish":
        # act = swish
        act = tf.nn.swish

    elif act == "mish":
        act = mish

    atoms = Input(name='atom_inputs', shape=(num_atoms, num_features))
    bonds = Input(name='bond_inputs', shape=(num_atoms, num_atoms, num_bond_features))
    adjms = Input(name='adjm_inputs', shape=(num_atoms, num_atoms))
    dists = Input(name='coor_inputs', shape=(num_atoms, num_atoms, 3))
    
    sc, vc, bf = GraphEmbed_edge()([atoms, dists, bonds])
    # bf = Dense(units_edge, activation='relu', kernel_regularizer=l2(0.005))(bf)

    for _ in range(num_layers):
        sc_s = GraphSToS(units_conv, activation=act)(sc)  # 'relu'
        sc_v = GraphVToS(units_conv, activation=act)([vc, dists])
        
        vc_s = GraphSToV(units_conv, activation='tanh')([sc, dists])
        vc_v = GraphVToV(units_conv, activation='tanh')(vc)

        sc = GraphConvS_edge(units_conv, units_edge, pooling='sum', activation=act)([sc_s, sc_v, bf])
        vc = GraphConvV_edge(units_conv, units_edge, pooling='sum', activation='tanh')([vc_s, vc_v, bf])
    
    if pooling == 's2s':
        sc = Set2SetS(units_dense)(sc)
        vc = Set2SetV(units_dense)(vc)
    elif ',' in pooling:  # multipooling
        # print('pooling: ', pooling[:3], ' + ', pooling[3:])
        # for one_pooling in pooling.split(","):
        poolings = pooling.split(",")
        sc1, vc1 = GraphGather(pooling=poolings[0])([sc, vc])
        sc2, vc2 = GraphGather(pooling=poolings[1])([sc, vc])
        sc = Concatenate(axis=-1)([sc1, sc2])
        vc = Concatenate(axis=-1)([vc1, vc2])
    else:
        sc, vc = GraphGather(pooling=pooling)([sc, vc])

    sc_out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc)
    sc_out = Dense(units_dense, activation=act, kernel_regularizer=l2(0.005))(sc_out)
    
    vc_out = TimeDistributed(Dense(units_dense, activation=act, kernel_regularizer=l2(0.005)))(vc)
    vc_out = TimeDistributed(Dense(units_dense, activation=act, kernel_regularizer=l2(0.005)))(vc_out)
    vc_out = Flatten()(vc_out)
    
    out = Concatenate(axis=-1)([sc_out, vc_out])
    
    if task == "regression":
        out = Dense(outputs, activation='linear', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss, metrics=[std_mae(std=std), std_rmse(std=std)])
    elif task == "binary":
        out = Dense(outputs, activation='sigmoid', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    elif task == "classification":
        out = Dense(outputs, activation='softmax', name='output')(out)
        model = Model(inputs=[atoms, bonds, adjms, dists], outputs=out)
        model.compile(optimizer=Adam(lr=0.001), loss=loss)
    else:
        raise ValueError("Unsupported task on model generation.")
    
    return model
