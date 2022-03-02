import keras.backend as K
import tensorflow as tf


def std_mae(std=1):
    def mae(y_true, y_pred):
        return K.mean(K.abs(y_pred - y_true)) * std

    return mae


def std_rmse(std=1):
    def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square((y_pred - y_true)))) * std

    return rmse
