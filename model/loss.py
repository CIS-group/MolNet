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

def std_r2(std=1):
    def r2(y_true, y_pred):
        ss_res = K.sum(K.square((y_true - y_pred) * std))
        ss_tot = K.sum(K.square((y_true - K.mean(y_true) * std)))
        return 1 - ss_res / (ss_tot + K.epsilon())
    
def metric_r2(std=1):
    from sklearn.metrics import r2_score
    def r2(y_true, y_pred):
        # return r2_score(y_true.eval(), y_pred.eval())  # numpy()
        total_error = tf.reduce_sum(tf.square(tf.subtract(y_true, tf.reduce_mean(y_true)) * std))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(y_true, y_pred)) * std)
        r_squared = tf.subtract(tf.cast(1, tf.float32), tf.divide(unexplained_error, total_error))
        return r_squared
    return r2


def metric_wsr(mean=0, std=1):
    def wsr(y_true, y_pred):
        y_true = (y_true * std) + mean
        y_pred = (y_pred * std) + mean
        wrong_sign = tf.where(tf.sign(y_true) == tf.sign(y_pred), tf.ones_like(y_true), tf.zeros_like(y_true))
        # wrong_sign = tf.where(tf.sign(tf.multiply(y_true, y_pred)) == -1., tf.ones_like(y_true), tf.zeros_like(y_true))
        # wrong_sign = tf.where(tf.multiply(y_true, y_pred) < 0., tf.ones_like(y_true), tf.zeros_like(y_true))
        wrong_sign_ratio = K.mean(wrong_sign)

        return wrong_sign_ratio

    return wsr
