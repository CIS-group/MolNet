import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as backend
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op
import math
from keras import backend as K
from keras.callbacks import Callback, TensorBoard
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error


class Tensorboard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


class Roc(Callback):
    def __init__(self, val_gen):
        super(Roc, self).__init__()

        self.val_gen = val_gen

    def on_epoch_end(self, epoch, logs={}):
        val_roc, val_pr = calculate_roc_pr(self.model, self.val_gen)

        logs.update({'val_roc': val_roc, 'val_pr': val_pr})
        print('\rval_roc: %s - val_pr: %s' % (str(round(val_roc, 4)), str(round(val_pr, 4))), end=100 * ' ' + '\n')


def calculate_roc_pr(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence, use_multiprocessing=True, workers=6)

    if y_true.ndim == 1:
        val_roc = roc_auc_score(y_true, y_pred)
        val_pr = average_precision_score(y_true, y_pred)

    elif y_true.ndim == 2:
        y_true = y_true.transpose()
        y_pred = y_pred.transpose()

        unmask_idx = [np.where(y != mask)[0] for y in y_true]
        val_roc = [roc_auc_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]
        val_pr = [average_precision_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]

        val_roc = np.array(val_roc).mean()
        val_pr = np.array(val_pr).mean()
        y_pred = y_pred.transpose()

    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_roc, val_pr, y_pred
    else:
        return val_roc, val_pr


def calculate_roc_pr_f1(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence, use_multiprocessing=True, workers=6)

    if y_true.ndim == 1:
        val_roc = roc_auc_score(y_true, y_pred)
        val_pr = average_precision_score(y_true, y_pred)
        val_f1 = f1_score(y_true, y_pred.round())

    elif y_true.ndim == 2:
        y_true = y_true.transpose()
        y_pred = y_pred.transpose()

        unmask_idx = [np.where(y != mask)[0] for y in y_true]
        val_roc = [roc_auc_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]
        val_pr = [average_precision_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]
        val_f1 = [f1_score(yt[idx], yp[idx].round()) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]

        val_roc = np.array(val_roc).mean()
        val_pr = np.array(val_pr).mean()
        val_f1 = np.array(val_f1).mean()
        y_pred = y_pred.transpose()

    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_roc, val_pr, val_f1, y_pred
    else:
        return val_roc, val_pr, val_f1


def calculate_f1_acc(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence)
    #y_pred = model.predict_generator(sequence, use_multiprocessing=True, workers=6)

    if y_true.ndim == 1:
        #print(type(y_pred))
        val_f1 = f1_score(y_true, y_pred.round())
        val_accuracy = np.equal(y_true, y_pred.round()).astype(int).mean()  # np.array(y_pred.round()).mean()

    elif y_true.ndim == 2:
        y_true = y_true.transpose()
        y_pred = y_pred.transpose()

        unmask_idx = [np.where(y != mask)[0] for y in y_true]
        val_pr = [average_precision_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]
        val_f1 = [f1_score(yt[idx], yp[idx].round()) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]
        val_accuracy = [accuracy_score(yt[idx], yp[idx].round()) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]

        val_pr = np.array(val_pr).mean()
        val_f1 = np.array(val_f1).mean()
        val_accuracy = np.array(val_accuracy).mean()

        y_pred = y_pred.transpose()

    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_f1, val_accuracy, y_true, y_pred
    else:
        return val_f1, val_accuracy


def calculate_acc(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence)

    if y_true.ndim == 1:
        val_accuracy = np.equal(y_true, y_pred.round()).astype(int).mean()  # np.array(y_pred.round()).mean()

    elif y_true.ndim == 2:
        y_true = y_true.transpose()
        y_pred = y_pred.transpose()

        unmask_idx = [np.where(y != mask)[0] for y in y_true]
        val_accuracy = [accuracy_score(yt[idx], yp[idx].round()) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]
        val_accuracy = np.array(val_accuracy).mean()

        #y_pred = y_pred.transpose()

    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_accuracy, y_true, y_pred
    else:
        return val_accuracy


def calculate_pr(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence, use_multiprocessing=True, workers=6)

    if y_true.ndim == 1:
        val_pr = average_precision_score(y_true, y_pred)

    elif y_true.ndim == 2:
        y_true = y_true.transpose()
        y_pred = y_pred.transpose()

        unmask_idx = [np.where(y != mask)[0] for y in y_true]
        val_pr = [average_precision_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]

        val_pr = np.array(val_pr).mean()
        y_pred = y_pred.transpose()

    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_pr, y_pred
    else:
        return val_pr


def calculate_f1(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence, use_multiprocessing=True, workers=6)

    if y_true.ndim == 1:
        val_f1 = f1_score(y_true, y_pred.round())

    elif y_true.ndim == 2:
        y_true = y_true.transpose()
        y_pred = y_pred.transpose()

        unmask_idx = [np.where(y != mask)[0] for y in y_true]
        val_f1 = [f1_score(yt[idx], yp[idx]) for (yt, yp, idx) in zip(y_true, y_pred, unmask_idx)]

        val_f1 = np.array(val_f1).mean()
        y_pred = y_pred.transpose()

    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_f1, y_pred
    else:
        return val_f1
    
def calculate_reg_esolv(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence) #, use_multiprocessing=True, workers=6)

    if y_true.ndim == 1:
        val_mae = mean_absolute_error(y_true, np.squeeze(y_pred))
        val_rmse = mean_squared_error(y_true, np.squeeze(y_pred), squared=False)
        val_wrong_sign = np.not_equal(np.sign(np.squeeze(y_pred)), np.sign(y_true)).astype(np.float32)
        val_wsr = np.mean(val_wrong_sign, axis=-1)

    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_mae, val_rmse, val_wsr, y_pred
    else:
        return val_mae, val_rmse, val_wsr
    
    
def calculate_reg(model, sequence, mask=-1, return_pred=False):
    y_true = sequence.y
    y_pred = model.predict_generator(sequence) #, use_multiprocessing=True, workers=6)

    if y_true.ndim == 1:
        mae = tf.keras.losses.MeanAbsoluteError()
        val_mae = mae(y_true, y_pred)
        rmse = tf.keras.metrics.RootMeanSquaredError()
        val_rmse = rmse(y_true, y_pred)
        #mape = tf.keras.losses.MeanAbsolutePercentageError()
        #val_mape = mape(y_true, y_pred)
        value, idx, count = tf.unique_with_counts(tf.math.sign(tf.math.multiply(y_true, y_pred)))
        for i, val in enumerate(value):
            if val == -1.0:
                # print('wrong: {}'.format(count_list[1][i].cpu().detach().numpy()))
                val_wrong_sign = count[i]

        print('wrong: ', val_wrong_sign, '/', len(y_pred))
        return val_mae, val_rmse, val_wrong_sign / len(y_pred)


    else:
        raise ValueError("Unsupported output shape for auc calculation")

    if return_pred:
        return val_mae, val_rmse, val_mape, y_pred
    else:
        return val_mae, val_rmse, val_mape



class CosineDecayRestarts(tf.keras.callbacks.Callback):
    def __init__(self, initial_learning_rate, first_decay_steps, alpha=0.0, t_mul=2.0, m_mul=1.0):
        super(CosineDecayRestarts, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.alpha = alpha
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.batch_step = 0

    def on_train_batch_begin(self, step, logs=None):
        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(self.batch_step, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        self.batch_step += 1

    def schedule(self, step, lr):
        def compute_step(completed_fraction, geometric=False):
            """Helper for `cond` operation."""
            if geometric:
                i_restart = math_ops.floor(
                  math_ops.log(1.0 - completed_fraction * (1.0 - self.t_mul)) /
                  math_ops.log(self.t_mul))

                sum_r = (1.0 - self.t_mul**i_restart) / (1.0 - self.t_mul)
                completed_fraction = (completed_fraction - sum_r) / self.t_mul**i_restart

            else:
                i_restart = math_ops.floor(completed_fraction)
                completed_fraction -= i_restart

            return i_restart, completed_fraction

        completed_fraction = step / self.first_decay_steps

        i_restart, completed_fraction = control_flow_ops.cond(
          math_ops.equal(self.t_mul, 1.0),
          lambda: compute_step(completed_fraction, geometric=False),
          lambda: compute_step(completed_fraction, geometric=True))

        m_fac = self.m_mul**i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + math_ops.cos(
          constant_op.constant(math.pi) * completed_fraction))
        decayed = (1 - self.alpha) * cosine_decayed + self.alpha

        return math_ops.multiply(self.initial_learning_rate, decayed)