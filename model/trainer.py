import tensorflow as tf
import tensorflow.keras.backend as backend
from tensorflow.python.keras import backend as K
import math
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.framework import constant_op

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import argparse
from rdkit import Chem

import math
import time
import csv
import os

# import from current directory
from dataset import *
import model as m
from callback import *


class Trainer(object):
    def __init__(self, dataset, split_type):
        self.data = None
        self.model = None
        self.model_type = None
        self.hyper = {"dataset": dataset}
        self.split = split_type
        self.log = {}
        
        self.add_bf = False
        self.adj_cutoff = None
        
    def __repr__(self):
        text = ""
        for key, value in self.log.items():
            text += "{}:\t".format(key)
            for error in value[0]:
                text += "{0:.4f} ".format(float(error))
            text += "\n"

        return text

    def load_data(self, iter, batch=128, fold=10):
        self.data = Dataset(self.hyper["dataset"], batch=batch, fold=fold, iter=iter, split=self.split,
                            adj_cutoff=self.adj_cutoff, add_bf=self.add_bf, model_type=self.model_type)
            
        self.hyper["num_train"] = len(self.data.y["train"])
        self.hyper["num_test"] = len(self.data.y["test"])
        self.hyper["num_atoms"] = self.data.max_atoms
        self.hyper["data_std"] = self.data.std
        self.hyper["data_mean"] = self.data.mean
        self.hyper["task"] = self.data.task
        self.hyper["outputs"] = self.data.outputs
        self.hyper["batch"] = batch

    def load_model(self, model, pooling, act_s, units_conv=128, units_dense=128, num_layers=2, loss="mse"):
        self.hyper["model"] = model
        self.hyper["units_conv"] = units_conv
        self.hyper["units_dense"] = units_dense
        self.hyper["num_layers"] = num_layers
        self.hyper["loss"] = loss
        self.hyper["act_s"] = act_s
        self.hyper["pooling"] = pooling
        
        try:
            print('Model: ', model)
            self.model = getattr(m, model)(self.hyper)
            
        except:
            print('Use different model built')
            self.model = getattr(m, model)(self.hyper)
            self.model = None
            
        # self.model.summary()

    def fit(self, model, epoch, pooling, act_s, si=0, st=None, batch=128, fold=10,
            units_conv=128, units_dense=128, num_layers=2,
            loss="mse", monitor="val_rmse", mode="min", use_multiprocessing=True, workers=1, max_queue_size=10,
            label="", **kwargs):
        
        si = int(si)

        # 1. Generate CV folder
        now = datetime.now()
        base_path = "../result/{}/{}/".format(model, self.hyper["dataset"])
        if not (os.path.isdir("../result/{}/".format(model))):
            os.mkdir("../result/{}/".format(model))
        if not (os.path.isdir(base_path)):
            os.mkdir(base_path)
            
        if si != 0 and st is not None:
            log_path = base_path + "{}_c{}_d{}_l{}_p{}_{}_{}{}/".format(batch, units_conv, units_dense, num_layers,
                                                                        pooling, act_s, label, st)
        else:
            log_path = base_path + "{}_c{}_d{}_l{}_p{}_{}_{}{}/".format(batch, units_conv, units_dense, num_layers,
                                                                        pooling, act_s, label, now.strftime("%m%d%H"))
            si = 0
            
        if not (os.path.isdir(log_path)):
            os.makedirs(log_path)

        print('save path: ', log_path)

        if "edge" in model or model in ["gcn", "mpnn", "weave"]:
            self.add_bf = True
        self.model_type = model

        results = []

        for i in range(si, fold):
            # 1. Start
            print('Fold {}/{}'.format(i, fold))
            start_time = time.time()
            
            tb_path = log_path + "trial_{}/".format(i)
            if not (os.path.isdir(tb_path)):
                os.mkdir(tb_path)
                
            # 2. Generate data
            self.load_data(batch=batch, fold=fold, iter=i)
            self.data.set_features(**kwargs)
            self.hyper["num_features"] = self.data.num_features
            if self.add_bf:
                self.hyper["num_bond_features"] = self.data.num_bond_features
                print('num features: ', self.hyper["num_features"], self.hyper["num_bond_features"])
            else:
                print('Not use bond features')
                print('num features: ', self.hyper["num_features"])

            print('features: ', self.data.use_overlap_area, self.data.use_full_ec, self.data.use_atom_symbol)
            
            feats, label = self.data.generator("train")[0]
            print("feat dim: ", [feat.shape for feat in feats])
            #for i in range(0, 10):
            #    print(bond[0, i])

            # 2-1. Save data split and test results
            for target in ["train", "test"]:
                self.data.save_dataset(tb_path, pred=None, target=target)

            if self.data.task == "regression":
                header = ["train_mae", "test_mae", "train_rmse", "test_rmse"]
            else:
                header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1", "train_acc", "test_acc"]

            # 3. Make model
            self.load_model(model, units_conv=units_conv, units_dense=units_dense, num_layers=num_layers, loss=loss,
                            pooling=pooling, act_s=act_s)
            #tf.global_variables_initializer()

            print(self.model.summary())
            print('End summary')

            # Load best weight
            if os.path.exists(tb_path + "hyper.csv"):
                continue
            epoch_ws = [int(filename.split("-")[0]) for filename in os.listdir(tb_path) if ('-' in filename) and ('.hdf5' in filename)]
            initial_epoch = 0
            if len(epoch_ws) > 0:
                best_w = max(epoch_ws)
                w_list = [filename for filename in os.listdir(tb_path) if '{}-'.format(best_w) in filename]
                self.model.load_weights(tb_path + w_list[0])
                print("Loaded Weights from {}".format(tb_path + w_list[0]))
                initial_epoch = int(w_list[0])
            # if os.path.exists(tb_path + "best_weight.hdf5"):
            #    self.model.load_weights(tb_path + "best_weight.hdf5")
            #    print("Loaded Weights from {}".format(tb_path + "best_weight.hdf5"))
            
            # 4. Callbacks
            callbacks = []
            if self.data.task != "regression":
                callbacks.append(Roc(self.data.generator("test")))
                mode = "max"
            #lr = 0.001
            #if self.model_type == "weave":
            #    lr = 0.003
            patience = 50
            callbacks += [Tensorboard(log_dir=tb_path, write_graph=False, histogram_freq=0, write_images=True),
                          ModelCheckpoint(tb_path + "{epoch:01d}-{" + monitor + ":.3f}.hdf5", monitor=monitor,
                                          save_weights_only=True, save_best_only=True, period=1, mode=mode),
                          #ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=10, min_lr=0.0005)]
                          EarlyStopping(patience=patience, restore_best_weights=True),  # 15, hiv=10
                          CosineAnnealingDecay(initial_learning_rate=0.001, first_decay_steps=10, alpha=0.0,
                                               t_mul=2.0, m_mul=1.0, verbose=0)
                          #EarlyStopping(patience=30, restore_best_weights=True),  # 15, hiv=10
                          #CosineAnnealingDecay(initial_learning_rate=0.001, first_decay_steps=10, alpha=0.0,
                          #                     t_mul=2.0, m_mul=0.8, verbose=0)  # t_mul=2.0,
                          ]

            # 5. Fit
            self.model.fit_generator(self.data.generator("train"), validation_data=self.data.generator("test"),
                                     epochs=epoch, callbacks=callbacks, verbose=2,
                                     #steps_per_epoch=len(self.data.generator("train")),
                                     #validation_steps=len(self.data.generator("test")),
                                     shuffle=True,
                                     initial_epoch=initial_epoch,
                                     max_queue_size=max_queue_size,
                                     use_multiprocessing=use_multiprocessing, workers=workers)
            self.model.save_weights(tb_path + "best_weight.hdf5")
            self.hyper["train_time"] = time.time() - start_time

            # 7. Save hyper
            with open(tb_path + "hyper.csv", "w") as file:
                writer = csv.DictWriter(file, fieldnames=list(self.hyper.keys()))
                writer.writeheader()
                writer.writerow(self.hyper)

            # 8. Save data split and test results
            for target in ["train", "test"]:
                pred = self.model.predict_generator(self.data.generator(target, task="input_only"),
                                                    use_multiprocessing=use_multiprocessing, workers=workers)
                self.data.save_dataset(tb_path, pred=pred, target=target)

        #if si > 0:
        self.load_data(batch=batch, fold=fold, iter=0)

        if self.data.task == "regression":
            header = ["train_mae", "test_mae", "train_rmse", "test_rmse"]
        else:
            header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1", "train_acc", "test_acc",
                      "train_tpr", "test_tpr", "train_tnr", "test_tnr"]
            #header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1", "train_acc", "test_acc"]

        #for i in range(0, si):
        for i in range(0, fold):
            print('Load result of Fold {}/{}'.format(i, fold))
            tb_path = log_path + "trial_{}/".format(i)
            #if not os.path.exists(tb_path + "fold_results.csv"):

            if self.split == "random" or fold == 1:
                target = "valid"
            else:
                target = "train"
    
            # 5. Load true and predicted value
            train_true, train_pred, test_true, test_pred = [], [], [], []
            train_mols = Chem.SDMolSupplier(tb_path + "{}.sdf".format(target))
            test_mols = Chem.SDMolSupplier(tb_path + "test.sdf")
    
            for mol in train_mols:
                if mol is not None:
                    _y = float(mol.GetProp("pred"))
                    y = float(mol.GetProp("true"))
                    if y == -1:
                        continue
                    else:
                        train_true.append(y)
                        train_pred.append(_y)
    
            for mol in test_mols:
                if mol is not None:
                    _y = float(mol.GetProp("pred"))
                    y = float(mol.GetProp("true"))
                    if y == -1:
                        continue
                    else:
                        test_true.append(y)
                        test_pred.append(_y)

            train_true, train_pred, test_true, test_pred = np.array(train_true), np.array(train_pred),\
                                                           np.array(test_true), np.array(test_pred)
    
            # 6. Save train, test losses
            if self.data.task == "regression":
                losses = []
                for y_true, y_pred in [[train_true, train_pred], [test_true, test_pred]]:
                    val_mae = mean_absolute_error(y_true, np.squeeze(y_pred))
                    val_rmse = mean_squared_error(y_true, np.squeeze(y_pred), squared=False)
            
                    losses.append(val_mae)
                    losses.append(val_rmse)
        
                results.append([losses[0], losses[2], losses[1], losses[3]])
    
            else:
                losses = []
                for y_true, y_pred in [[train_true, train_pred], [test_true, test_pred]]:
                    val_roc = roc_auc_score(y_true, y_pred)
                    val_pr = average_precision_score(y_true, y_pred)
                    val_f1 = f1_score(y_true, y_pred.round())
                    val_acc = np.equal(y_true, y_pred.round()).astype(int).mean()
                    #val_acc = accuracy_score(y_true, y_pred)
                    #val_acc = accuracy_score(y_true, y_pred.round())
                    #print('acc: ', val_acc, accuracy_score(y_true, y_pred.round()))

                    losses.append(val_roc)
                    losses.append(val_pr)
                    losses.append(val_f1)
                    losses.append(val_acc)
          
                # active set for TPR, inactive set for TNR
                #delete_label = {"active": "0", "inactive": "1"}
                for delete in ["0", "1"]:  # "active", "inactive"
                    train_true, train_pred, test_true, test_pred = [], [], [], []
                    train_mols = Chem.SDMolSupplier(tb_path + "{}.sdf".format(target))
                    test_mols = Chem.SDMolSupplier(tb_path + "test.sdf")
    
                    for mol in train_mols:
                        if mol is not None:
                            if mol.GetProp('active') == delete:
                                continue
                            _y = float(mol.GetProp("pred"))
                            y = float(mol.GetProp("true"))
                            if y == -1:
                                continue
                            else:
                                train_true.append(y)
                                train_pred.append(_y)
    
                    for mol in test_mols:
                        if mol is not None:
                            if mol.GetProp('active') == delete:
                                continue
                            _y = float(mol.GetProp("pred"))
                            y = float(mol.GetProp("true"))
                            if y == -1:
                                continue
                            else:
                                test_true.append(y)
                                test_pred.append(_y)
    
                    train_true, train_pred, test_true, test_pred = np.array(train_true), np.array(train_pred), \
                                                                   np.array(test_true), np.array(test_pred)

                    val_truerate = np.equal(train_true, train_pred.round()).astype(int).mean()
                    test_truerate = np.equal(test_true, test_pred.round()).astype(int).mean()
                    
                    losses.append(val_truerate)
                    losses.append(test_truerate)
                    
                assert len(losses) == len(header)

                results.append([losses[0], losses[4], losses[1], losses[5],
                                losses[2], losses[6], losses[3], losses[7],
                                losses[8], losses[9], losses[10], losses[11]])

            with open(tb_path + "fold_results.csv", "w") as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow(header)
                for r in results:
                    writer.writerow(r)

            print('results: ', results[-1])

        assert fold == ['trial_' in filename for filename in os.listdir(log_path)].count(True)

        # 9. Save cross validation results
        with open(log_path + "raw_results.csv", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(header)
            for r in results:
                writer.writerow(r)
         
        results = np.array(results).astype(np.float)
        print(results.shape)
        
        _results = [np.mean(results, axis=0), np.std(results, axis=0)]
        with open(log_path + "k_fold_results.csv", "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)
            for r in _results:
                writer.writerow(r)

        print(header)
        print(_results)
        
        # Update cross validation log
        self.log["{}_B{}_C{}_D{}_L{}_P{}".format(model, batch, units_conv, units_dense, num_layers, pooling,
                                                 )] = _results

        print(self)
        print('saved path: ', log_path)
        print("Training Ended")

    def eval(self, path, model, epoch, pooling, act_s, si=0, st=None, batch=128, fold=10,
            units_conv=128, units_dense=128, num_layers=2,
            loss="mse", monitor="val_rmse", mode="min", use_multiprocessing=True, workers=1, max_queue_size=10,
            label="", **kwargs):
        
        # 1. Generate CV folder
        base_path = "./result/{}/{}/".format(model, self.hyper["dataset"])
        
        log_path = base_path + path
        print('save path: ', log_path)

        if "edge" in model or model in ["gcn", "mpnn", "weave"]:
            self.add_bf = True

        self.model_type = model

        results = []

        self.load_data(batch=batch, fold=fold, iter=0)

        if self.data.task == "regression":
            header = ["train_mae", "test_mae", "train_rmse", "test_rmse"]
        else:
            header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1", "train_acc", "test_acc",
                      "train_tpr", "test_tpr", "train_tnr", "test_tnr"]
            #header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1", "train_acc", "test_acc"]

        for i in range(0, fold):
            print('Load result of Fold {}/{}'.format(i, fold))
            tb_path = log_path + "trial_{}/".format(i)
            #if not os.path.exists(tb_path + "fold_results.csv"):

            if self.split == "random" or fold == 1:
                target = "valid"
            else:
                target = "train"
    
            # 5. Load true and predicted value

            train_true, train_pred, test_true, test_pred = [], [], [], []
            try:
                train_mols = Chem.SDMolSupplier(tb_path + "{}.sdf".format(target))
                test_mols = Chem.SDMolSupplier(tb_path + "test.sdf")
            except:
                self.load_data(batch=batch, fold=fold, iter=i)
                self.data.set_features(**kwargs)
                self.hyper["num_features"] = self.data.num_features
                if self.add_bf:
                    self.hyper["num_bond_features"] = self.data.num_bond_features
                    print('num features: ', self.hyper["num_features"], self.hyper["num_bond_features"])
                else:
                    print('Not use bond features')
                    print('num features: ', self.hyper["num_features"])

                print('features: ', self.data.use_overlap_area, self.data.use_full_ec, self.data.use_atom_symbol)

                self.load_model(model, units_conv=units_conv, units_dense=units_dense, num_layers=num_layers, loss=loss,
                                pooling=pooling, act_s=act_s)
                # Load best weight
                self.model.load_weights(tb_path + "/best_weight.hdf5")
                print("Loaded Weights from {}".format("/best_weight.hdf5"))

                for target in ["train", "test"]:
                    pred = self.model.predict_generator(self.data.generator(target, task="input_only"),
                                                        use_multiprocessing=use_multiprocessing, workers=workers)
                    self.data.save_dataset(tb_path, pred=pred, target=target)
                train_mols = Chem.SDMolSupplier(tb_path + "{}.sdf".format(target))
                test_mols = Chem.SDMolSupplier(tb_path + "test.sdf")

            for mol in train_mols:
                if mol is not None:
                    _y = float(mol.GetProp("pred"))
                    y = float(mol.GetProp("true"))
                    if y == -1:
                        continue
                    else:
                        train_true.append(y)
                        train_pred.append(_y)
    
            for mol in test_mols:
                if mol is not None:
                    _y = float(mol.GetProp("pred"))
                    y = float(mol.GetProp("true"))
                    if y == -1:
                        continue
                    else:
                        test_true.append(y)
                        test_pred.append(_y)

            train_true, train_pred, test_true, test_pred = np.array(train_true), np.array(train_pred),\
                                                           np.array(test_true), np.array(test_pred)
    
            # 6. Save train, test losses
            if self.data.task == "regression":
                losses = []
                for y_true, y_pred in [[train_true, train_pred], [test_true, test_pred]]:
                    val_mae = mean_absolute_error(y_true, np.squeeze(y_pred))
                    val_rmse = mean_squared_error(y_true, np.squeeze(y_pred), squared=False)
            
                    losses.append(val_mae)
                    losses.append(val_rmse)
        
                results.append([losses[0], losses[2], losses[1], losses[3]])
    
            else:
                losses = []
                for y_true, y_pred in [[train_true, train_pred], [test_true, test_pred]]:
                    val_roc = roc_auc_score(y_true, y_pred)
                    val_pr = average_precision_score(y_true, y_pred)
                    val_f1 = f1_score(y_true, y_pred.round())
                    val_acc = np.equal(y_true, y_pred.round()).astype(int).mean()
                    #val_acc = accuracy_score(y_true, y_pred)
                    #val_acc = accuracy_score(y_true, y_pred.round())
                    #print('acc: ', val_acc, accuracy_score(y_true, y_pred.round()))

                    losses.append(val_roc)
                    losses.append(val_pr)
                    losses.append(val_f1)
                    losses.append(val_acc)
          
                # active set for TPR, inactive set for TNR
                #delete_label = {"active": "0", "inactive": "1"}
                for delete in ["0", "1"]:  # "active", "inactive"
                    train_true, train_pred, test_true, test_pred = [], [], [], []
                    train_mols = Chem.SDMolSupplier(tb_path + "{}.sdf".format(target))
                    test_mols = Chem.SDMolSupplier(tb_path + "test.sdf")
    
                    for mol in train_mols:
                        if mol is not None:
                            if mol.GetProp('active') == delete:
                                continue
                            _y = float(mol.GetProp("pred"))
                            y = float(mol.GetProp("true"))
                            if y == -1:
                                continue
                            else:
                                train_true.append(y)
                                train_pred.append(_y)
    
                    for mol in test_mols:
                        if mol is not None:
                            if mol.GetProp('active') == delete:
                                continue
                            _y = float(mol.GetProp("pred"))
                            y = float(mol.GetProp("true"))
                            if y == -1:
                                continue
                            else:
                                test_true.append(y)
                                test_pred.append(_y)
    
                    train_true, train_pred, test_true, test_pred = np.array(train_true), np.array(train_pred), \
                                                                   np.array(test_true), np.array(test_pred)

                    val_truerate = np.equal(train_true, train_pred.round()).astype(int).mean()
                    test_truerate = np.equal(test_true, test_pred.round()).astype(int).mean()
                    
                    losses.append(val_truerate)
                    losses.append(test_truerate)
                    
                assert len(losses) == len(header)

                results.append([losses[0], losses[4], losses[1], losses[5],
                                losses[2], losses[6], losses[3], losses[7],
                                losses[8], losses[9], losses[10], losses[11]])

            with open(tb_path + "fold_results.csv", "w") as file:
                writer = csv.writer(file, delimiter=",")
                writer.writerow(header)
                for r in results:
                    writer.writerow(r)

            print('results: ', results[-1])

        assert fold == ['trial_' in filename for filename in os.listdir(log_path)].count(True)

        # 9. Save cross validation results
        with open(log_path + "raw_results_eval.csv", "w") as file:
            writer = csv.writer(file, delimiter=",")
            writer.writerow(header)
            for r in results:
                writer.writerow(r)
         
        results = np.array(results).astype(np.float)
        print(results.shape)
        
        _results = [np.mean(results, axis=0), np.std(results, axis=0)]
        with open(log_path + "k_fold_results_eval.csv", "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=",")
            writer.writerow(header)
            for r in _results:
                writer.writerow(r)

        print(header)
        print(_results)
        
        # Update cross validation log
        self.log["{}_B{}_C{}_D{}_L{}_P{}".format(model, batch, units_conv, units_dense, num_layers, pooling,
                                                 )] = _results

        print(self)
        print('saved path: ', log_path)
        print("Training Ended")

    def draw_heatmap(self, path, model, epoch, pooling, act_s, si=0, st=None, batch=128, fold=10,
            units_conv=128, units_dense=128, num_layers=2,
            loss="mse", monitor="val_rmse", mode="min", use_multiprocessing=True, workers=1, max_queue_size=10,
            label="", **kwargs):
    
        from collections import Counter
        from keras.models import Model
        
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        from figure import Draw
        from figure.Draw import DrawingOptions
        from matplotlib import colors
        import numpy as np
        import csv, math
        
        # 1. Generate CV folder
        base_path = "./result/{}/{}/".format(model, self.hyper["dataset"])
        
        log_path = base_path + path
        print('save path: ', log_path)
        
        if "edge" in model or model in ["gcn", "mpnn", "weave"]:
            self.add_bf = True
            
        self.model_type = model

        results = []

        self.load_data(batch=batch, fold=fold, iter=0)

        if self.data.task == "regression":
            header = ["train_mae", "test_mae", "train_rmse", "test_rmse"]
        else:
            header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1", "train_acc", "test_acc",
                      "train_tpr", "test_tpr", "train_tnr", "test_tnr"]
            #header = ["train_roc", "test_roc", "train_pr", "test_pr", "train_f1", "test_f1", "train_acc", "test_acc"]
        
        for i in range(0, fold):
            si = int(si)
            if si != 0 and i != si:
                continue
            print('Load result of Fold {}/{}'.format(i, fold))
            tb_path = log_path + "trial_{}/".format(i)
            #if not os.path.exists(tb_path + "fold_results.csv"):

            if self.split == "random" or fold == 1:
                target = "valid"
            else:
                target = "train"
    
            # 5. Load true and predicted value

            self.load_data(batch=batch, fold=fold, iter=i)
            self.data.set_features(**kwargs)
            self.hyper["num_features"] = self.data.num_features
            if self.add_bf:
                self.hyper["num_bond_features"] = self.data.num_bond_features
                print('num features: ', self.hyper["num_features"], self.hyper["num_bond_features"])
            else:
                print('Not use bond features')
                print('num features: ', self.hyper["num_features"])

            print('features: ', self.data.use_overlap_area, self.data.use_full_ec, self.data.use_atom_symbol)

            self.load_model(model, units_conv=units_conv, units_dense=units_dense, num_layers=num_layers, loss=loss,
                            pooling=pooling, act_s=act_s)
            self.model.summary()
            # Load best weight
            self.model.load_weights(tb_path + "/best_weight.hdf5")
            print("Loaded Weights from {}".format("/best_weight.hdf5"))
            
            try:
                train_mols = Chem.SDMolSupplier(tb_path + "{}.sdf".format(target))
                test_mols = Chem.SDMolSupplier(tb_path + "test.sdf")
            except:
                for target in ["train", "test"]:
                    pred = self.model.predict_generator(self.data.generator(target, task="input_only"),
                                                        use_multiprocessing=use_multiprocessing, workers=workers)
                    self.data.save_dataset(tb_path, pred=pred, target=target)
                    
                train_mols = Chem.SDMolSupplier(tb_path + "{}.sdf".format(target))
                test_mols = Chem.SDMolSupplier(tb_path + "test.sdf")

            if self.model_type == "model_3DGCN_edge_v5_equiv5":
                # Make submodel for retreiving features
                feature_model = Model(inputs=self.model.input,
                                      outputs=[self.model.get_layer("graph_conv_s_2").output,
                                               self.model.get_layer("graph_conv_v_2").output])
                
            elif self.model_type == "model_3DGCN_v5_equiv5_woBF":
                # Make submodel for retreiving features
                feature_model = Model(inputs=self.model.input,
                                      outputs=[self.model.get_layer("add_5").output,
                                               self.model.get_layer("add_7").output])
                                    # outputs=[self.model.get_layer("graph_conv_s_2").output, self.model.get_layer("graph_conv_v_2").output])
                
            scalar_c, scalar_nc = feature_model.predict_generator(self.data.generator("test"))
            print(scalar_c.shape)
            
            # Parse feature to heatmap index
            scalar_c = np.insert(scalar_c, 0, 10e-6, axis=1)  # To find 0 column, push atom index by 1
            scalar_c_idx = np.argmax(scalar_c, axis=1)

            scalar_c_idx_dict = []
            for scalar in scalar_c_idx:
                dic = Counter(scalar)
                if 0 in dic.keys():
                    dic.pop(0)
                new_dic = {key - 1: value for key, value in dic.items()}
    
                idx = []
                for atom_idx in range(self.hyper["num_atoms"]):
                    if atom_idx in new_dic:
                        idx.append(new_dic[atom_idx] / units_conv)
                    else:
                        idx.append(0)
                scalar_c_idx_dict.append(idx)

            scalar_nc = np.insert(scalar_nc, 0, 10e-6, axis=1)  # To find 0 column, push atom index by 1
            scalar_nc_idx = np.argmax(scalar_nc, axis=1)

            scalar_nc_idx_dict = []
            for scalar in scalar_nc_idx:
                dic = Counter(scalar)
                if 0 in dic.keys():
                    dic.pop(0)
                new_dic = {key - 1: value for key, value in dic.items()}
    
                idx = []
                for atom_idx in range(self.hyper["num_atoms"]):
                    if atom_idx in new_dic:
                        idx.append(new_dic[atom_idx] / units_conv)
                    else:
                        idx.append(0)
                scalar_nc_idx_dict.append(idx)

            # Get 2D coordinates
            mols = []
            for mol in test_mols:
                AllChem.Compute2DCoords(mol)
                mols.append(mol)

            DrawingOptions.bondLineWidth = 1.5
            DrawingOptions.elemDict = {}
            DrawingOptions.dotsPerAngstrom = 8
            DrawingOptions.atomLabelFontSize = 6
            DrawingOptions.atomLabelMinFontSize = 4
            DrawingOptions.dblBondOffset = 0.3
            cmap = colors.LinearSegmentedColormap.from_list("", ["white", "#fbcfb7", "#e68469", "#c03638"])
            cmap_r = colors.LinearSegmentedColormap.from_list("", ["white", "#fbcfb7", "#ED3F3E", "#DA0F13"])
            cmap_b = colors.LinearSegmentedColormap.from_list("", ["white", "#BFBFFF", "#7879FF", "#1F1FFF"])
            # cmap_b = colors.LinearSegmentedColormap.from_list("", ["white", "#00A6D7", "#0058B3", "#001B87"])

            save_p = "./result/heatmap/{}/{}/{}".format(self.hyper["dataset"], model, path)
            if not (os.path.isdir("./result/heatmap/")):
                os.mkdir("./result/heatmap/")
            if not (os.path.isdir("./result/heatmap/{}/".format(self.hyper["dataset"]))):
                os.mkdir("./result/heatmap/{}/".format(self.hyper["dataset"]))
            if not (os.path.isdir("./result/heatmap/{}/{}/".format(self.hyper["dataset"], model))):
                os.mkdir("./result/heatmap/{}/{}/".format(self.hyper["dataset"], model))
            if not (os.path.isdir(save_p)):
                os.mkdir(save_p)
            
            for idx, (mol, scalar_c_dic, scalar_nc_dic) in enumerate(
                    zip(mols[:20], scalar_c_idx_dict[:20], scalar_nc_idx_dict[:20])):
                
                img_size = 900  # 200
                
                fig = Draw.MolToMPL(mol, coordScale=1, size=(img_size, img_size))
                fig.savefig(save_p + "test_mol{}.png".format(idx), bbox_inches='tight')

                x, y, _z = Draw.calcAtomGaussians(mol, 0.015, weights=scalar_c_dic, step=0.0025)
                z = np.zeros((400, 400))
                z[:399, :399] = np.array(_z)[1:, 1:]
                max_scale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
    
                fig.axes[0].imshow(z, cmap=cmap, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1),
                                   vmin=0, vmax=max_scale)
                fig.axes[0].set_axis_off()
                fig.savefig(save_p + "test_{}_scalar_c.png".format(idx), bbox_inches='tight')
                #fig.clf()

                fig = Draw.MolToMPL(mol, coordScale=1, size=(img_size, img_size))
                x, y, _z = Draw.calcAtomGaussians(mol, 0.015, weights=scalar_nc_dic, step=0.0025)
                z = np.zeros((400, 400))
                z[:399, :399] = np.array(_z)[1:, 1:]
                max_scale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
    
                fig.axes[0].imshow(z, cmap=cmap, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1),
                                   vmin=0, vmax=max_scale)
                fig.axes[0].set_axis_off()
                fig.savefig(save_p + "test_{}_scalar_nc.png".format(idx), bbox_inches='tight')
                #fig.clf()

                fig = Draw.MolToMPL(mol, coordScale=1, size=(img_size, img_size))
                x, y, _z = Draw.calcAtomGaussians(mol, 0.015, weights=np.add(scalar_c_dic, scalar_nc_dic) / 2, step=0.0025)
                z = np.zeros((400, 400))
                z[:399, :399] = np.array(_z)[1:, 1:]
                max_scale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
    
                fig.axes[0].imshow(z, cmap=cmap, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1),
                                   vmin=0, vmax=max_scale)
                fig.axes[0].set_axis_off()
                fig.savefig(save_p + "test_{}_merge.png".format(idx), bbox_inches='tight')
                #fig.clf()
                
                # show c and nc in one figure

                fig = Draw.MolToMPL(mol, coordScale=1, size=(img_size, img_size), imageType='svg')
                x, y, _z = Draw.calcAtomGaussians(mol, 0.015, weights=scalar_c_dic, step=0.0025)
                z = np.zeros((400, 400))
                z[:399, :399] = np.array(_z)[1:, 1:]
                max_scale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
                fig.axes[0].imshow(z, cmap=cmap_r, alpha=1.0, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1),
                                   vmin=0, vmax=max_scale)
                fig.axes[0].set_axis_off()
                
                x, y, _z = Draw.calcAtomGaussians(mol, 0.015, weights=scalar_nc_dic, step=0.0025)
                z = np.zeros((400, 400))
                z[:399, :399] = np.array(_z)[1:, 1:]
                max_scale = max(math.fabs(np.min(z)), math.fabs(np.max(z)))
                fig.axes[0].imshow(z, cmap=cmap_b, alpha=0.7, interpolation='bilinear', origin='lower', extent=(0, 1, 0, 1),
                                   vmin=0, vmax=max_scale)
                fig.axes[0].set_axis_off()
                fig.savefig(save_p + "test_{}_mix.svg".format(idx), bbox_inches='tight')
                # fig.clf()
                
                plt.close('all')
                
                print(idx, 'compound: ', mol.GetProp("PARENT_CMPD_ID"), mol.GetProp("target"))

        print(self)
        print("Heatmap Generation Ended")


class CosineAnnealingDecay(tf.keras.callbacks.Callback):
    def __init__(self,
                 initial_learning_rate=0.001,
                 first_decay_steps=10,
                 alpha=0.0,
                 t_mul=2.0,
                 m_mul=1.0,
                 verbose=0,
                 **kwargs):
        super(CosineAnnealingDecay, self).__init__()
        
        if alpha < 0.0:
            raise ValueError('CosineAnnealingDecay ' 'does not support an alpha < 0.0.')
        self.initial_learning_rate = initial_learning_rate
        self.first_decay_steps = first_decay_steps
        self.alpha = alpha
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.verbose = verbose
        self.turn = 0
        self.current_sum_steps = 0
        self.current_decay_steps = self.first_decay_steps
        self.current_lr = self.initial_learning_rate
    
    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            # lr = float(K.get_value(self.model.optimizer.lr))
            
            current_epoch = epoch - self.current_sum_steps
            if current_epoch >= self.current_decay_steps:
                current_epoch = current_epoch - self.current_decay_steps
                self.current_sum_steps = self.current_decay_steps + self.current_sum_steps
                self.current_decay_steps = self.current_decay_steps * self.t_mul
                self.current_lr = self.current_lr * self.m_mul
                self.turn += 1
                # print(epoch, ' current: ', turn, current_lr, current_decay_steps, current_epoch)
            
            lr = self.alpha + 0.5 * (self.current_lr - self.alpha) * (
                    1.0 + math.cos(math.pi * current_epoch / self.current_decay_steps))

            # print(self.current_lr, self.current_sum_steps, self.current_decay_steps, self.turn, '-> ', lr)
            
            # print(epoch, ' current: ', turn, current_lr, current_decay_steps, current_epoch, ' -> ', lr)
        
        except TypeError:  # Support for old API for backward compatibility
            print('Error!!!')
        if not isinstance(lr, (float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                             'should be float.')
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler reducing learning '
                  'rate to %s.' % (epoch + 1, lr))
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)


def random_rotation_matrix():
    theta = np.random.rand() * 2 * np.pi
    r_x = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand() * 2 * np.pi
    r_y = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    theta = np.random.rand() * 2 * np.pi
    r_z = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])

    return np.matmul(np.matmul(r_x, r_y), r_z)


def degree_rotation_matrix(axis, degree):
    theta = degree / 180 * np.pi
    if axis == "x":
        r = np.array([1, 0, 0, 0, np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta)]).reshape([3, 3])
    elif axis == "y":
        r = np.array([np.cos(theta), 0, np.sin(theta), 0, 1, 0, -np.sin(theta), 0, np.cos(theta)]).reshape([3, 3])
    elif axis == "z":
        r = np.array([np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, 0, 0, 1]).reshape([3, 3])
    else:
        raise ValueError("Unsupported axis for rotation: {}".format(axis))

    return r
