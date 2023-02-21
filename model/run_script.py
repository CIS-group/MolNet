from trainer import *

import time
import os
import sys
import csv
import argparse
import numpy as np

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="bace_cla", help="Dataset type")
parser.add_argument('--pooling', default='avg', help="Pooling type")
parser.add_argument('--act', default='relu', help="Activation function for scalar conv")
parser.add_argument('--gpu', default=0, help="GPU number you want to use")
parser.add_argument('--fold', default=10, type=int, help="k-fold cross validation")
parser.add_argument('--model', default="model_molnet", type=str, help="model type to use")
parser.add_argument('--split', default="", type=str, help="How to split dataset")
parser.add_argument('--st', default=None, help="start time")
parser.add_argument('--si', default=0, help="start iter")
parser.add_argument('--useoa', default=True, help="use overlap area or not")
parser.add_argument('--path', default="", help="Model path")
# "model_3DGCN_changeact", "model_3DGCN_adv", "model_3DGCN_lrelu"

args = parser.parse_args()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # 0
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

    if args.dataset in ["bace_cla", "bbbp", "hiv"]:
        loss = "binary_crossentropy"
        monitor = "val_roc"
        # monitor = "val_acc"
    elif args.dataset in ["bace_reg", "delaney", "freesolv"]:
        loss = "mse"
        monitor = "val_rmse"

    if args.dataset == "hiv":
        batch = 4
    elif args.dataset == "delaney":
        batch = 32
    else:
        batch = 8

    hyperparameters = {"epoch": 500, "model": args.model, "batch": batch, "fold": args.fold, "st": args.st,
                       "si": args.si,
                       "units_conv": 128, "units_dense": 128,
                       "num_layers": 2, "pooling": args.pooling, "act_s": args.act,
                       "loss": loss, "monitor": monitor, "label": ""}

    features = {"use_overlap_area": args.useoa, "use_full_ec": False, "use_formal_charge": True,
                "use_atom_symbol": True, "use_degree": True, "use_hybridization": True, "use_implicit_valence": True,
                "use_partial_charge": True, "use_ring_size": True, "use_hydrogen_bonding": True,
                "use_acid_base": True, "use_aromaticity": True, "use_chirality": True, "use_num_hydrogen": True}

    # Baseline
    trainer = Trainer(args.dataset, split_type=args.split)
    trainer.fit(use_multiprocessing=False, workers=1, max_queue_size=8, **hyperparameters, **features)
    #trainer.eval(args.path, use_multiprocessing=False, **hyperparameters, **features)
    #trainer.draw_heatmap(args.path, use_multiprocessing=False, **hyperparameters, **features)

