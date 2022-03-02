# MolNet: A Chemically Intuitive Graph Neural Network for Prediction of Molecular Properties
This is an implementation of our paper "MolNet: A Chemically Intuitive Graph Neural Network for Prediction of Molecular Properties":

Yeji Kim, Yoonho Jeong, Jihoo Kim, Eok Kyun Lee, Won June Kim, Insung S. Choi, [MolNet: A Chemically Intuitive Graph Neural Network for Prediction of Molecular Properties] (Chem. Asian J. 2022) (submitted)


## Requirements

* Python 3.6.1
* Tensorflow 1.15
* tensorflow-probability=0.8.0
* Keras 2.25
* RDKit
* scikit-learn

## Data

* BACE
* Freesolv
* ESOL (= delaney)
* HIV

## Models

The `model` folder contains python scripts for building, training, and evaluation of the MolNet model.

The 'dataset.py' cleans and prepares the dataset for the model training with data agumentation.  
The 'layer.py' and 'model.py' build the model structure.  
The 'loss.py' and 'callbacks.py' assign the loss and metrics that we wanted to use.  
The 'trainer.py' and 'run_script.py' are for training of the model.  
