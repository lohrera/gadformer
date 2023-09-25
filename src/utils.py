import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" #":4096:8" #":16:8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import copy
import random 
import os

import matplotlib.pyplot as plt

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_all(seed=42):
    
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    rs = RandomState(MT19937(SeedSequence(seed)))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.enabled = False
    torch.random.manual_seed(seed)

seed_all()

torch.set_float32_matmul_precision('high') #latest version 


class EarlyStopping: # source of original version: main-tul (adapted for GADFormer PyTorch Version)
    """[Early stops the training if validation loss doesn't improve after a given patience.]
    """

    def __init__(self, logger, model_file_path, patience=5, verbose=False, delta=0):
        """[Receive optional parameters]

        Args:
            patience (int, optional): [How long to wait after last time validation loss improved.]. Defaults to 7.
            verbose (bool, optional): [If True, prints a message for each validation loss improvement. ]. Defaults to False.
            delta (int, optional): [Minimum change in the monitored quantity to qualify as an improvement.]. Defaults to 0.
        """
        self.logger = logger
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.es_score_min_model = None
        self.es_score_min = np.Inf
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_file_path = model_file_path
        self.es_score_min_epoch = None
        self.val_loss_min_epoch = None

    def __call__(self, es_score, val_loss, model, epoch, val_loss_valid=True):
        """[this is a Callback function]

        Args:
            val_loss ([float]): [The loss of receiving verification was changed to accuracy as the stop criterion in our experiment]
            model (Object): [model waiting to be saved]
        """
        
        best_model_updated=False
        
        if val_loss_valid:  # -> val_loss >= trn_loss
            #score = -val_loss # original
            score = es_score # adaption for valid_loss_only=False
            
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(es_score, val_loss, model, epoch)
                best_model_updated = True
                self.counter = 0
            #elif score < self.best_score + self.delta: # original
            elif score >= self.best_score: # adaption for valid_loss_only=False
                self.counter += 1
                #self.logger.info(
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')                
            else:
                self.best_score = score
                self.save_checkpoint(es_score, val_loss, model, epoch)
                best_model_updated = True
                self.counter = 0
        
            if val_loss < self.val_loss_min:        
                self.val_loss_min = val_loss
                self.val_loss_min_epoch = epoch        
        else:
            self.counter += 1
            #self.logger.info(
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        
        
        if self.counter >= self.patience:
            self.early_stop = True
            
        return best_model_updated
    
    def save_checkpoint(self, es_score, val_loss, model, epoch):
        """[Saves model when validation loss decrease.]

        Args:
            val_loss ([type]): [The loss value corresponding to the best checkpoint needs to be saved]
            model (Object): [Save the model corresponding to the best checkpoint]
        """
        if self.verbose:
            #self.logger.info(
            print(f'Score decreased ({self.es_score_min:.6f} --> {es_score:.6f}).  Saving model {self.model_file_path} ...')
        
        torch.save(model.state_dict(), self.model_file_path)
        self.es_score_min = es_score
        self.es_score_min_epoch = epoch
        self.es_score_min_model = model

        
