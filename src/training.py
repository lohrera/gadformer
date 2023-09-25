

import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_value_, clip_grad_norm_
from sklearn.metrics import roc_auc_score, average_precision_score

from tqdm import tqdm
import pytorch_optimizer
import copy
import os
import gc


from src.utils import EarlyStopping
from src.utils import seed_worker
num_worker = os.cpu_count()-1

def train_model(model, dl_trn, dl_vld, args, dict_hist):

    avg_trn_losses = []
    avg_vld_losses = []
    
    opti = getattr(torch.optim, args.opt) # RAdam, NAdam, Ranger, AdamP, AdamW (wd!=0), SGD
    optimizer = opti(model.parameters(), lr=args.lr, eps=1e-08, weight_decay=args.wd)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', threshold_mode ='abs', threshold=1e-4, factor=0.1, patience=args.pat_sched, min_lr=args.min_lr, eps=1e-08, verbose=True) 

    device='cuda'
    model=model.to(device)

    early_stopping = EarlyStopping(None, args.model_file_path, patience=args.pat_es, verbose=True)
    if model.save_attn: model.reset_attn()

    for epoch_idx in range(args.epochs):

        model.train()
        phase = 'trn'
        best_model_updated=False
        
        loss_trn_list = []
        y_hat_score_list = []
        y_true_list = []

        for batch_idx, (x_batch, _) in enumerate(tqdm(dl_trn)):
            optimizer.zero_grad()

            bs = x_batch.shape[0]
            seq_len = x_batch.shape[1]

            x_batch = x_batch.float().to(device)
            y_aux_labels = torch.tensor((bs*[0.])).unsqueeze(1) # auxilary target label based on assumption of the majority of normal groups
            y_aux_labels = y_aux_labels.float().to(device)

            y_hat_score = model.forward(x_batch, epoch_idx, early_stopping, phase) # predicted probability for an abnormal group (0. = normal group)
            
            loss = F.binary_cross_entropy(y_hat_score, y_aux_labels)

            loss.backward()    
            if args.clip_value is not None:
                clip_grad_norm_(model.parameters(), max_norm=args.clip_value)
            optimizer.step()       


            loss_trn_list.append(loss.item())

            y_hat_score_list.extend(y_hat_score.clone().detach().cpu().numpy().ravel())
            y_true_list.extend(y_aux_labels.clone().detach().cpu().numpy().ravel())
            x_batch = None
            gc.collect(generation=0)


        #dict_hist['trn']['y_hat_scores'].append(y_hat_score_list)    
        #dict_hist['trn']['y_true'].append(y_true_list)    
        avg_trn_losses.append(np.mean(loss_trn_list))
        dict_hist['trn']['losses'].append(avg_trn_losses[-1])
        
        
        model.eval()
        phase = 'vld'

        loss_vld_list = []
        y_hat_score_list = []
        y_true_list = []

        # validate --
        with torch.no_grad():
            for batch_idx, (x_batch, y_true_batch_labels) in enumerate(tqdm(dl_vld)):

                bs = x_batch.shape[0]
                seq_len = x_batch.shape[1]

                x_batch = x_batch.float().to(device)
                y_true_batch_labels = y_true_batch_labels.unsqueeze(1)
                y_true_batch_labels = y_true_batch_labels.float().to(device)

                y_hat_score = model.forward(x_batch, epoch_idx, early_stopping, phase) # predicted probability for an abnormal group (0. = normal group)
                
                loss_ = F.binary_cross_entropy(y_hat_score, y_true_batch_labels) 

                loss_vld_list.append(loss_.item())          

                y_hat_score_list.extend(y_hat_score.clone().detach().cpu().numpy().ravel())
                y_true_list.extend(y_true_batch_labels.clone().detach().cpu().numpy().ravel())
        # validate --
        
        #dict_hist['vld']['y_hat_scores'].append(y_hat_score_list)        
        #dict_hist['vld']['y_true'].append(y_true_list)   
        avg_vld_losses.append(np.mean(loss_vld_list))
        dict_hist['vld']['losses'].append(avg_vld_losses[-1]) 

        auroc = roc_auc_score(np.array(y_true_list), np.array(y_hat_score_list))
        auprc = average_precision_score(np.array(y_true_list), np.array(y_hat_score_list))
        dict_hist['vld']['auroc'].append(auroc)
        dict_hist['vld']['auprc'].append(auprc)
        
        
        if args.valid_loss_only: # original: valid_loss_only=True
            es_score = avg_vld_losses[-1] # original
            dict_hist['cp2'].append(es_score)
            dict_hist['cp'].append(es_score)
            #best_model_updated=early_stopping(es_score, model, val_loss, epoch_idx, (avg_vld_losses[-1] >= avg_trn_losses[-1]))
            best_model_updated=early_stopping(es_score, avg_vld_losses[-1], model, epoch_idx, True)
        else:
            ### optional ###
            ld = np.abs(avg_trn_losses[-1] - avg_vld_losses[-1])
            pe = np.abs(max(0., avg_trn_losses[-1] - avg_vld_losses[-1]))
            cld = avg_trn_losses[-1] + ld
            if len(avg_trn_losses) > 1:
                tld = avg_trn_losses[-2] - avg_trn_losses[-1]
                vld = avg_vld_losses[-2] - avg_vld_losses[-1]
                ld_prev = np.abs(avg_trn_losses[-1] - avg_vld_losses[-1])
                cld_prev = avg_trn_losses[-2] + ld_prev
                dict_hist['cp2'].append(((ld+ld_prev)/2)-((vld+tld)/2))
                dict_hist['cp'].append(((ld+ld_prev)/2)+pe-(vld+tld))
            else:
                dict_hist['cp2'].append(cld)
                dict_hist['cp'].append(cld)
            es_score = dict_hist['cp'][-1]
            best_model_updated=early_stopping(es_score, avg_vld_losses[-1], model, epoch_idx, True)
            ### optional ###
                
        prec=5
        print(f"{epoch_idx} trn|vld|cp[{round(avg_trn_losses[-1],prec)}|{round(avg_vld_losses[-1],prec)}|{round(dict_hist['cp'][-1],prec)}] - best score: {round(early_stopping.es_score_min,prec)} ({early_stopping.es_score_min_epoch})") #" - seed {seed}") #todo: add seed print

        if early_stopping.early_stop:
            print('Early Stop!')
            break
        else:
            if args.pat_sched >= 0: scheduler.step(early_stopping.val_loss_min)
                
        # attn update        
        if best_model_updated and model.save_attn and '_hgrps_all_dct' in model.__dict__.keys():
            del dict_hist['hgrps_all']
            gc.collect(generation=0)
            dict_hist['hgrps_all'] = copy.deepcopy(model._hgrps_all_dct) # trn and vld/tst
            model.reset_attn()
            
                
                
    return dict_hist, early_stopping.es_score_min_model #, early_stopping.es_score_min_epoch



def test_model(model, dl_tst, args, dict_hist):
    
    model.load_state_dict(torch.load(args.model_file_path))
    
    model.eval()
    phase = 'tst'
    device = 'cuda'
    
    loss_tst_list = []
    y_hat_score_tst_list = []
    y_true_tst_list = []
    
    # test --
    with torch.no_grad():
        for batch_idx, (x_batch, y_true_batch_labels) in enumerate(tqdm(dl_tst)):

            bs = x_batch.shape[0]
            seq_len = x_batch.shape[1]

            x_batch = x_batch.float().to(device)
            y_true_batch_labels = y_true_batch_labels.unsqueeze(1)
            y_true_batch_labels = y_true_batch_labels.float().to(device)

            y_hat_score = model.forward(x_batch, 0, None, phase) # predicted probability for an abnormal group (0. = normal group)

            loss_ = F.binary_cross_entropy(y_hat_score, y_true_batch_labels) 
    
            loss_tst_list.append(loss_.item())          

            y_hat_score_tst_list.extend(y_hat_score.clone().detach().cpu().numpy().ravel())
            y_true_tst_list.extend(y_true_batch_labels.clone().detach().cpu().numpy().ravel())
            x_batch = None
            gc.collect(generation=0)
    # test --
    
    
    #dict_hist['tst']['y_hat_scores'].append(y_hat_score_tst_list)        
    #dict_hist['tst']['y_true'].append(y_true_tst_list)   
    avg_tst_loss = np.mean(loss_tst_list)
    dict_hist['tst']['losses'].append(avg_tst_loss) # todo: dict_hist['tst']['loss'].append(avg_tst_loss)

    auroc = roc_auc_score(np.array(y_true_tst_list), np.array(y_hat_score_tst_list))
    auprc = average_precision_score(np.array(y_true_tst_list), np.array(y_hat_score_tst_list))
    dict_hist['tst']['auroc'].append(auroc)
    dict_hist['tst']['auprc'].append(auprc)
    
    if args.bas_phase == 'tst' and '_hgrps_all_dct' in model.__dict__.keys():
        del dict_hist['hgrps_all']['tst']
        gc.collect(generation=0)
        dict_hist['hgrps_all']['tst'] = copy.deepcopy(model._hgrps_all_dct['tst'])
    
    return dict_hist
