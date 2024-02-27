
import numpy as np

from utils import EarlyStopping
from tqdm import tqdm
import os
from utils import seed_worker
num_worker = os.cpu_count()-1

from sklearn.metrics import roc_auc_score, average_precision_score

from loaddataset import TrajectoryDataset
from torch.utils.data import DataLoader
from utils import seed_all
import argparse
import time
import glob
import gc
import os

from src.models import GADFormer
from src.training import train_model, test_model

from experiment_configs import settings, datasets, models, global_seeds, experiments_all, experiments_unsup_orig, experiments_semisup_orig, experiments_unsup_noise, experiments_semisup_noise, experiments_unsup_novelty, experiments_semisup_novelty
from datetime import datetime


root_dir = './datasets/files_valid/'  # content is 'valid'


def run(root_dir):
    
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    exp_all = [experiments_unsup_orig]
    for experiment_runs in exp_all:
        for exp in experiment_runs:
            if exp['model']['model_type'] != 'GADFormer': continue 
            #if exp['model']['model_type'] != 'GRU': continue 
            #if exp['model']['model_type'] != 'MainTulGAD': continue 

            # if experiment result file exists then continue to next experiment
            exp_done = glob.glob(f"./results/*_{exp['setting']['sco']}_{exp['setting']['ds']['dataset_name']}_{exp['etype']}_{exp['model']['model_type']}_*.txt")
            if len(exp_done)>0: 
                print(f"{exp_done}")
                continue


            ds_trn_id, ds_vld_id, ds_tst_id = exp['setting']['ds_train'], exp['setting']['ds_valid'], exp['setting']['ds_test']

            traj_step_features=exp['setting']['ds']['trj_step_feat']

            seeds = exp['seeds']
            seq_len = exp['setting']['ds']['input_dims']
            run_history = {}

            exp_setting=f"{ts}_{exp['setting']['sco']}_{exp['setting']['ds']['dataset_name']}_{exp['etype']}_{exp['model']['model_type']}_{exp['scaler']}_trn{ds_trn_id}_vld{ds_vld_id}_tst{ds_tst_id}"
            print(exp_setting)

            ##############################################################

            dataset_name=exp['setting']['ds']['dataset_name'] #'brightkite' #'synthetic' # 'amazon' # 'dbcargo'

            model_path="./temp/"
            os.makedirs(model_path, exist_ok=True)        
            model_name=f"{exp['model']['model_type']}_checkpoint_trn{ds_trn_id}_vld{ds_vld_id}_tst{ds_tst_id}.pt"

            prec=5
            num_seeds = len(seeds) #10
            best_model = None
            seed_performances = {'auroc': [], 'auprc': []}


            if 'yn' in dataset_name:
                bs=256      
                lr=1e-03 
                wd=0. 
                drp=0.
                seq_len=72 
                seq_len_dk=seq_len #72
                num_heads=12 

            if 'amazon' in dataset_name:
                bs=256      
                lr=1e-03 
                wd=0. 
                drp=0.
                seq_len=72 
                seq_len_dk=seq_len #72
                num_heads=12


            if 'brightkite' in dataset_name:
                bs=256      
                lr=1e-04 
                wd=0. 
                drp=0.
                seq_len=500 
                seq_len_dk=seq_len #500
                num_heads=8


            if 'dbcargo' in dataset_name:
                bs=256
                lr=(1e-02, 1e-03)['RU' in model_name]
                wd=0.
                drp=0.
                seq_len=72 
                seq_len_dk=seq_len #72
                num_heads=12


            num_layers=(4,2)['RU' in model_name] 
            clip_value = (None, 5)['RU' in model_name] 
            opt='RAdam'
            seg_len=2

            print(f"lr:{lr} bs:{bs} wd:{wd} drp:{drp} ds:{dataset_name} seq_len:{seq_len} seq_len_dk:{seq_len_dk} seg_len:{seg_len}")

            for seed in seeds:

                t1_ = time.time()
                print(f" ################ SEED {seed} - trn{ds_trn_id}_vld{ds_vld_id}_tst{ds_tst_id} ################")

                seed_all(seed)

                args = argparse.Namespace(d_step_feat=len(traj_step_features), d_inp_size=seq_len, d_inp_embed_size=16, 
                                          d_k=seq_len_dk, d_ffn_embed_size=2048, enc_layers=num_layers, num_heads=num_heads, num_nonlinearity_layers=2,
                                          k=None, sc='standard', lr=lr, batch_size=bs, epochs=150, clip_value=clip_value, 
                                          pat_es=20, #5, 
                                          pat_sched=10,
                                          wd=wd, min_lr=1e-6, drop=drp, valid_loss_only=True, 
                                          winit_orig=None, progressive_training=0, 
                                          model_file_path=f"{model_path}{model_name}",
                                          seg_len=seg_len, padding_mode='circular', padding_val=None, verbose=False, log_engs=False, bas_phase='tst', 
                                          opt=opt, temperature=10, lambda_parm=1, d_hidden_size=seq_len
                                          )
                print("-------------")
                print(args)
                print("-------------")

                print(f"{model_name} running...")
                if 'ormer' in model_name:
                    model = GADFormer(d_step_feat=args.d_step_feat, d_inp_size=args.d_inp_size, d_inp_embed_size=args.d_inp_embed_size, d_k=args.d_k, d_ffn_embed_size=args.d_ffn_embed_size, 
                                      num_heads=args.num_heads, num_layers=args.enc_layers, num_nonlinearity_layers=args.num_nonlinearity_layers, dropr=args.drop, save_attn=False, 
                                      winit_orig=args.winit_orig, progressive_training=args.progressive_training, seg_len=args.seg_len, padding_mode=args.padding_mode, 
                                      padding_val=args.padding_val, verbose=args.verbose, log_engs=args.log_engs)

                if 'RU' in model_name:               

                    model = GRUBaseline(d_step_feat=args.d_step_feat, d_inp_size=args.d_inp_size, d_inp_embed_size=args.d_inp_embed_size, 
                                d_hidden_size=args.d_k, # to check the best pendant for d_hidden_size from gru to transformer (d_k, d_model)
                                batch_size=bs,
                                num_layers=args.enc_layers, # gru layers as pendant for tranformer block layers (although literature recommends 2 GRU layers as sufficient)
                                num_nonlinearity_layers=num_nonlinearity_layers, # custom task head remains equal as well as progressive training
                                dropr=args.drop, progressive_training=None, verbose=False)


                ds_trn = TrajectoryDataset(root_dir, file_id=ds_trn_id, mode='train', sc=args.sc, submode='', traj_step_features=traj_step_features, k=None, seed=seed) #25
                ds_vld = TrajectoryDataset(root_dir, file_id=ds_vld_id, mode='valid', sc=args.sc, submode='', traj_step_features=traj_step_features, k=None, seed=seed) #22
                ds_tst = TrajectoryDataset(root_dir, file_id=ds_tst_id, mode='test', sc=args.sc, submode='', traj_step_features=traj_step_features, k=None, seed=seed) #24

                dl_trn = DataLoader(ds_trn, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_worker, worker_init_fn=seed_worker)
                dl_vld = DataLoader(ds_vld, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_worker, worker_init_fn=seed_worker)
                dl_tst = DataLoader(ds_tst, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=num_worker, worker_init_fn=seed_worker)

                dict_hist = {
                    'trn': {'y_hat_scores': [], 'y_true': [], 'losses': []},
                    'vld': {'y_hat_scores': [], 'y_true': [], 'losses': [], 'auroc': [], 'auprc': []},
                    'tst': {'y_hat_scores': [], 'y_true': [], 'losses': [], 'auroc': [], 'auprc': []},
                    'cp': [], 'cp2': [], 'hgrps_all': {'trn': [], 'vld': [], 'tst': []}
                }

                dict_hist, best_model = train_model(model, dl_trn, dl_vld, args, dict_hist)
                dict_hist = test_model(best_model, dl_tst, args, dict_hist)

                model = None
                gc.collect(generation=0)

                seed_test_auroc = dict_hist['tst']['auroc'][-1]
                seed_test_auprc = dict_hist['tst']['auprc'][-1]

                t2_ = time.time()
                print(f" ################ SEED {seed} - trn{ds_trn_id}_vld{ds_vld_id}_tst{ds_tst_id}: AUROC: {round(seed_test_auroc,prec)} AUPRC: {round(seed_test_auprc, prec)} duration: {round((t2_-t1_)/60.,1)}min ################")

                seed_performances['auroc'].append(seed_test_auroc)
                seed_performances['auprc'].append(seed_test_auprc)

                run_history[seed] = dict_hist

            avg_auroc = np.array(seed_performances['auroc']).mean()
            avg_auprc = np.array(seed_performances['auprc']).mean()
            std_auroc = np.array(seed_performances['auroc']).std()
            std_auprc = np.array(seed_performances['auprc']).std()

            print(f"AVG AUROC: {round(avg_auroc,prec)}+-{round(std_auroc,3)} AVG AUPRC: {round(avg_auprc, prec)}+-{round(std_auprc,3)}")


            ##############################################################

            #result_filename=f"{ts}_{exp['setting']['sco']}_{exp['setting']['ds']['dataset_name']}_{exp['etype']}_{exp['model']['model_type']}_{round(avg_auroc,3)}_{round(avg_auprc,3)}_{exp['scaler']}"
            result_filename=f"{ts}_{exp['setting']['sco']}_{exp['setting']['ds']['dataset_name']}_{exp['etype']}_{exp['model']['model_type']}_{round(avg_auroc,3)}_{round(avg_auprc,3)}"
            print(f"### {result_filename} ({len(seeds)} seeds): AUROC: {round(avg_auroc, prec)}+/-{round(std_auroc,3)} AUPRC: {round(avg_auprc, prec)}+/-{round(std_auprc,3)}")

            perf = {'auroc': avg_auroc, 'auroc_std': std_auroc, 'auprc': avg_auprc, 'auprc_std': std_auprc}

            with open(f"./results/{result_filename}.txt", "w") as f:
                f.write(str({'summary': perf, 'hist': run_history, 'args': args, 'exp': exp}))

    print(f"RUNS {ts} FINISHED")



