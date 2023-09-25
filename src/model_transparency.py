
import math
import numpy as np
import torch


def calc_attn_stats(dict_hist, ratio_abnormal=0.05): 
    
    hgrps_all=None
    mu_attn = {}
    
    # bl_idx -> b
    # trj representation  -> (m represents an index for trajectories/groups and its representations/embeddings)
    # d_k -> 72
    # attn -> d_k x d_k
    
    for bl_idx, _ in enumerate(dict_hist['hgrps_all']['trn']):
    
        hgrps_all = torch.tensor(np.concatenate(dict_hist['hgrps_all']['trn'][bl_idx], axis=0))

        #hgrps_all.shape # torch.Size([2000, 72, 72]) # 8 batches

        total = hgrps_all.shape[0]
        top_k = math.ceil(ratio_abnormal * total)        
        tmp_attn_mu_normal = hgrps_all.mean(dim=0) # torch.Size([72, 72]) - mu_attn_mb (mean of all heads attention mat.)

        dist_hgrp2tmp_mu_normal = torch.norm(hgrps_all - tmp_attn_mu_normal, p=2, dim=(1,2))
        idx_abnormal = torch.topk(dist_hgrp2tmp_mu_normal, top_k, largest=True)[1]
        idx_normal = list(set(list(range(0, total))) - set(idx_abnormal.numpy()))

        mu_normal =   hgrps_all[idx_normal, :, :].mean(dim=0)    # torch.Size([72, 72]) - normal_mu_attn_b (mean of all normal trj attn mat)
        mu_abnormal = hgrps_all[idx_abnormal, :, :].mean(dim=0)  # torch.Size([72, 72]) - abnormal_mu_attn_b (mean of all abnormal trj attn mat)

        mu_attn[bl_idx] = {'normal': mu_normal, 'abnormal': mu_abnormal} # (normal and abnormal mean for one block)

    return mu_attn

def calc_block_attention_anomaly_score(dict_hist, bas_phase='tst', ratio_abnormal_=0.05):
    
    mu_attn_stats = calc_attn_stats(dict_hist, ratio_abnormal_)
    
    bas = {}
    for bl_idx, _ in enumerate(dict_hist['hgrps_all']['trn']):
        
        hgrps_all   = torch.tensor(np.concatenate(dict_hist['hgrps_all'][bas_phase][bl_idx], axis=0))
        mu_normal   = mu_attn_stats[bl_idx]['normal']
        mu_abnormal = mu_attn_stats[bl_idx]['abnormal']
        bas[bl_idx] = torch.clamp(torch.norm(hgrps_all - mu_normal, p=2, dim=(1,2)) / torch.norm(mu_abnormal - mu_normal, p=2), max=1.) # torch.Size([2000])
        
    return bas