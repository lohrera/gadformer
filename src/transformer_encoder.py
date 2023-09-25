
import numpy as np
import torch
import torch.nn as nn

from .attention import MHAtt


class TransformerEncoderBlockLayer(nn.Module):
    
    def __init__(self, d_inp_size, d_inp_embed_size, d_model, d_k, d_ffn_embed_size, num_heads, dropr=0.0, save_attn=True, winit_orig=True, log_engs=False):
        super(TransformerEncoderBlockLayer, self).__init__()
        
        self.d_inp_size = d_inp_size
        self.d_inp_embed_size = d_inp_embed_size
        self.d_ffn_embed_size = d_ffn_embed_size
        self.d_model = d_model
        self.d_k = d_k
        self.num_heads = num_heads
        self.save_attn = save_attn
        self.attn = None
        self.log_engs = log_engs
        self.energies = {'trn': {}, 'vld': {}}
        
        self.mh_att = MHAtt(d_model, d_k, d_inp_embed_size, num_heads, save_attn, winit_orig)
        self.dropout = nn.Dropout(dropr) 
        
        self.layer_norm1 = nn.LayerNorm([d_model, d_inp_embed_size]) # v4
        self.layer_norm2 = nn.LayerNorm([d_model, d_inp_embed_size]) # v4
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn_embed_size), # v4
            nn.Dropout(dropr), 
            nn.Tanh(), # nn.ReLU() (Vaswani et al.)
            nn.Linear(d_ffn_embed_size, d_model), # v4
            nn.Dropout(dropr) 
        )   
    
                
    def collect_attn(self, mh_attn_matrices, hgrps_all_):
        
        # mh_attn_matrices                  # (16, 256, 72, 72)
        
        mhatt_ = np.array(mh_attn_matrices) # (16, 256, 72, 72)
        mhatt_ = np.swapaxes(mhatt_, 0, 1)  # (256, 16, 72, 72) 
                
        hgrps_ = mhatt_.mean(axis=1)        # (256, 72, 72)        
        hgrps_all_.append(hgrps_)
        
        del mh_attn_matrices
    
        
    def log_energies(self, key, ep, x):
        
        if self.log_engs:
            phase = ('vld','trn')[self.training]        
            if key not in self.energies[phase].keys():
                self.energies[phase][key] = {}
            if ep not in self.energies[phase][key].keys():                
                self.energies[phase][key][ep] = []    
            self.energies[phase][key][ep].append(x)
        
    
    def forward(self, x, ep, blidx, hgrps_all_): # v4
        
        m = x.shape[0] # bs
        
        x0, self.attn = self.mh_att(x) # torch.Size([256, d_model, 8]) (d_model due to fc_out, before: qk * v and concat), attn: torch.Size([256, d_k,  d_k])  # v4
        
        if self.save_attn and hgrps_all_ is not None: self.collect_attn(self.attn, hgrps_all_)                
        if self.log_engs: self.log_energies(f"o_mha{blidx}", ep, x0.clone().detach().cpu().numpy())
        
        # x - torch.Size([256, d_model, 16]) v4 
        # x0 - torch.Size([256, d_model, 16]) v4
        
        x2 = self.layer_norm1(x + x0)              # Add & Norm - v4: torch.Size([256, d_model, 16]) 
        
        if self.log_engs: self.log_energies(f"i_ffn{blidx}", ep, x2.clone().detach().cpu().numpy())        
        
        x3 = x2.view((m, self.d_inp_embed_size, self.d_model))  # v4: torch.Size([256, 16, d_model])
        
        x4 = self.ffn(x3)                                       # v4: torch.Size([256, 16, d_model]) 
        
        if self.log_engs: self.log_energies(f"o_ffn{blidx}", ep, x4.clone().detach().cpu().numpy())
        
        x5 = x4.view((m, self.d_model, self.d_inp_embed_size))  # v4: torch.Size([256, d_model, 16])
        
        x6 = self.layer_norm2(x5 + x2)             # Add & Norm - v4: torch.Size([256, d_model, 16])
        
        if self.log_engs: self.log_energies(f"o_nrm{blidx}", ep, x6.clone().detach().cpu().numpy())
                
        return x6
