
import numpy as np

import torch
import torch.nn as nn
import copy

from .input_representation import InputEmbedding, PositionalEncoding, BERTInputRepresentation
from .transformer_encoder import TransformerEncoderBlockLayer


class CustomTaskHead(nn.Module): # for GAD on Trajectories
           
    def __init__(self, d_model, d_inp_embed_size, num_nonlinearity_layers=2, dropr=0.): #v4
        super(CustomTaskHead, self).__init__()
        
        self.d_model = d_model
        self.d_inp_embed_size = d_inp_embed_size
        self.dropr = dropr
        self.num_nonlinearity_layers = num_nonlinearity_layers
        
        
        self.task_nonlinearity_layers = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(d_model, d_model),   # v4: torch.Size([256, 72, 16]) 
                nn.ReLU()                      
            ) for _ in range(0, num_nonlinearity_layers)
        ])
        
        self.prj_inp_embed = nn.Linear(d_inp_embed_size, 1) 
        self.prj_d_model = nn.Linear(d_model, 1) # v4
        self.dropout = nn.Dropout(dropr)
        
    def forward(self, x):
        
        m = x.shape[0] # bs        
        x_ = x 
        
        for lidx in range(0, self.num_nonlinearity_layers):
            
            # x_ - torch.Size([256, 16, 72])
            x_ = self.task_nonlinearity_layers[lidx](x_)         # torch.Size([256, 16, 72])
            
        x1 = x_.view((m, self.d_model, self.d_inp_embed_size))   # torch.Size([256, 72, 16]) # v4
        
        x2 = self.prj_inp_embed(x1)                              # torch.Size([256, 72, 1])
        
        x3 = x2.view((m, 1, self.d_model))                       # torch.Size([256, 1, 72]) # v4
        
        x4 = self.prj_d_model(x3)                                # torch.Size([256, 1, 1])  # v4
        x5 = self.dropout(x4)                                    
        
        return x5

def weight_reinitialization(model):
    initru=0.1
    for name, p in model.named_parameters():
        if 'prj' in name or 'task' in name:
            if 'weight' in name:
                nn.init.xavier_uniform_(p.data)
        elif 'weight_ih' in name:
            None
            #nn.init.xavier_uniform_(p.data)
        elif 'weight_hh' in name:
            None
            #nn.init.uniform_(p.data, a=-initru, b=initru) 
            #nn.init.xavier_uniform_(p.data)               
        if 'bias' in name:
            p.data.fill_(0)
            

def freeze_weights(model, param_name, unfreeze, verbose=False):
    for a, b in model.named_parameters():
        if param_name in a:
            b.requires_grad=unfreeze
            if verbose: print(b)
                

class GADFormer(nn.Module):

    def __init__(self, d_step_feat, d_inp_size, d_inp_embed_size, d_k, d_ffn_embed_size, num_heads, num_layers=4, num_nonlinearity_layers=2, dropr=0.0, save_attn=True, winit_orig=True, progressive_training=None, seg_len=1, padding_mode='circular', padding_val=None, verbose=False, log_engs=False, bas_phase='tst'):
    
        super(GADFormer, self).__init__()
        
        self.d_step_feat = d_step_feat
        self.d_inp_size = d_inp_size
        self.d_model = d_inp_size    # for sequences d_model has to be d_inp_size (=seq_len), otherwise InputEmbedding needs to conduct an additional projection. (v4)
        self.d_k = d_k
        self.d_inp_embed_size = d_inp_embed_size
        self.d_ffn_embed_size = d_ffn_embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.log_engs = log_engs
        self.energies = {'trn': {}, 'vld': {}}
        bl_dct = {bl_idx: [] for bl_idx in range(0, num_layers)}
        self._hgrps_all_dct = {'trn': copy.deepcopy(bl_dct), 'vld': copy.deepcopy(bl_dct), 'tst': copy.deepcopy(bl_dct)}
        self.progressive_training = progressive_training
        self.seg_len = seg_len
        self.padding_mode = padding_mode
        self.padding_val = padding_val
        self.save_attn = save_attn
        self.bas_phase = bas_phase
        self.verbose = verbose
        self.unfrozen = False
        
        
        self.input_embedding = InputEmbedding(d_inp_size, d_step_feat, d_inp_embed_size) 
        
        self.positional_encoding = PositionalEncoding(d_model=d_inp_size, d_inp_embed_size=d_inp_embed_size) #v4
        
        self.bert_input_embedding = BERTInputRepresentation(self.input_embedding, self.positional_encoding, self.d_inp_size, seg_len, padding_mode, padding_val, verbose) 
        
        self.encoder = nn.ModuleList(
            [TransformerEncoderBlockLayer(d_inp_size, d_inp_embed_size, self.d_model, self.d_k, d_ffn_embed_size, num_heads, dropr, save_attn, winit_orig, log_engs) for _ in range(0, num_layers)] # v4
        )
        
        self.custom_task_head = CustomTaskHead(self.d_model, d_inp_embed_size, num_nonlinearity_layers, dropr) #v4
        
        if self.progressive_training is not None:
            freeze_weights(self, 'custom_task_head', unfreeze=False)
            print("custom_task_head frozen")
    
    def reset_attn(self): 
        
        del self._hgrps_all_dct
        gc.collect(generation=0)
        
        bl_dct = {bl_idx: [] for bl_idx in range(0, self.num_layers)}
        self._hgrps_all_dct = {'trn': copy.deepcopy(bl_dct), 'vld': copy.deepcopy(bl_dct), 'tst': copy.deepcopy(bl_dct)}
                
    
    def log_energies(self, key, ep, x):
        
        if self.log_engs:
            phase = ('vld','trn')[self.training]        
            if key not in self.energies[phase].keys():
                self.energies[phase][key] = {}
            if ep not in self.energies[phase][key].keys():                
                self.energies[phase][key][ep] = []    
            self.energies[phase][key][ep].append(x)
        
    
    def forward(self, x, ep, es=None, phase='trn'):
        
        # x - torch.Size([256, 72, 2]) 
        m = x.shape[0]
                
        x1 = self.bert_input_embedding.forward(x)
                
        x_in = x1
        for i, enc_block in enumerate(self.encoder):
            
            _hgprs_all_ = (None, self._hgrps_all_dct[phase][i])[phase=='trn' or self.bas_phase==phase]
            
            x2 = enc_block.forward(x_in, ep, i, _hgprs_all_)
                
            x_in = x2                                  # torch.Size([256, d_model, d_imp_embed_size]) , torch.Size([256, 72, 16])    
            
        x3 = x_in                                      # v4: torch.Size([256, 72, 16])
        
        x4 = x3.view(m, self.d_inp_embed_size, self.d_model) # v4: torch.Size([256, 16, 72])
        
        if self.training:
            if es is None or (self.progressive_training is not None and es.es_score_min != np.Inf and es.counter > self.progressive_training) or self.unfrozen:
                freeze_weights(self, 'custom_task_head', unfreeze=True)
                self.unfrozen = True
         
        x5 = self.custom_task_head(x4)                # torch.Size([256, 1, 1])
        
        
        if self.log_engs: self.log_energies(f"enc{i}ct", ep, x5.clone().detach().cpu().numpy())
        
        x6 = torch.sigmoid(x5)                        # torch.Size([256, 1, 1])
        
        if self.log_engs: self.log_energies(f"enc{i}sio", ep, x6.clone().detach().cpu().numpy())
        
        x7 = x6.squeeze(2)                            # torch.Size([256, 1])
        
        return x7
    
            

class GRUBaseline(nn.Module):
    
    def __init__(self, d_step_feat, d_inp_size, d_inp_embed_size, d_hidden_size, batch_size, num_layers=4, num_nonlinearity_layers=2, dropr=0.0, 
                 progressive_training=None, verbose=False):
        super(GRUBaseline, self).__init__()
        
        self.d_inp_size = d_inp_size
        self.d_inp_embed_size = d_inp_embed_size
        self.num_layers = num_layers
        self.d_hidden_size = d_hidden_size
        self.progressive_training = progressive_training
        self.dropr = dropr
        self.bidirectional = False
        self.D = (1,2)[self.bidirectional]
        self.verbose = verbose
        self.unfrozen = False
        self.save_attn = False
        
        self.input_embedding = InputEmbedding(d_inp_size, d_step_feat, d_inp_embed_size) # v2
        
        #self.ih0 = torch.zeros(self.D * self.num_layers, batch_size, self.d_hidden_size).to(x.device) # shape (D * num_layers, H_out) or (D*num_layers, N, H_out) 
        
        self.gru_baseline_net = nn.GRU(d_inp_embed_size, d_hidden_size, num_layers, bidirectional=self.bidirectional, batch_first=True, bias=True, dropout=dropr)
        self.prj_gru_to_task = nn.Linear(d_hidden_size, d_inp_embed_size)
        
        self.custom_task_head = CustomTaskHead(d_inp_size, d_inp_embed_size, num_nonlinearity_layers, dropr) #v4
        
        if self.progressive_training is not None:
            freeze_weights(self, 'custom_task_head', unfreeze=False)
            print("custom_task_head frozen")
            
        self.apply(weight_reinitialization)
        
    
    def forward(self, x, ep, es=None, phase='trn'):
        
        m = x.shape[0]
        
        if self.verbose: print(f"gruou  x {x.shape}")       # torch.Size([256, 72, 2])
        
        x1 = self.input_embedding(x)                        # torch.Size([256, 72, 16])
        if self.verbose: print(f"gruou  x1 {x1.shape}")
        
        self.ih0 = torch.zeros(self.D * self.num_layers, m, self.d_hidden_size).to(x.device) # shape (D * num_layers, H_out) or (D*num_layers, N, H_out) 
        
        x2, hn = self.gru_baseline_net(x1, self.ih0[:,:m,:].to(x1.device))
        #x2, hn = self.gru_baseline_net(x1)
        
        if self.verbose: print(f"gruou x2 {x2.shape}")      # torch.Size([256, 72, 50])
        if self.verbose: print(f"gruou hn {hn.shape}")                                     # torch.Size([4, 256, 50])
        
        
        x3 = self.prj_gru_to_task(x2)
        if self.verbose: print(f"gruou x3 {x3.shape}")
        
        x4 = x3.view(m, self.d_inp_embed_size, self.d_inp_size)
        if self.verbose: print(f"gruou x4 {x4.shape}")
        
        if self.training:
            if es is None or (self.progressive_training is not None and es.es_score_min != np.Inf and es.counter > self.progressive_training) or self.unfrozen:
                freeze_weights(self, 'custom_task_head', unfreeze=True)
                self.unfrozen = True
                
        x5 = self.custom_task_head(x4)
        if self.verbose: print(f"gruou x5 {x5.shape}")
        
        x6 = torch.sigmoid(x5)        
        if self.verbose: print(f"gruou x6 {x6.shape}")
        
        x7 = x6.squeeze(2)      
        if self.verbose: print(f"gruou x7 {x7.shape}")
        
        return x7
    
