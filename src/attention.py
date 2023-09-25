import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class SHAtt(nn.Module):
    
    def __init__(self, d_model, d_k, d_inp_embed_size, winit_orig=True):
        super(SHAtt, self).__init__()
        
        self.d_model = d_model
        self.d_k = d_k # ( Vaswani et al. -> d_q = d_k = d_v = d_model )    
        
        self.d_inp_embed_size = d_inp_embed_size 
        self.d_qkv = 3*d_k # v4
        
        if winit_orig is None or winit_orig:
            self.fc_qkv = nn.Linear(self.d_model, self.d_qkv, bias=True) # SHA v4
        else:
            self.fc_qkv = nn.Linear(self.d_model, self.d_qkv, bias=False) # Karpathy et al. (-> d_k != d_model, d_k=d_inp_size, d_model=d_model) # SHA v4
        
        self._reset_parameters(winit_orig)
        

    def _reset_parameters(self, winit_orig=True):
        if winit_orig is not None:
            # Original Transformer initialization, see PyTorch documentation
            if winit_orig:
                # https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html (Tutorial)
                nn.init.xavier_uniform_(self.fc_qkv.weight)
                self.fc_qkv.bias.data.fill_(0)
            else:
                # Karpathy et al.
                nn.init.normal_(self.fc_qkv.weight, mean=0., std=np.sqrt(2/(self.d_k+self.d_model)))

    def scaled_dot_product(self, q, k, v, mask=None):
        # source: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html (Tutorial)
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention
        
    def forward(self, x):
        
        # x - torch.Size([256,  72, 16])
        # x - torch.Size([256, d_k, 16])
        
        m = x.shape[0] # number of groups in batch = bs
        
        x1 = x.permute((0, 2, 1))                             # v2: torch.Size([256, 16, 72])  
        x2 = self.fc_qkv(x1)                                  # v2: torch.Size([256, 16, 216]) 
        x3 = x2.permute((0, 2, 1))                            # v2: torch.Size([256, 216, 16])
        
        q = x3[:, :self.d_k, :].clone()
        k = x3[:, self.d_k:2*self.d_k, :].clone()
        v = x3[:, 2*self.d_k:, :].clone()
        
        out, att = self.scaled_dot_product(q, k, v, mask=None) # v4: torch.Size([256, d_k, 8]) 
        
        return out, att


class MHAtt(nn.Module):
    
    def __init__(self, d_model, d_k, d_inp_embed_size, num_heads, save_attn=False, winit_orig=True):
        super(MHAtt, self).__init__()
        
        self.d_model = d_model 
        self.d_k = d_k       
        self.d_inp_embed_size = d_inp_embed_size
        self.num_heads = num_heads
        self.save_attn = save_attn
        
        
        self.mha = nn.Sequential(
            *[SHAtt(d_model, d_k, d_inp_embed_size, winit_orig) for _ in range(0,self.num_heads)]
        )
        
        if winit_orig is None or winit_orig:
            self.fc_out = nn.Linear(num_heads * d_k, d_model, bias=True) # original (tutorial) v4
        else:
            self.fc_out = nn.Linear(num_heads * d_k, d_model, bias=False) # no bias - Karpathy et al. v4
            
        self._reset_parameters(winit_orig)

    def _reset_parameters(self, winit_orig=True):
        # Original Transformer initialization, see PyTorch documentation
        if winit_orig is not None:
            if winit_orig:
                nn.init.xavier_uniform_(self.fc_out.weight)
                self.fc_out.bias.data.fill_(0)
            else:
                # Karpathy et al.
                nn.init.normal_(self.fc_out.weight, mean=0., std=np.sqrt(2/(self.d_inp_size+self.d_k)))

    def forward(self, x):
        
        m = x.shape[0] #bs
        attn=[]
        x_=None
        
        for h in range(0, self.num_heads):
            
            out, att = self.mha[h].forward(x)            
            if x_ is None: #todo
                x_ = out
            else:
                x_ = torch.concat([x_, out])
            if self.save_attn: attn.append(att.clone().detach().cpu().numpy())
        
        # e.g. 
        # num_heads=1: num_heads * out: torch.Size([256,         1 *  72,  16]) - attn: torch.Size([256, 72,   72])
        # num_heads=1: num_heads * out: torch.Size([256, num_heads * d_k,  16]) - attn: torch.Size([256, d_k, d_k])       # v4
        # ...
        
        x_ = x_.reshape(m, self.d_inp_embed_size, self.num_heads * self.d_k) # v4
        # num_heads=1: torch.Size([256, 16, num_heads * d_k]) 
        
        o = self.fc_out(x_) # torch.Size([256, 16, d_model])         
                
        o = o.reshape(m, self.d_model, self.d_inp_embed_size) # v4
        
        # o - torch.Size([256,      72, 16])
        # o - torch.Size([256, d_model, 16]) # v4
        
        return o, attn