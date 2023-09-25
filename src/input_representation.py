import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbedding(nn.Module):
    
    def __init__(self, d_inp_size, d_step_feat, d_inp_embed_size, static_encoding=False):
        super(InputEmbedding, self).__init__()
        
        self.d_k = d_inp_size # d_k # 72 # trajectory steps
        self.d_step_feat = d_step_feat # 2
        self.d_inp_embed_size = d_inp_embed_size # 8
        
        self.inp_prj_feat = nn.Linear(d_step_feat, d_inp_embed_size)
    
        
    def forward(self, x):
        
        # x - torch.Size([256, 72, 2])
        # m = x.shape[0] #bs
        
        x1 = self.inp_prj_feat(x) # torch.Size([256, 72, 16])      
        
        return x1  
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, d_inp_embed_size):
        super(PositionalEncoding, self).__init__()
        
        # d_model = d_inp_size
        
        self.pe = torch.zeros(d_inp_embed_size, d_model) # 16, 72        
        self.pe.requires_grad = False
        
        for pos in range(0, d_inp_embed_size): 
            for i in range(0, d_model):
                
                if 2*i < d_model: denom = 10000. ** (2*i/d_model)    
                if 2*i < d_model: self.pe[pos,2*i] = math.sin( pos / denom )
                if 2*i+1 < d_model: self.pe[pos,2*i+1] = math.cos( pos / denom )
        
        self.pe = self.pe.permute((1, 0)) # 72, 16
        
        
    def to(self, device):
        self.pe = self.pe.to(torch.device(device))
        return self
        
    def forward(self, x, **kwargs):
        return self.pe[:, :x.size(1)].to(x.device.type) # 72, 16
    
    
class BERTInputRepresentation(nn.Module):
    
    def __init__(self, input_token_embedding, positional_encoding, seq_len, seg_len=1, padding_mode='circular', padding_val=None, verbose=False):
        super(BERTInputRepresentation, self).__init__()
        
        self.input_token_embedding = input_token_embedding
        self.positional_encoding = positional_encoding
        self.verbose = verbose
        self.seq_len = seq_len
        self.seg_len = seg_len
        self.padding_mode = padding_mode # 'constant', 'reflect', 'replicate' or 'circular'
        self.padding_val = padding_val # default: None
        
        # calculate sequence separation indicies
        self.lst_split_indicies=list(set(set([i if i % self.seg_len == 0 and i>0 else None for i in range(0, self.seq_len)])-set([None])))
        self.lst_split_indicies.sort()
        if self.verbose: print(lst_split_indicies)
            
        self.fc_seg_embedding = nn.Linear(seg_len, 1)
        
        
    def calculate_segment_embedding(self, x):
        
        # --- segment embedding creation
        
        #t0 = time.time()
        
        # split sequences at separation indicies into segments (chunks)
        chunks = torch.tensor_split(x, self.lst_split_indicies, dim=1)       # torch.Size([256, 22, 16]) - foreach        
        
        seg_embeddings = torch.zeros((x.shape[0], x.shape[1], x.shape[2]))   # torch.Size([256, 72, 16])      
        seg_embeddings.requires_grad = False
        
        # create embedding for each segment
        for idx, chunk in enumerate(chunks):
            #t1 = time.time()
            
            x3 = chunk                                                       # torch.Size([256, 22, 16])
            
            # calculate pad_len
            remaining_seg_len = chunk.shape[1]
            pad_len = self.seg_len - remaining_seg_len
            pad = (0, pad_len)

            # pad segment if necessary according to pad_len
            if pad_len == 0:
                x4_ = x3
            else:
                x3_ = x3.permute((0, 2, 1))                                          # torch.Size([256, 16, 22])
                x4 = F.pad(x3_, pad, mode=self.padding_mode, value=self.padding_val) # torch.Size([256, 16, 25]) # todo: reproducibility issue?
                x4_ = x4.permute((0, 2, 1))                                          # torch.Size([256, 25, 16])

            if self.verbose:
                pd.DataFrame(x4_[0].detach().cpu().numpy(), list(range(0,x4_.shape[1]))).plot(xlim=[-1,26],ylim=[-4,4],legend=None)
                pd.DataFrame(x4_[0,:-pad_len,:].detach().cpu().numpy(), list(range(0,x4_.shape[1]-pad_len))).plot(xlim=[-1,26],ylim=[-4,4],legend=None)

            # embed seg_len tokens as the related segment representation
            x5 = x4_.permute((0, 2, 1))      # torch.Size([256, 16, 25])
            x5_ = self.fc_seg_embedding(x5)  # torch.Size([256, 16, 1]) 
            x6 = x5_.permute((0, 2, 1))      # torch.Size([256, 1, 16])
                        
            seg_embedding_tokens = x6.repeat(1, remaining_seg_len, 1).cpu()
            
            indices = torch.tensor(list(range(0,remaining_seg_len)))+(idx*self.seg_len)
            
            # --- token-wise segment embedding assignment in form of batch chunks
            seg_embeddings.index_add_(dim=1, index=indices, source=seg_embedding_tokens)
            seg_embedding_tokens = None
            indices = None
            
            
        chunks = None
        
        return seg_embeddings.to(x.device.type)       # torch.Size([256, 72, 16])
        
    def forward(self, x):
        
        # BERT (Devlin et al): 
        # input representation is constructed by 
        # summing the corresponding token (x1), segment (x2), and position embeddings (x3)
        
                                                                 # torch.Size([256, d_k, 16])
        x1 = self.input_token_embedding.forward(x)               # torch.Size([256, 72, 16])
        
        if self.seg_len == 1:
            x2 = x1                                              # torch.Size([256, 72, 16])
        else:
            x2 = x1 + self.calculate_segment_embedding(x1)       # torch.Size([256, 72, 16])
        
        x3 = x2 + self.positional_encoding(x2)                   # torch.Size([256, 72, 16])
        
        return x3 
