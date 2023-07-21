
import torch
import numpy as np
import pandas as pd
import umap
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from experiment_configs import files


def get_traj_embeddings(traj_data_, labels=None, verbose=False, seed=42, sc='robust'):
    
    traj_data_flat = traj_data_.reshape(-1, traj_data_.shape[1]*traj_data_.shape[2])
    
    if sc=='robust':
        scaler=RobustScaler() 
    else:
        scaler=StandardScaler()    
    
    traj_data_scaled = scaler.fit_transform(traj_data_flat)
    
    m = umap.UMAP(random_state=seed)
    traj_embeddings = m.fit_transform(traj_data_scaled)
    if verbose and labels is not None: plt.scatter(traj_embeddings[:, 0], traj_embeddings[:, 1], c=labels)
    
    return traj_embeddings

def get_augmented_trajectories(traj_data_, aug_traj_len, num_traj_step_features, topknn_traj_indices, k):
        
    aug_trajectories=[]
    aug_traj_len = traj_data_.shape[1]
    aug_traj_seg_len = math.ceil(aug_traj_len / k)

    for idx_query_trj, trjknn in enumerate(topknn_traj_indices):
        
        # create augmented trajectory from segments of shuffled knn trajectories
        # todo: define seglen randomly and attach via round robin from knn trajectories until desired aug_tran_seg_len reached
        curstep,idx_knn=0,0
        aug_traj=torch.zeros((aug_traj_len, num_traj_step_features))
        while curstep < aug_traj_len:
            seglen = (aug_traj_seg_len,aug_traj_len-curstep)[(curstep+aug_traj_seg_len)>aug_traj_len]
            aug_traj[curstep:curstep+seglen] = torch.tensor(traj_data_[topknn_traj_indices[idx_query_trj, idx_knn], curstep:curstep+seglen,:])
            curstep=curstep+seglen
            idx_knn+=1
        aug_trajectories.append(aug_traj.numpy())
        
    return torch.tensor(aug_trajectories)

def load_trajectory_data(root_dir, file_id:int=0, mode:str='train', traj_step_features=['X_Coord','Y_Coord'], sc='robust'):
    
    filepath=f"{root_dir}{files[file_id]}"
    print(f'\n-------------Creating *{mode}* dataset from dataset file {filepath} -----\n')

    df = pd.read_csv(filepath)
    labels = np.array(df['Label'])

    print(df)
    
    file = filepath.split('/')[-1]

    if 'driver' in file:
        traj_len, total_trajectories = 72, int(file[-7:-4])
    
    elif 'bright' in file:
        traj_len, total_trajectories = int(file.split('_')[1]), int(file.split('_')[2])

    else: # synthetic
        _, traj_len, total_trajectories, _ = file.split('_')
        traj_len, total_trajectories = int(traj_len), int(total_trajectories)

    print(f"trajectory length: {traj_len}, abnormal trajectories: {np.sum(labels[:total_trajectories])}/{total_trajectories}")
    
    traj_Entity_IDs=list(set(i for i in df.Entity))[:total_trajectories]

    coords=None # changed column names since original columns did not met policies
    for traj_entity_ID in traj_Entity_IDs: 

        df_temp=df.loc[df['Entity'] == traj_entity_ID] # trajectory steps of entity
        df_temp=df_temp.sort_values(by=['Step'])
        df_temp=df_temp[:traj_len] # restrict trajectory to max. traj_len steps
        df_temp=df_temp[traj_step_features].to_numpy()

        try:
            coords=np.vstack((coords,df_temp))
        except:
            coords=df_temp

    # coords.shape (72000,2)

    if sc=='robust':
        scaler=RobustScaler() 
    else:
        scaler=StandardScaler() 

    scaler.fit(coords)
    coords_scaled = scaler.transform(coords)
    traj_data = np.reshape(coords_scaled, (len(traj_Entity_IDs), traj_len, len(traj_step_features)))
    # traj_data.shape (1000, 72, 2)

    traj_labels = labels[:total_trajectories]
    # traj_labels.shape (1000,)

    return traj_data, traj_labels


class TrajectoryDataset(Dataset): # refactored version for MainTUL Comparison (Reviewer feedback response for missing related work comparison)

    def __init__(self, root_dir, file_id, mode, sc, submode='', traj_step_features=['X_Coord','Y_Coord'], k=None, seed=42):
        super().__init__()
        
        self.k=k
        self.data, self.labels = load_trajectory_data(root_dir, file_id, mode, traj_step_features, sc)
        
        if self.k is not None: 
            #MainTUL Comparison
            # get shuffled knn trajectories (including query trajectory) and create augmented trajectories
            
            traj_embeddings = torch.tensor(get_traj_embeddings(self.data, labels=None, verbose=False, seed=seed))
            traj_pw_distmat = torch.cdist(traj_embeddings, traj_embeddings, p=2)

            topknn_traj_indices = traj_pw_distmat.topk(k, largest=False)[1]
            topknn_traj_indices = torch.tensor([list(topknn_traj_indices[idx, :][torch.randperm(k)].numpy()) for idx in range(0,topknn_traj_indices.shape[0])])
            
            self.aug_data = get_augmented_trajectories(self.data, self.data[0].shape[1], len(traj_step_features), topknn_traj_indices, k)
        
        self.size=len(self.data)
        self.mode=mode
        self.details=[file_id,mode,submode,sc,k,seed]
        self.anomalies=np.sum(self.labels)
        self.normals=self.size-self.anomalies
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):   
        if self.k is None: 
            # default
            return self.data[idx], self.labels[idx]
        else:
            # #MainTUL Comparison
            return self.data[idx], self.labels[idx], self.data[idx].shape[0], self.aug_data[idx], self.aug_data[idx].shape[0]
