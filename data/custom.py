import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas
import os
from pandas import DataFrame
import seaborn as sns
from scipy.stats import pearsonr
from abc import ABCMeta, abstractmethod
from glob import glob
import random
from sklearn.neighbors import KernelDensity
# from progress.bar import IncrementalBar

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# root_dir = r'C:\Users\Taehun\OneDrive\SNU\HAI\2. Code\1. Research\99. Dataset'

class AbstractDataset(Dataset):
    __metaclass__ = ABCMeta

    @abstractmethod
    def prepare_data(self):
        pass
    
class masstiff_metasurface(AbstractDataset):
    def __init__(self, opt, split:str = 'train'):
        self.current_dir = os.getcwd()
        # os.chdir(root_dir)
        torch.cuda.empty_cache()
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.manual_seed(opt.seed)
        self.opt = opt
        self.split = split
        self.design_var_path = opt.data_cfg['design_var_path']
        self.response_path = opt.data_cfg['response_path']
        # self.design_var_dir = glob(rf"./data/dataset/{opt.data_cfg[split]['n_design_var']}_design_variables/Design_Variable*.mat")
        # self.response_dir = glob(rf"./data/dataset/{opt.data_cfg[split]['n_design_var']}_design_variables/Metasurface_Data*.mat")
        self.X, self.T, self.R, self.B = self._prepare_data()
        print(f'[{split} dataset] X: {self.X.shape}, T: {self.T.shape}, R: {self.R.shape}, B: {self.B.shape}')
        os.chdir(self.current_dir)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X_torch = torch.tensor(self.X[idx], dtype=torch.float32).clone()
        T_torch = torch.tensor(self.T[idx], dtype=torch.float32).clone()
        # P_torch = torch.tensor(self.P[idx], dtype=torch.float32).to(device).clone()
        R_torch = torch.tensor(self.R[idx], dtype=torch.float32).clone()
        B_torch = torch.tensor(self.B[idx], dtype=torch.float32).clone()
        return X_torch, T_torch, R_torch, B_torch
    
    def _return_idx(self, low, high):
        init_list = np.random.choice(np.arange(low, high), self.opt.data_cfg[self.split]['n_data'], replace = False).tolist()
        return init_list
    
    def _prepare_data(self):
        # Var, Tr, Ph, Re = [], [], [], []
        Var, Tr_real, Tr_imag, Re = [], [], [], []
        for i in range(len(self.response_path)):
            response = scipy.io.loadmat(self.response_path[i])
            n_data = response['transmission'].shape[0]
            design_var = scipy.io.loadmat(self.design_var_path[i])
            Tr_mask = np.squeeze(np.square(np.abs(response['transmission']))) >= self.opt.data_cfg['T_threshold']
            Var.extend(design_var['design_variables'][:n_data][Tr_mask])
            # Var.extend(design_var['design_variables'][Tr_mask])
            Tr_real.extend(response['transmission'][Tr_mask].real[:,0])
            Tr_imag.extend(response['transmission'][Tr_mask].imag[:,0])
            Re.extend(np.abs(response['reflection'][Tr_mask]))
        L = len(Var)
        
        if self.split == 'train':
            init_list = self._return_idx(low = 0, high = int(0.7*L))
        elif self.split == 'valid':
            init_list = self._return_idx(low = int(0.7*L), high = int(0.85*L))
        elif self.split == 'test':
            init_list = self._return_idx(low = int(0.85*L), high = L)
        
        X, T, R, B = [], [], [], [] # B: bandgap
        Tr = np.squeeze(np.square(np.abs(response['transmission'][Tr_mask])))
        for init in init_list[:self.opt.data_cfg[self.split]['n_data']]:
            X.append(Var[init])
            T.append([Tr_real[init], Tr_imag[init]])
            # P.append(Ph[init])
            R.append(Re[init])
            # if Tr[init] < 0.1: B.append([1, 0, 0])
            # elif (Tr[init] < 0.9) and (Tr[init] >= 0.1): B.append([0, 1, 0])
            # elif Tr[init] >= 0.9: B.append([0, 0, 1])
            
            # if Tr[init] >= 0.9: B.append([1])
            # else: B.append([Tr[init]-0.9])
            
            threshold = 0.9
            if Tr[init] >= threshold:
                B.append([(Tr[init] - threshold)/(1-threshold)])
            else:
                B.append([Tr[init]-threshold])
            # B.append([Tr[init]-0.9])
            
            # if Tr[init] >= 0.9: B.append([0,1])
            # else: B.append([Tr[init]-0.9, 0])
        return np.array(X), np.array(T), np.array(R), np.array(B)

if __name__ == "__main__":
    import sys
    import os
    import yaml
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.join(current_dir, '..')
    sys.path.append(parent_dir)
    from arguments import get_args
    opt = get_args()
    opt.data_cfg = yaml.load(open(parent_dir + r'/config/psaat_gen_l2l5.yaml'), Loader=yaml.FullLoader)['data']
    train_ds = masstiff_metasurface(opt, split = 'train')