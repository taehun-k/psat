import torch
from tqdm.notebook import tqdm
import sys
import wandb
import random
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.autograd as autograd
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from model.forward import AttnFWD, BaseFWD
import os
from torch.utils.data import TensorDataset
from utils import weights_init, weights_init_orthogonal
from model.inverse import PSAAT, apply_spectral_norm
import functools
import torch.nn as nn
import torch.optim as optim
from data.custom import masstiff_metasurface
from accelerate import Accelerator
# from progress.bar import IncrementalBar
torch.set_float32_matmul_precision('highest')

if __name__ == "train":
    from tqdm import tqdm
else: 
    from tqdm.notebook import tqdm

params = {'axes.labelsize' : 10,
         'axes.titlesize' : 12,
         'xtick.labelsize' : 10,
         'ytick.labelsize' : 10,
         'legend.fontsize' : 10,
         'font.family' : 'Times New Roman'}
plt.rcParams.update(params)

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def preserve_rng_state(func):
    """
    PyTorch의 CPU 및 CUDA RNG 상태를 함수의 실행 전후로 보존하는 데코레이터.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # --- 함수 실행 전: RNG 상태 저장 ---
        cpu_rng_state = torch.get_rng_state()
        
        gpu_rng_states = None
        if torch.cuda.is_available():
            # 모든 GPU의 RNG 상태를 저장
            gpu_rng_states = torch.cuda.get_rng_state_all()

        try:
            # --- 원본 함수 실행 ---
            result = func(*args, **kwargs)
            return result
        finally:
            # --- 함수 실행 후: RNG 상태 복원 ---
            # try...finally 구문을 사용하여 함수 실행 중 오류가 발생해도
            # 반드시 RNG 상태가 복원되도록 보장합니다.
            torch.set_rng_state(cpu_rng_state)
            
            if torch.cuda.is_available() and gpu_rng_states is not None:
                torch.cuda.set_rng_state_all(gpu_rng_states)
                
    return wrapper

def random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    if device == 'cuda': torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    elif device == 'mps': torch.mps.manual_seed(seed);
    torch.manual_seed(seed)

class Trainer:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.get_dataloader()
        self.build_model()
        self.get_optimizer_scheduler()
        self.get_criterion()
    
    def get_dataloader(self):
        print('Loading data...')
        self.train_ds = masstiff_metasurface(opt = self.opt, split='train')
        self.valid_ds = masstiff_metasurface(opt = self.opt, split='valid')
        self.test_ds = masstiff_metasurface(opt = self.opt, split='test')
        
        self.train_dl = DataLoader(self.train_ds, batch_size = self.opt.train_cfg['batch_size'], num_workers = 4, persistent_workers = True, pin_memory = True, shuffle = True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = self.opt.data_cfg['valid']['n_data'], num_workers = 1, persistent_workers = True, pin_memory = True, shuffle = False)
        self.test_dl = DataLoader(self.test_ds, batch_size = self.opt.data_cfg['test']['n_data'], num_workers = 1, persistent_workers = True, pin_memory = True, shuffle = False)
        
    def build_model(self):

        print('\nInitializing model...')
        # self.psaat = PSAAT(opt=self.opt)
        # self.FWD = AttnFWD(opt=self.opt)
        self.psaat = PSAAT(opt=self.opt).to(self.opt.device)
        self.FWD = AttnFWD(opt=self.opt).to(self.opt.device)
        
        print('\nInitializing best models...')
        self.best_psaat = PSAAT(opt=self.opt).to(self.opt.device)
        self.best_FWD = AttnFWD(opt=self.opt).to(self.opt.device)

    def get_optimizer_scheduler(self):
        self.opt_PSAT = optim.Adam(self.psaat.PSAT.parameters(), lr=self.opt.model_cfg['psat']['lr'], betas = tuple(self.opt.model_cfg['psat']['betas']))
        self.opt_D = optim.Adam(self.psaat.D.parameters(), lr=self.opt.model_cfg['discriminator']['lr'], betas = tuple(self.opt.model_cfg['discriminator']['betas']))
        self.opt_FWD = optim.Adam(self.FWD.parameters(), lr=self.opt.model_cfg['forward']['lr'])
        self.sch_PSAT, self.sch_D = None, None
        # self.sch_FWD = None
        self.sch_FWD = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt_FWD, T_0=50, T_mult=2, eta_min=1e-6)
    
    def get_criterion(self):
        self.criterion_PSAAT = nn.BCEWithLogitsLoss()
        self.criterion_FWD = nn.MSELoss(reduction = 'sum')

    def train(self):
        
        torch.cuda.empty_cache()
        random_seed(self.opt.seed)
        self.FWD.apply(weights_init)
        self.psaat.apply(weights_init_orthogonal)

        PSAT_loss, D_loss = [], []
        FWD_valid_loss = []
        [self.D_in_cond1, self.D_in_cond2, self.D_in_cond3], [self.PSAT_in_cond1, self.PSAT_in_cond2] = self.define_condition()
        pbar = tqdm(range(1, self.opt.train_cfg.n_epochs+1), desc = 'Train  ', unit = 'epoch', ncols = 150, position = 0, leave = True)
        for epoch in pbar:
            # Train PSAT & FWD
            PSAT_loss_epoch, D_loss_epoch, FWD_loss_epoch = self.psaat_fwd_train_epoch(epoch)
            PSAT_loss.append(PSAT_loss_epoch); D_loss.append(D_loss_epoch);
            if self.sch_FWD: self.sch_FWD.step();
            if self.sch_PSAT: self.sch_PSAT.step();
            if self.sch_D: self.sch_D.step();

            # Validate FWD
            FWD_val_loss = self.eval_fwd()
            FWD_valid_loss.append(FWD_val_loss)
            pbar.set_postfix({'PSAT': f'{PSAT_loss[-1]:.2e}', 'D': f'{D_loss[-1]:.2e}', 'FWD(train)': f'{FWD_loss_epoch:.2e}', 'FWD(valid)': f'{FWD_valid_loss[-1]:.2e}'})
            if epoch == 1: fwd_gen_min = float('inf')
            if epoch >= 2:
                fwd_gen_min = self.save_best_models(epoch = epoch, FWD_valid_loss = FWD_valid_loss, fwd_gen_min = fwd_gen_min, loss = [PSAT_loss, D_loss])

        print('Best model checkpoint:', self.opt.model_cfg.save_path)
        return {'PSAT loss': PSAT_loss, 'D loss': D_loss, 'FWD loss': FWD_valid_loss}

    @torch.no_grad()
    def save_best_models(self, epoch, FWD_valid_loss, fwd_gen_min, loss):
        self.psaat.eval(); self.FWD.eval(); self.best_psaat.eval(); self.best_FWD.eval()
        PSAT_loss, D_loss = loss
        
        if FWD_valid_loss[-1] <= min(FWD_valid_loss[:-1]):
            self.best_FWD.load_state_dict(self.FWD.state_dict())
            self.save_checkpoint(path = self.opt.model_cfg.save_path)
            fwd_gen_min = self.eval_gen_samples(model = [self.best_psaat, self.best_FWD])
        fwd_gen_tmp = self.eval_gen_samples(model = [self.psaat, self.best_FWD])
        if (fwd_gen_min != float('inf')) and (fwd_gen_tmp <= fwd_gen_min):
            self.best_psaat.load_state_dict(self.psaat.state_dict())
            self.save_checkpoint(path = self.opt.model_cfg.save_path)
            fwd_gen_min = deepcopy(fwd_gen_tmp)
        return fwd_gen_min

    def psaat_fwd_train_epoch(self, epoch):
        self.psaat.train(); self.FWD.train();
        PSAT_loss_epoch, D_loss_epoch = 0, 0
        FWD_loss_epoch = 0;

        pbar = tqdm(self.train_dl, desc = f'Epoch {epoch}', unit = 'it', ncols = 150, position = 1, leave = False)
        for n_batch, (X, T, _, B) in enumerate(pbar):
            # n_data = X.shape[0]
            X, T, B = X.to(self.device), T.to(self.device), B.to(self.device)
        # for _, (X, T, _, B) in enumerate(self.train_dl):
            # Train FWD
            FWD_loss_epoch = self.train_fwd(data = [X, T], FWD_loss_epoch = FWD_loss_epoch)
            # Construct real and fake samples
            real, fake = self.construct_real_fake_samples(data = [X, T, B])
            # Train discriminator
            for _ in range(self.opt.model_cfg['discriminator']['n_critic']):
                D_loss_epoch = self.train_discriminator(data = [real, fake, B], D_loss_epoch = D_loss_epoch)
            # Train psat
            PSAT_loss_epoch = self.train_psat(data = [fake, B], PSAT_loss_epoch = PSAT_loss_epoch)
            pbar.set_postfix({'PSAT': f'{(PSAT_loss_epoch / (n_batch+1)):.2e}', 'D': f'{(D_loss_epoch / (n_batch+1)):.2e}', 'FWD (train)': f'{(FWD_loss_epoch / (X.shape[0]*(n_batch+1)*2)):.2e}'})

        FWD_loss_epoch = FWD_loss_epoch / (len(self.train_dl.dataset) * 2)
        PSAT_loss_epoch = PSAT_loss_epoch / len(self.train_dl)
        D_loss_epoch = D_loss_epoch / (len(self.train_dl) * self.opt.model_cfg['discriminator']['n_critic'])

        return PSAT_loss_epoch, D_loss_epoch, FWD_loss_epoch

    def define_condition(self):
        D_in_cond1 = self.opt.model_cfg['discriminator']['input_dim'] == 2
        D_in_cond2 = (int(self.opt.data_cfg.n_design_var) == 8) and (self.opt.model_cfg['discriminator']['input_dim'] == 2 + int(self.opt.data_cfg.n_design_var))
        D_in_cond3 = (int(self.opt.data_cfg.n_design_var) != 8) and (self.opt.model_cfg['discriminator']['input_dim'] == 2+8) # target response에 generator idx + fixed idx 다 들어가는 경우
        
        PSAT_in_cond1 = self.opt.model_cfg['psat']['input_dim'] == 2
        if int(self.opt.data_cfg.n_design_var) == 8: PSAT_in_cond2 = self.opt.model_cfg['psat']['input_dim'] == 2 + self.opt.model_cfg['psat']['output_dim']
        else: PSAT_in_cond2 = self.opt.model_cfg['psat']['input_dim'] == 2 + (8-self.opt.model_cfg['psat']['output_dim'])
        return [D_in_cond1, D_in_cond2, D_in_cond3], [PSAT_in_cond1, PSAT_in_cond2]

    def construct_real_fake_samples(self, data):
        X, T, B = data
        z = torch.normal(mean=0, std=1, size=(X.shape[0], self.opt.model_cfg.psat.latent_dim), device = self.device)

        if self.PSAT_in_cond1: # generator input으로 target response만 들어가는 경우
            fake = torch.empty(2, X.shape[0], len(self.opt.gen_idx), device = self.device)
            fake[0], fake[1] = self.psaat.PSAT(x = T, z = z, b = B)
        elif self.PSAT_in_cond2: # generator input으로 target response + fixed idx 들어가는 경우
            if int(self.opt.data_cfg.n_design_var) == 8:
                X_mean = torch.tensor([0.6/2, 4.1/2, 5/2, 9/2, 2.1/2, 4.1/2, 8.1/2, 1.6/2], device = self.device).repeat(X.shape[0], 1)
                X_fake = torch.cat([X_mean, T], dim = 1) # X_fake는 generator input으로 들어가는 설계 변수 + target response
            else:
                X_fake = torch.cat([X[:,self.opt.fixed_idx], T], dim = 1) # X_fake는 generator input으로 들어가는 설계 변수 + target response
            fake = torch.empty(2, X.shape[0], len(self.opt.gen_idx), device = self.device)
            fake[0], fake[1] = self.psaat.PSAT(x = X_fake, z = z, b = B)
        
        if self.D_in_cond1: # discriminator 입력으로 생성 설계 변수만 들어가는 경우
            real = X[:,self.opt.gen_idx]
        # elif cond2:
        elif self.D_in_cond2: # discriminator 입력으로 target response도 들어가는 경우
            fake = torch.cat([fake, T.repeat(repeats = (2, 1, 1))], dim = 2)
            real = torch.cat([X[:,self.opt.gen_idx], T], dim = 1)
        elif self.D_in_cond3: # gen_idx는 생성한 설계 변수, 나머지 idx는 fixed_idx로 실제 설계 변수가 discriminator 입력으로 들어가는 경우
            fake_tmp = X.repeat(repeats = (2,1,1)).clone()
            fake_tmp[0,:,self.opt.gen_idx], fake_tmp[1,:,self.opt.gen_idx] = fake
            fake = torch.cat([fake_tmp, T.repeat(repeats = (2, 1, 1))], dim = 2)
            real = torch.cat([X, T], dim = 1)
        return real, fake

    def train_fwd(self, data, FWD_loss_epoch):
        self.FWD.train()
        X, T = data
        self.opt_FWD.zero_grad()
        pred_T = self.FWD(X)
        FWD_loss_batch = self.criterion_FWD(pred_T, T)
        # self.accelerator.backward(FWD_loss_batch)
        FWD_loss_batch.backward()
        self.opt_FWD.step()
        FWD_loss_epoch = FWD_loss_epoch + FWD_loss_batch.item()
        return FWD_loss_epoch

    def train_discriminator(self, data, D_loss_epoch):
        self.psaat.D.train()
        real, fake, B = data
        self.opt_D.zero_grad()
        if 'wgan' in self.opt.model_cfg.name.inv:
            D_loss_adv = -self.psaat.D(real, B).mean() + self.psaat.D(fake.detach()[1], B).mean()
            if 'gp' in self.opt.model_cfg.name.inv:
                gradient_penalty = self.compute_gradient_penalty(real = real, fake = fake[1].detach(), B = B)
                D_loss_batch = D_loss_adv + self.opt.lambda_gp * gradient_penalty
            else:
                D_loss_batch = D_loss_adv
        else:
            D_loss_batch = self.criterion_PSAAT(self.psaat.D(real, B), torch.ones((real.shape[0], 1), device = self.device)) + \
                            self.criterion_PSAAT(self.psaat.D(fake.detach()[1], B), torch.zeros((fake[1].shape[0], 1), device = self.device))
        D_loss_batch.backward()
        # self.accelerator.backward(D_loss_batch)
        self.opt_D.step()
        D_loss_epoch = D_loss_epoch + D_loss_batch.item()
        return D_loss_epoch

    def train_psat(self, data, PSAT_loss_epoch):
        self.psaat.PSAT.train()
        fake, B = data
        self.opt_PSAT.zero_grad()
        if 'wgan' in self.opt.model_cfg.name.inv:
            PSAT_loss_batch = -self.psaat.D(fake[1], B).mean()
        else:
            PSAT_loss_batch = self.criterion_PSAAT(self.psaat.D(fake[1], B), torch.ones((fake[1].shape[0], 1), device = self.device))
        PSAT_loss_batch.backward()
        # self.accelerator.backward(PSAT_loss_batch)
        self.opt_PSAT.step()
        PSAT_loss_epoch = PSAT_loss_epoch + PSAT_loss_batch.item()
        return PSAT_loss_epoch

    @torch.no_grad()
    def eval_fwd(self):
        self.FWD.eval();
        valid_loss = 0
        for X_val, T_val, _, B_val in self.valid_dl:
            X_val, T_val = X_val.to(device), T_val.to(device)
            pred_val = self.FWD(X_val)
            val_loss = self.criterion_FWD(pred_val, T_val)
            valid_loss = valid_loss + val_loss.item()
        valid_loss = valid_loss / (len(self.valid_dl.dataset) * 2)
        return valid_loss

    def save_checkpoint(self, path):
        
        state = {
            'PSAT_state_dict': self.best_psaat.PSAT.state_dict(),
            'D_state_dict': self.best_psaat.D.state_dict(),
            'FWD_state_dict': self.best_FWD.state_dict(),
        }
        temp_path = path + '.tmp'
        try:
            torch.save(state, temp_path)
            os.replace(temp_path, path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @torch.no_grad()
    @preserve_rng_state
    def eval_gen_samples(self, model): # fwd
        psaat, FWD = model
        FWD.eval(); psaat.PSAT.eval()
        FWD_val_loss_epoch = 0; n_data_val = 0
        for X_val, T_val, _, B_val in self.valid_dl:
            X_val, T_val, B_val = X_val.to(self.device), T_val.to(self.device), B_val.to(self.device)
            z_val = torch.normal(mean=0, std=1, size=(X_val.shape[0], self.opt.model_cfg.psat.latent_dim), device = self.device)
            if self.PSAT_in_cond1:
                fake_val = torch.empty(2, X_val.shape[0], len(self.opt.gen_idx), device = self.device)
                fake_val[0], fake_val[1] = psaat.PSAT(x = T_val, z = z_val, b = B_val)
                FWD_val_loss = self.criterion_FWD(FWD(fake_val[1]), T_val)
            elif self.PSAT_in_cond2:
                if int(self.opt.data_cfg.n_design_var) == 8:
                    X_mean_val = torch.tensor([0.6/2, 4.1/2, 5/2, 9/2, 2.1/2, 4.1/2, 8.1/2, 1.6/2], device = self.device).repeat(X_val.shape[0], 1)
                    X_fake_val = torch.cat([X_mean_val, T_val], dim = 1) # X_fake는 generator input으로 들어가는 설계 변수 + target response
                else:
                    X_fake_val = torch.cat([X_val[:,self.opt.fixed_idx], T_val], dim = 1) # X_fake는 generator input으로 들어가는 설계 변수 + target response
                fake_val_tmp = torch.empty(2, X_val.shape[0], len(self.opt.gen_idx), device = self.device)
                fake_val_tmp[0], fake_val_tmp[1] = psaat.PSAT(x = X_fake_val, z = z_val, b = B_val)
                fake_val = X_val.clone()
                fake_val[:,self.opt.gen_idx] = fake_val_tmp[1]
                FWD_val_loss = self.criterion_FWD(FWD(fake_val), T_val)
            n_data_val = n_data_val + X_val.shape[0]
            FWD_val_loss_epoch = FWD_val_loss_epoch + FWD_val_loss.item()
        return FWD_val_loss_epoch / len(self.valid_dl)

class Trainer_FWD:
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.get_dataloader()
        self.build_model()
        self.get_optimizer_scheduler()
        self.get_criterion()
    
    def get_dataloader(self):
        print('Loading data...')
        self.train_ds = masstiff_metasurface(opt = self.opt, split='train')
        self.valid_ds = masstiff_metasurface(opt = self.opt, split='valid')
        self.test_ds = masstiff_metasurface(opt = self.opt, split='test')
        
        self.train_dl = DataLoader(self.train_ds, batch_size = self.opt.train_cfg['batch_size'], num_workers = 4, persistent_workers = True, pin_memory = True, shuffle = True)
        self.valid_dl = DataLoader(self.valid_ds, batch_size = self.opt.data_cfg['valid']['n_data'], num_workers = 1, persistent_workers = True, pin_memory = True, shuffle = False)
        self.test_dl = DataLoader(self.test_ds, batch_size = self.opt.data_cfg['test']['n_data'], num_workers = 1, persistent_workers = True, pin_memory = True, shuffle = False)

    def build_model(self):

        print('\nInitializing models...')
        if self.opt.model_cfg.name.fwd.lower() == 'attn_fwd': 
            self.FWD = AttnFWD(opt = self.opt).to(self.device)
            self.best_FWD = AttnFWD(opt = self.opt).to(self.device)
        elif self.opt.model_cfg.name.fwd.lower() == 'base_fwd': 
            self.FWD = BaseFWD(opt = self.opt).to(self.device)
            self.best_FWD = BaseFWD(opt = self.opt).to(self.device)
    
    def get_optimizer_scheduler(self):
        self.opt_FWD = optim.Adam(self.FWD.parameters(), lr=self.opt.model_cfg['forward']['lr'])
        self.sch_FWD = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.opt_FWD, T_0=50, T_mult=2, eta_min=1e-6)
    
    def get_criterion(self):
        self.criterion_FWD = nn.MSELoss(reduction = 'sum')

    def train(self):
        torch.cuda.empty_cache()
        random_seed(self.opt.seed)
        self.FWD.apply(weights_init)

        FWD_valid_loss = [];
        pbar = tqdm(range(1, self.opt.train_cfg.n_epochs+1), desc = 'Train  ', unit = 'epoch', ncols = 150, position = 0, leave = True)
        for epoch in pbar:
            FWD_loss_epoch = self.fwd_train_epoch(epoch)
            self.sch_FWD.step()
            
            FWD_val_loss = self.eval_fwd()
            FWD_valid_loss.append(FWD_val_loss)
            pbar.set_postfix({'FWD|train': f'{FWD_loss_epoch:.2e}', 'FWD|valid':f'{FWD_valid_loss[-1]:.2e}'})
            if epoch >= 2:
                self.save_best_models(epoch = epoch, loss = FWD_valid_loss)

        print('Best model checkpoint:', self.opt.model_cfg.save_path)
        return {f'{self.opt.model_cfg.name.fwd} loss': FWD_valid_loss}

    @torch.no_grad()
    def save_best_models(self, epoch, loss):
        self.FWD.eval(); self.best_FWD.eval()
        FWD_valid_loss = loss
        if FWD_valid_loss[-1] <= min(FWD_valid_loss[:-1]):
            self.best_FWD.load_state_dict(self.FWD.state_dict())
            
        self.save_checkpoint(path = self.opt.model_cfg.save_path)
        if __name__ != "train": print(f'FWD Saved. Epoch: {epoch} | FWD (valid): {FWD_valid_loss[-1]:.2e}')

    def fwd_train_epoch(self, epoch):
        self.FWD.train()
        FWD_loss_epoch = 0

        pbar = tqdm(self.train_dl, desc = f'Epoch {epoch}', unit = 'it', ncols = 150, position = 1, leave = False)
        for n_batch, (X, T, _, B) in enumerate(pbar):
            # n_data = X.shape[0]
            X, T = X.to(self.device), T.to(self.device)
            # Train FWD
            FWD_loss_epoch = self.train_fwd(data = [X, T], FWD_loss_epoch = FWD_loss_epoch)
            pbar.set_postfix({'FWD (train)': f'{(FWD_loss_epoch / (X.shape[0]*(n_batch+1)*2)):.2e}'})

        # FWD_loss_epoch = FWD_loss_epoch / len(self.train_dl)
        FWD_loss_epoch = FWD_loss_epoch / (len(self.train_dl.dataset) * 2)
        return FWD_loss_epoch

    def train_fwd(self, data, FWD_loss_epoch):
        self.FWD.train()
        X, T = data
        self.opt_FWD.zero_grad()
        pred_T = self.FWD(X)
        FWD_loss_batch = self.criterion_FWD(pred_T, T)
        FWD_loss_batch.backward()
        self.opt_FWD.step()
        
        return FWD_loss_epoch + FWD_loss_batch.item()

    @torch.no_grad()
    def eval_fwd(self):
        self.FWD.eval()
        valid_loss = 0
        for X_val, T_val, _, B_val in self.valid_dl:
            X_val, T_val = X_val.to(self.device), T_val.to(self.device)
            pred_val = self.FWD(X_val)
            val_loss = self.criterion_FWD(pred_val, T_val)
            valid_loss = valid_loss + val_loss.item()
        valid_loss = valid_loss / (len(self.valid_dl.dataset) * 2)
        return valid_loss

    def save_checkpoint(self, path):
        state = {
            'FWD_state_dict': self.best_FWD.state_dict()
        }
        temp_path = path + '.tmp'
        try:
            torch.save(state, temp_path)
            os.replace(temp_path, path)
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)