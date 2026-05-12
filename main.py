from torchinfo import summary
import matplotlib.pyplot as plt
# from ptflops import get_model_complexity_info
from train import Trainer_FWD, Trainer
from arguments import get_args
from utils import plot_learning_curve
import os
from pathlib import Path
import pickle
import shutil
import torch
from test import *
import numpy as np
import hydra
from hydra.core.hydra_config import HydraConfig
from torch.utils.data import TensorDataset
from copy import deepcopy
torch.set_float32_matmul_precision('highest')

history = []

@hydra.main(config_path = 'config', version_base = '1.2')
def run(config):
    opt = get_args()
    opt.model_cfg = config.model
    opt.data_cfg = config.data
    opt.train_cfg = config.train
    opt.gen_idx = config.gen_idx
    opt.gen_variables = config.gen_variables
    opt.fixed_idx = list({i for i in range(8)} - set(opt.gen_idx))
    opt.material['freq'] = int(config.freq[:-1])
    opt.material['k0'] = 2*np.pi*opt.material['freq']*1e3 / np.sqrt(opt.material['E']/(1-opt.material['poisson']**2)/opt.material['rho'])
    opt.material['c_p'] = np.sqrt(opt.material['E'] / (opt.material['rho']*(1-opt.material['poisson']**2))) # phase velocity
    opt.material['lamb'] = opt.material['c_p'] / opt.material['freq'] # wavelength
    opt.material['h'] = 0.19*opt.material['lamb'] # mesh size

    opt.output_dir = HydraConfig.get().runtime.output_dir
    print(f'output dir:', opt.output_dir)
    train_mode = input("(a) model-spec, (b) fwd-train, (c) fwd-test, (d) inv-train, (e) inv-test, (f) generate, (g) comsol: ")
    
    if train_mode.lower() == 'model-spec' or train_mode.lower() == 'a' or train_mode.lower() == '(a)':
        if 'grad' in opt.model_cfg.name.inv:
            opt.model_cfg.name.fwd = 'attn_fwd'
            trainer_attn = Trainer_FWD(deepcopy(opt))
            opt.model_cfg.name.fwd = 'base_fwd'
            trainer_base = Trainer_FWD(deepcopy(opt))
            stats_attn_fwd = summary(trainer_attn.FWD, input_size=[(1, opt.model_cfg.forward.input_dim)], device = trainer_attn.opt.device)
            stats_base_fwd = summary(trainer_base.FWD, input_size=[(1, opt.model_cfg.forward.input_dim)], device = trainer_base.opt.device)
        else:
            trainer = Trainer(opt)
            stats_psat = summary(trainer.psaat.PSAT, input_size=[(1, opt.model_cfg.psat.input_dim), (1, opt.model_cfg.psat.latent_dim), (1, 1)], device = trainer.opt.device)
            stats_d = summary(trainer.psaat.D, input_size=[(1, opt.model_cfg.discriminator.input_dim), (1, 1)], device = trainer.opt.device)
            stats_fwd = summary(trainer.FWD, input_size=[(1, opt.model_cfg.forward.input_dim)], device = trainer.opt.device)
            print(f'FLOPs - PSAT: {stats_psat.total_mult_adds}, D: {stats_d.total_mult_adds}, FWD: {stats_fwd.total_mult_adds}')
    
    elif train_mode.lower() == 'fwd-train' or train_mode.lower() == 'b' or train_mode.lower() == '(b)':
        opt.model_cfg.name.fwd = 'attn_fwd'
        opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.fwd}_gen_{opt.gen_variables}_{config.freq}.pt' # Best model in Ubuntu
        # opt.model_cfg.save_path = opt.model_cfg.save_dir + 'best_psaat_gen_l2l5_100k_draft.pt' # Model path denoted in manuscript
        trainer_attn = Trainer_FWD(deepcopy(opt))
        
        opt.model_cfg.name.fwd = 'base_fwd'
        opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.fwd}_gen_{opt.gen_variables}_{config.freq}.pt' # Best model in Ubuntu
        # opt.model_cfg.save_path = opt.model_cfg.save_dir + 'best_base_fwd_gen_l2l5_100k_draft.pt' # Model path denoted in manuscript
        trainer_base = Trainer_FWD(deepcopy(opt))
        
        os.makedirs('./experiments/learning_curve/', exist_ok=True)
        os.makedirs('./experiments/saved_model/', exist_ok=True)
        model_type = input('Select the forward model type for training (a) AttnFWD, (b) BaseFWD, (c) Both: ')
        hist_attn = {}; hist_base = {}
        if model_type.lower() == 'attnfwd' or model_type.lower() == 'a' or model_type.lower() == '(a)': hist_attn = trainer_attn.train()
        elif model_type.lower() == 'basefwd' or model_type.lower() == 'b' or model_type.lower() == '(b)': hist_base = trainer_base.train()
        elif model_type.lower() == 'both' or model_type.lower() == 'c' or model_type.lower() == '(c)':
            hist_attn = trainer_attn.train()
            hist_base = trainer_base.train()
        hist = hist_attn | hist_base
        history.append(hist)
        plot_learning_curve(hist, save_path = f'./experiments/learning_curve/fwd_gen_{opt.gen_variables}.png', xlabel = 'Epochs', ylabel = 'Loss')
        if len(history) != 0: print(history, file = open(f'./experiments/learning_curve/training_history_fwd_gen_{opt.gen_variables}.txt', 'w'))
        
    elif train_mode.lower() == 'fwd-test' or train_mode.lower() == 'c' or train_mode.lower() == '(c)':
        opt.model_cfg.name.fwd = 'attn_fwd'
        opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.fwd}_gen_{opt.gen_variables}_{config.freq}.pt' # Best model in Ubuntu
        # opt.model_cfg.save_path = opt.model_cfg.save_dir + 'best_psaat_gen_l2l5_100k_draft.pt' # Model path denoted in manuscript
        trainer_attn = Trainer_FWD(deepcopy(opt))
        
        opt.model_cfg.name.fwd = 'base_fwd'
        opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.fwd}_gen_{opt.gen_variables}_{config.freq}.pt' # Best model in Ubuntu
        # opt.model_cfg.save_path = opt.model_cfg.save_dir + 'best_base_fwd_gen_l2l5_100k_draft.pt' # Model path denoted in manuscript
        trainer_base = Trainer_FWD(deepcopy(opt))
        
        best_AttnFWD = trainer_attn.best_FWD.to(opt.device)
        best_BaseFWD = trainer_base.best_FWD.to(opt.device)
        test_ds = TensorDataset(*[t.to(opt.device) for t in trainer_attn.test_ds[:]])
        ckpt_attn = torch.load(trainer_attn.opt.model_cfg.save_path, weights_only = True)
        ckpt_base = torch.load(trainer_base.opt.model_cfg.save_path, weights_only = True)
        best_AttnFWD.load_state_dict(ckpt_attn['FWD_state_dict'])
        # best_BaseFWD.load_state_dict(ckpt_base['model_state_dict'])
        best_BaseFWD.load_state_dict(ckpt_base['FWD_state_dict'])
        
        evaluate_fwd(model = [best_AttnFWD, best_BaseFWD], test_ds = test_ds)
        
    elif train_mode.lower() == 'inv-train' or train_mode.lower() == 'd' or train_mode.lower() == '(d)':
        opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.inv}_gen_{opt.gen_variables}_{config.freq}.pt'
        trainer = Trainer(opt)
        hist = trainer.train()
        history.append(hist)
        os.makedirs('./experiments/learning_curve/', exist_ok=True)
        os.makedirs('./experiments/saved_model/', exist_ok=True)
        keys_gen = ['PSAT loss', 'D loss']
        keys_fwd = ['FWD loss']
        plot_learning_curve({k: hist[k] for k in keys_gen}, save_path = f'./experiments/learning_curve/{opt.model_cfg.name.inv.lower()}_gen_{opt.gen_variables}.png', xlabel = 'Epochs', ylabel = 'Loss')
        plot_learning_curve({k: hist[k] for k in keys_fwd}, save_path = f'./experiments/learning_curve/{opt.model_cfg.name.fwd.lower()}_fwd_gen_{opt.gen_variables}.png', xlabel = 'Epochs', ylabel = 'Loss')
        
        if len(history) != 0: print(history, file = open(f'./experiments/learning_curve/training_history_psaat_gen_{opt.gen_variables}.txt', 'w'))
        
    elif train_mode.lower() == 'inv-test' or train_mode.lower() == 'e' or train_mode.lower() == '(e)':
        if 'grad' in opt.model_cfg.name.inv:
            opt.model_cfg.name.fwd = 'attn_fwd'
            opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.fwd}_gen_{opt.gen_variables}_{config.freq}.pt' # Best model in Ubuntu
            # opt.model_cfg.save_path = opt.model_cfg.save_dir + 'best_psaat_gen_l2l5_100k_draft.pt' # Model path denoted in manuscript
            trainer = Trainer_FWD(opt)
            ckpt_attn = torch.load(trainer.opt.model_cfg.save_path, weights_only = True)
            trainer.best_FWD.load_state_dict(ckpt_attn['FWD_state_dict'])
            if trainer.opt.model_cfg.optim.method == 'mma': mma_based_optim(trainer)
            else: grad_based_optim(trainer)
        
        else:
            opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.inv}_gen_{opt.gen_variables}_{config.freq}.pt' # Best model in Ubuntu
            # opt.model_cfg.save_path = opt.model_cfg.save_dir + '/best_psaat_gen_l2l5_100k_draft.pt'
            trainer = Trainer(opt)        
            trainer.best_psaat.PSAT, trainer.best_FWD = trainer.best_psaat.PSAT.to(opt.device), trainer.best_FWD.to(opt.device)
            trainer.test_ds = TensorDataset(*[t.to(opt.device) for t in trainer.test_ds[:]])
            
            ckpt = torch.load(opt.model_cfg.save_path, weights_only = True)
            trainer.best_psaat.PSAT.load_state_dict(ckpt['PSAT_state_dict'])
            trainer.best_FWD.load_state_dict(ckpt['FWD_state_dict'])
            
            evaluate_design_multiple_latent(model_name = f'{opt.model_cfg.name.inv}', \
                models = [trainer.best_psaat.PSAT, trainer.best_FWD], \
                test_ds = trainer.test_ds,\
                T_bound = [0.90, 1.01], P_bound = [-180, 180], N_repeat = 100)
            
            evaluate_design(model_name = f'{opt.model_cfg.name.inv}', \
                models = [trainer.best_psaat.PSAT, trainer.best_FWD], \
                test_ds = torch.utils.data.TensorDataset(*[trainer.test_ds[:][i].repeat(repeats = (5,1)) for i in range(len(trainer.test_ds[:]))]),\
                T_bound = [0.90, 1.01], P_bound = [-180, 180])
            
            physical_analysis_attention(trainer = trainer)
        
    elif train_mode.lower() == 'generate' or train_mode.lower() == 'f' or train_mode.lower() == '(f)':
        if 'grad' in opt.model_cfg.name.inv:
            opt.model_cfg.name.fwd = 'attn_fwd'
            opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.fwd}_gen_{opt.gen_variables}_{config.freq}.pt' # Best model in Ubuntu
            trainer = Trainer_FWD(opt)
            ckpt_attn = torch.load(trainer.opt.model_cfg.save_path, weights_only = True)
            trainer.best_FWD.load_state_dict(ckpt_attn['FWD_state_dict'])
        else:
            opt.model_cfg.save_path = opt.model_cfg.save_dir + f'best_{opt.model_cfg.name.inv}_gen_{opt.gen_variables}_{config.freq}.pt' # Best model in Ubuntu
            # opt.model_cfg.save_path = opt.model_cfg.save_dir + 'best_psaat_gen_l2l5_100k_draft.pt' # Model path denoted in manuscript
            trainer = Trainer(opt)       
            trainer.best_psaat.PSAT, trainer.best_FWD = trainer.best_psaat.PSAT.to(opt.device), trainer.best_FWD.to(opt.device)
            trainer.test_ds = TensorDataset(*[t.to(opt.device) for t in trainer.test_ds[:]])
            ckpt = torch.load(opt.model_cfg.save_path, weights_only = True)
            # breakpoint()
            trainer.best_psaat.PSAT.load_state_dict(ckpt['PSAT_state_dict'])
            trainer.best_FWD.load_state_dict(ckpt['FWD_state_dict'])

        gen_mode = input("Generate designs for refracting (r) / focusing (f): ")
        gen_mode = 'refracting' if gen_mode.lower() == 'r' else 'focusing' if gen_mode.lower() == 'f' else gen_mode.lower()
        if gen_mode.lower() == 'refracting':
            trainer.opt.refracted_angle = np.deg2rad(float(input('Target refracted angle (deg): ')))
            n_cells = int(input('Number of unit cell: '))
            if n_cells % 2 == 0: l_bound = -(n_cells//2 - 1); u_bound = (n_cells//2 + 1)
            else: l_bound = -(n_cells//2); u_bound = (n_cells//2) + 1
            y = np.arange(l_bound*opt.material['H'], u_bound*opt.material['H'], opt.material['H'])
            profile = refracting_phase_profile(refracted_angle = trainer.opt.refracted_angle, dy = opt.material['H'], k_t = opt.material['k0'], n_cells = n_cells)[:len(y)] # k_t는 transmitted된 물질이 달라질 경우 다르게 넣어야 함

        elif gen_mode.lower() == 'focusing':
            trainer.opt.material['f_l'] = float(input('Focal length (0.1, 0.15, 0.2, 0.3 [m]): '))
            n_cells = int(input('Number of unit cells: '))
            if n_cells % 2 == 0: l_bound = -(n_cells//2 - 1); u_bound = (n_cells//2 + 1)
            else: l_bound = -(n_cells//2); u_bound = (n_cells//2) + 1
            # y_arr = np.concatenate([np.arange(l_bound*opt.material['H'], 0, opt.material['H']), np.arange(0, u_bound*opt.material['H'], opt.material['H'])])
            y_arr = np.arange(l_bound*opt.material['H'], u_bound*opt.material['H'], opt.material['H'])
            profile = focusing_phase_profile(y = y_arr, k0 = opt.material['k0'], f_l = opt.material['f_l'])
        
        if 'grad' in opt.model_cfg.name.inv: 
            optimize_design_practical_case(trainer = trainer, profile = profile, type = gen_mode.lower())
        else:
            generate_design_practical_case(trainer = trainer, profile = profile, type = gen_mode.lower())
        
    elif train_mode.lower() == 'comsol' or train_mode.lower() == 'g' or train_mode.lower() == '(g)':
        if 'grad' in opt.model_cfg.name.inv: trainer = Trainer_FWD(opt)
        else: trainer = Trainer(opt)        
        comsol_mode = input("Generate designs for refracting (r) / focusing (f): ")
        comsol_mode = 'refracting' if comsol_mode.lower() == 'r' else 'focusing' if comsol_mode.lower() == 'f' else comsol_mode.lower()
        if comsol_mode.lower() == 'refracting':
            trainer.opt.refracted_angle = np.deg2rad(float(input('Target refracted angle (deg): ')))
            n_cells = int(input('Number of unit cells: '))
            if n_cells % 2 == 0: l_bound = -(n_cells//2 - 1); u_bound = (n_cells//2 + 1)
            else: l_bound = -(n_cells//2); u_bound = (n_cells//2) + 1
            y = np.arange(l_bound*opt.material['H'], u_bound*opt.material['H'], opt.material['H'])
            profile = refracting_phase_profile(refracted_angle = trainer.opt.refracted_angle, dy = opt.material['H'], k_t = opt.material['k0'], n_cells = n_cells)[:len(y)] # k_t는 transmitted된 물질이 달라질 경우 다르게 넣어야 함

        elif comsol_mode.lower() == 'focusing':
            trainer.opt.material['f_l'] = float(input('Focal length (0.1, 0.15, 0.2, 0.3 [m]): '))
            n_cells = int(input('Number of unit cells: '))
            if n_cells % 2 == 0: l_bound = -(n_cells//2 - 1); u_bound = (n_cells//2 + 1)
            else: l_bound = -(n_cells//2); u_bound = (n_cells//2) + 1
            # y_arr = np.concatenate([np.arange(l_bound*opt.material['H'], 0, opt.material['H']), np.arange(0, u_bound*opt.material['H'], opt.material['H'])])
            y_arr = np.arange(l_bound*opt.material['H'], u_bound*opt.material['H'], opt.material['H'])
            profile = focusing_phase_profile(y = y_arr, k0 = opt.material['k0'], f_l = opt.material['f_l'])

        evaluate_design_practical_case(trainer = trainer, profile = profile, type = comsol_mode.lower())
        
if __name__ == '__main__':
    run()
