from pyexpat import model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import os
import scipy.io as spio
import time
from matplotlib.ticker import AutoLocator
import random
import torch.optim as optim
import nlopt
import matplotlib.colors as mcolors

if __name__ == "test":
    from tqdm import tqdm
else: 
    from tqdm.notebook import tqdm

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    if device == 'cuda': torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    elif device == 'mps': torch.mps.manual_seed(seed)
    torch.manual_seed(seed)

@torch.no_grad()
def evaluate_fwd(model, test_ds):
    fwd, fwd_no_attn = model
    random_seed(fwd.opt.seed)
    fwd.eval(), fwd_no_attn.eval()
    T_comp = torch.complex(test_ds[:][1][:,0], test_ds[:][1][:,1])
    tr_true = torch.square(torch.abs(T_comp))
    tr_thre = 0.9
    tr_mask = tr_true >= tr_thre
    T_comp_pred_tmp = fwd(test_ds[:][0])
    T_comp_pred = torch.complex(T_comp_pred_tmp[:,0], T_comp_pred_tmp[:,1])
    tr_pred = torch.square(torch.abs(T_comp_pred))
    ph_true_tmp = torch.angle(T_comp)
    ph_pred_tmp = torch.angle(T_comp_pred)
    ph_true = torch.tensor([ph + 2*torch.pi if ph < 0 else ph for ph in ph_true_tmp], device = ph_true_tmp.device)
    ph_pred = torch.tensor([ph + 2*torch.pi if ph < 0 else ph for ph in ph_pred_tmp], device = ph_pred_tmp.device)

    T_error = torch.abs(T_comp_pred_tmp - test_ds[:][1])[tr_mask]
    tr_error = torch.abs(tr_pred - tr_true)[tr_mask]
    ph_error = torch.abs(ph_pred - ph_true)[tr_mask]
    T_l1 = torch.mean(T_error)
    tr_l1 = torch.mean(tr_error)
    ph_l1 = torch.mean(ph_error)

    print(f'\nForward Model Evaluation (Test Dataset)')
    print('-'*40, 'FWD (w/ Attn)', '-'*40)
    print('MAE of Transmission Coeff. | Transmittance | Phase Shift :', f'{T_l1:.2e} | {tr_l1:.2e} | {ph_l1:.2e}\n')

    print('Cumulative Ratio of Transmittance Error :', f'{(tr_error < 0.01).sum().item() / tr_error.shape[0]:.3f} (< 1%FS)',\
                                                         '|', f'{(tr_error < 0.001).sum().item() / tr_error.shape[0]:.3f} (< 0.1%FS)')
    print('Cumulative Ratio of Phase Shift Error   :', f'{(ph_error < (2*torch.pi*0.01)).sum().item() / ph_error.shape[0]:.3f} (< 1%FS)',\
                                                         '|', f'{(ph_error < (2*torch.pi*0.001)).sum().item() / ph_error.shape[0]:.3f} (< 0.1%FS)')

    xmin = test_ds[:][0][:,fwd.opt.gen_idx[0]].min().detach().cpu(); xmax = test_ds[:][0][:,fwd.opt.gen_idx[0]].max().detach().cpu()
    ymin = test_ds[:][0][:,fwd.opt.gen_idx[1]].min().detach().cpu(); ymax = test_ds[:][0][:,fwd.opt.gen_idx[1]].max().detach().cpu()
    dx = (xmax - xmin) / 3
    dy = (ymax - ymin) / 3

    fig1, ax1 = plt.subplots(1,1, figsize = (5*5/12*1.5, 5*5/12)); fig2, ax2 = plt.subplots(1,1, figsize = (5*5/12*1.5, 5*5/12))
    # t1 = ax1.scatter(test_ds[:][0][tr_mask,fwd.opt.gen_idx[0]].detach().cpu(), test_ds[:][0][tr_mask,fwd.opt.gen_idx[1]].detach().cpu(),\
    #                     c = tr_error.detach().cpu(), cmap = 'Blues', s = 20, edgecolor = 'k', linewidths= 0.2)
    # t2 = ax2.scatter(test_ds[:][0][tr_mask,fwd.opt.gen_idx[0]].detach().cpu(), test_ds[:][0][tr_mask,fwd.opt.gen_idx[1]].detach().cpu(),\
    #                     c = ph_error.detach().cpu()*180/np.pi, cmap = 'Reds', s = 20, edgecolor = 'k', linewidths= 0.2)
    t1 = ax1.scatter(test_ds[:][0][tr_mask,fwd.opt.gen_idx[0]].detach().cpu(), test_ds[:][0][tr_mask,fwd.opt.gen_idx[1]].detach().cpu(),\
                        c = tr_error.detach().cpu(), cmap = 'Blues', s = 10, alpha = 0.7)
    t2 = ax2.scatter(test_ds[:][0][tr_mask,fwd.opt.gen_idx[0]].detach().cpu(), test_ds[:][0][tr_mask,fwd.opt.gen_idx[1]].detach().cpu(),\
                        c = ph_error.detach().cpu()*180/np.pi, cmap = 'Reds', s = 10, alpha = 0.7)
    ax1.set_xticks(np.arange(xmin, xmax+dx-1e-8, dx)); ax2.set_xticks(np.arange(xmin, xmax+dx-1e-8, dx))
    # breakpoint()
    ax1.set_yticks(np.arange(ymin, ymax+dy-1e-8, dy)); ax2.set_yticks(np.arange(ymin, ymax+dy-1e-8, dy))
    ax1.set_xlim(xmin-0.2, xmax+0.2); ax2.set_xlim(xmin-0.2, xmax+0.2)
    ax1.set_ylim(ymin-0.1, ymax+0.1); ax2.set_ylim(ymin-0.1, ymax+0.1)
    t1.set_clim([0, 0.01]); t2.set_clim([0, 360*0.01])
    ax1.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
    ax2.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
    fig1.tight_layout(); fig2.tight_layout()
    os.makedirs('./experiments/fwd_evaluation', exist_ok=True)
    fig1.savefig(f'./experiments/fwd_evaluation/fwd_w_attn_gen_{fwd.opt.gen_variables}_transmittance_tr{tr_thre}.png', dpi = 500, transparent = True)
    fig2.savefig(f'./experiments/fwd_evaluation/fwd_w_attn_gen_{fwd.opt.gen_variables}_phase_shift_tr{tr_thre}.png', dpi = 500, transparent = True)
    plt.close(fig1)
    plt.close(fig2)

    print('-'*39, 'FWD (w/o Attn)', '-'*40)
    T_comp_pred_tmp = fwd_no_attn(test_ds[:][0], test_ds[:][3])
    T_comp_pred = torch.complex(T_comp_pred_tmp[:,0], T_comp_pred_tmp[:,1])
    tr_pred = torch.square(torch.abs(T_comp_pred))
    ph_pred_tmp = torch.angle(T_comp_pred)
    ph_pred = torch.tensor([ph + (2*torch.pi) if ph < 0 else ph for ph in ph_pred_tmp], device = ph_pred_tmp.device)
    T_error = torch.abs(T_comp_pred_tmp - test_ds[:][1])[tr_mask]
    tr_error = torch.abs(tr_pred - tr_true)[tr_mask]
    ph_error = torch.abs(ph_pred - ph_true)[tr_mask]
    T_l1 = torch.mean(T_error)
    tr_l1 = torch.mean(tr_error)
    ph_l1 = torch.mean(ph_error)
    print('MSE of Transmission Coeff. | Transmittance | Phase Shift :', f'{T_l1:.2e} | {tr_l1:.2e} | {ph_l1:.2e}\n')

    print('Cumulative Ratio of Transmittance Error :', f'{(tr_error < 0.01).sum().item() / tr_error.shape[0]:.3f} (< 1%FS)',\
                                                         '|', f'{(tr_error < 0.001).sum().item() / tr_error.shape[0]:.3f} (< 0.1%FS)')

    print('Cumulative Ratio of Phase Shift Error   :', f'{(ph_error < (2*torch.pi*0.01)).sum().item() / ph_error.shape[0]:.3f} (< 1%FS)',\
                                                         '|', f'{(ph_error < (2*torch.pi*0.001)).sum().item() / ph_error.shape[0]:.3f} (< 0.1%FS)')

    fig3, ax3 = plt.subplots(1,1, figsize = (5*5/12*1.5, 5*5/12)); fig4, ax4 = plt.subplots(1,1, figsize = (5*5/12*1.5, 5*5/12))
    t1 = ax3.scatter(test_ds[:][0][tr_mask,fwd.opt.gen_idx[0]].detach().cpu(), test_ds[:][0][tr_mask,fwd.opt.gen_idx[1]].detach().cpu(),\
                        c = tr_error.detach().cpu(), cmap = 'Blues', s = 10, alpha = 0.7)
    t2 = ax4.scatter(test_ds[:][0][tr_mask,fwd.opt.gen_idx[0]].detach().cpu(), test_ds[:][0][tr_mask,fwd.opt.gen_idx[1]].detach().cpu(),\
                        c = ph_error.detach().cpu()*180/np.pi, cmap = 'Reds', s = 10, alpha = 0.7)
    ax3.set_xticks(np.arange(xmin, xmax+dx-1e-8, dx)); ax4.set_xticks(np.arange(xmin, xmax+dx-1e-8, dx))
    ax3.set_yticks(np.arange(ymin, ymax+dy-1e-8, dy)); ax4.set_yticks(np.arange(ymin, ymax+dy-1e-8, dy))
    ax3.set_xlim(xmin-0.2, xmax+0.2); ax4.set_xlim(xmin-0.2, xmax+0.2)
    ax3.set_ylim(ymin-0.1, ymax+0.1); ax4.set_ylim(ymin-0.1, ymax+0.1)
    t1.set_clim([0, 0.01]); t2.set_clim([0, 360*0.01])
    ax3.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
    ax4.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
    fig3.tight_layout(); fig4.tight_layout()

    os.makedirs('./experiments/fwd_evaluation', exist_ok=True)
    fig3.savefig(f'./experiments/fwd_evaluation/fwd_wo_attn_gen_{fwd.opt.gen_variables}_transmittance_tr{tr_thre}.png', dpi = 500, transparent = True)
    plt.close(fig3)
    fig4.savefig(f'./experiments/fwd_evaluation/fwd_wo_attn_gen_{fwd.opt.gen_variables}_phase_shift_tr{tr_thre}.png', dpi = 500, transparent = True)
    plt.close(fig4)

@torch.no_grad()
def evaluate_design_multiple_latent(model_name: str, models, test_ds, T_bound=[0.9, 1], P_bound=[-50, -30], N_repeat=100):
    model, FWD = models
    random_seed(model.opt.seed)

    T_complex = torch.complex(test_ds[:][1][:, 0], test_ds[:][1][:, 1])
    tr_true = torch.square(torch.abs(T_complex))
    ph_true = torch.angle(T_complex)

    T_lower_mask = torch.square(torch.abs(T_complex)) > T_bound[0]
    T_upper_mask = torch.square(torch.abs(T_complex)) < T_bound[1]
    T_bound_mask = T_lower_mask & T_upper_mask

    P_lower_mask = torch.angle(T_complex) * 180 / torch.pi > P_bound[0]
    P_upper_mask = torch.angle(T_complex) * 180 / torch.pi < P_bound[1]
    P_bound_mask = P_lower_mask & P_upper_mask
    mask_idx = T_bound_mask & P_bound_mask

    X_gen_masked = test_ds[:][0][mask_idx]
    T_gen_masked = torch.square(torch.abs(T_complex[mask_idx])).detach().cpu()
    P_gen_masked = torch.angle(T_complex[mask_idx]).detach().cpu()

    X_ref_masked = test_ds[:][0][mask_idx]
    T_ref_masked = torch.square(torch.abs(T_complex[mask_idx])).detach().cpu()
    P_ref_masked = torch.angle(T_complex[mask_idx]).detach().cpu()

    model.eval()
    exp = deepcopy(model.opt.model_cfg['psat'])
    cond1 = exp['input_dim'] == 2
    if (exp['output_dim'] == 8):
        cond2 = exp['input_dim'] == 2 + exp['output_dim']
    else:
        cond2 = exp['input_dim'] == 2 + (8 - exp['output_dim'])

    N = X_gen_masked.shape[0]

    for i in range(2):
        fig1 = plt.figure(figsize=(5 * 5 / 12, 5 * 5 / 12))
        fig2 = plt.figure(figsize=(5 * 5 / 12, 5 * 5 / 12))
        ax1 = fig1.add_subplot(1, 1, 1, projection='3d')
        ax2 = fig2.add_subplot(1, 1, 1, projection='3d')

        with torch.no_grad():
            if i == 1:
                # ── N_repeat 번 생성 후 best 선택 ──────────────────
                best_X_pred = None
                best_losses = torch.full((N,), float('inf'), device=X_gen_masked.device)

                for r in range(N_repeat):
                    z = torch.normal(
                        mean=0, std=1,
                        size=(N, model.opt.model_cfg.psat.latent_dim),
                        device=X_gen_masked.device,
                    )

                    if cond1:
                        X_pred_r = model(x=test_ds[:][1][mask_idx], z=z)
                    elif cond2:
                        if exp['output_dim'] == 8:
                            X_mean = torch.tensor(
                                [0.6/2, 4.1/2, 5/2, 9/2, 2.1/2, 4.1/2, 8.1/2, 1.6/2],
                                device=X_gen_masked.device,
                            ).repeat(N, 1)
                            X_pred_r = model(
                                x=torch.cat([X_mean, test_ds[:][1][mask_idx]], dim=1),
                                z=z, b=test_ds[:][3][mask_idx],
                            )
                        else:
                            X_pred_r = model(
                                x=torch.cat([X_gen_masked[:, model.opt.fixed_idx], test_ds[:][1][mask_idx]], dim=1),
                                z=z, b=test_ds[:][3][mask_idx],
                            )

                    X_pred_r = X_pred_r[1]  # gen 변수만

                    # forward model로 평가
                    X_whole_r = torch.empty(N, X_ref_masked.shape[1], device=X_gen_masked.device)
                    X_whole_r[:, model.opt.fixed_idx] = X_ref_masked[:, model.opt.fixed_idx].to(X_gen_masked.device)
                    X_whole_r[:, model.opt.gen_idx] = X_pred_r

                    T_pred_r = FWD(X_whole_r)
                    # 샘플별 transmission coefficient loss
                    losses_r = nn.functional.mse_loss(
                        T_pred_r, test_ds[:][1][mask_idx], reduction='none',
                    ).mean(dim=1)  # (N,)

                    # 샘플별 best 갱신
                    improved = losses_r < best_losses
                    if improved.any():
                        best_losses[improved] = losses_r[improved]
                        if best_X_pred is None:
                            best_X_pred = X_pred_r.clone()
                        best_X_pred[improved] = X_pred_r[improved]

                X_pred = best_X_pred.detach().cpu()
                print(f'[N_repeat={N_repeat}] Best-of-{N_repeat} selection done.')

            X_ref_masked = X_ref_masked.detach().cpu()

        if i == 1:
            pass  # X_pred already set above

        if i > 0:
            X_pred_whole = torch.empty(X_ref_masked.shape, device=test_ds[:][0].device)
            X_pred_whole[:, model.opt.fixed_idx] = X_ref_masked[:, model.opt.fixed_idx].to(test_ds[:][0].device)
            X_pred_whole[:, model.opt.gen_idx] = X_pred.to(test_ds[:][0].device)
            T_pred = FWD(X_pred_whole)
            T_comp_pred = torch.complex(T_pred[:, 0], T_pred[:, 1])
            tr_pred = torch.square(torch.abs(T_comp_pred))
            ph_pred = torch.angle(T_comp_pred)
            tr_mse = nn.L1Loss()(tr_pred, tr_true[mask_idx])
            ph_mse = nn.L1Loss()(ph_pred, ph_true[mask_idx])
            loss = nn.L1Loss()(T_pred, test_ds[:][1][mask_idx])
            print(f'Transmittance Range: {T_bound}, Phase Range: {P_bound}')
            print(f'Error (Transmission | Transmittance | Phase Shift): {loss:.2e} | {tr_mse:.2e} | {ph_mse:.2e}')

    infeasible_idx1 = (X_pred[:, 0].round(decimals=2) == 0).detach().cpu().sum()
    infeasible_idx2 = (X_pred[:, 1].round(decimals=2) > 1.8).detach().cpu().sum()
    print(f'Infeasible designs in generated designs: n(l2 <= 0) = {infeasible_idx1} / {X_pred.shape[0]}, n(l5 > 1.8) = {infeasible_idx2} / {X_pred.shape[0]}')

@torch.no_grad()
def evaluate_design(model_name:str, models, test_ds, T_bound = [0.9, 1], P_bound = [-50, -30]):
    model, FWD = models
    random_seed(model.opt.seed)

    T_complex = torch.complex(test_ds[:][1][:,0], test_ds[:][1][:,1])
    tr_true = torch.square(torch.abs(T_complex))
    ph_true = torch.angle(T_complex)

    T_lower_mask = torch.square(torch.abs(T_complex)) > T_bound[0]; T_upper_mask = torch.square(torch.abs(T_complex)) < T_bound[1]
    T_bound_mask = T_lower_mask & T_upper_mask

    P_lower_mask = torch.angle(T_complex)*180/torch.pi > P_bound[0]; P_upper_mask = torch.angle(T_complex)*180/torch.pi < P_bound[1]
    P_bound_mask = P_lower_mask & P_upper_mask
    mask_idx = T_bound_mask & P_bound_mask

    X_gen_masked = test_ds[:][0][mask_idx]; T_gen_masked = torch.square(torch.abs(T_complex[mask_idx])).detach().cpu(); P_gen_masked = torch.angle(T_complex[mask_idx]).detach().cpu()

    X_ref_masked = test_ds[:][0][mask_idx]; T_ref_masked = torch.square(torch.abs(T_complex[mask_idx])).detach().cpu(); P_ref_masked = torch.angle(T_complex[mask_idx]).detach().cpu()

    model.eval()
    exp = deepcopy(model.opt.model_cfg['psat'])
    cond1 = exp['input_dim'] == 2
    if (exp['output_dim'] == 8): cond2 = exp['input_dim'] == 2+exp['output_dim']
    else: cond2 = exp['input_dim'] == 2+(8-exp['output_dim'])

    for i in range(2):
        fig1 = plt.figure(figsize = (5*5/12, 5*5/12)); fig2 = plt.figure(figsize = (5*5/12, 5*5/12))
        ax1 = fig1.add_subplot(1, 1, 1, projection = '3d'); ax2 = fig2.add_subplot(1, 1, 1, projection = '3d')
        z = torch.normal(mean = 0, std = 1, size = (X_gen_masked.shape[0], model.opt.model_cfg.psat.latent_dim), device = X_gen_masked.device)
        with torch.no_grad():
            if i == 1:
                if cond1: 
                    X_pred = model(x = test_ds[:][1][mask_idx], z = z)
                elif cond2: 
                    if exp['output_dim'] == 8:
                        X_mean = torch.tensor([0.6/2, 4.1/2, 5/2, 9/2, 2.1/2, 4.1/2, 8.1/2, 1.6/2], device = X_gen_masked.device).repeat(X_gen_masked.shape[0], 1)
                        X_pred = model(x = torch.cat([X_mean, test_ds[:][1][mask_idx]], dim = 1), z = z, b = test_ds[:][3][mask_idx])
                    else:
                        X_pred = model(x = torch.cat([X_gen_masked[:,model.opt.fixed_idx],test_ds[:][1][mask_idx]], dim = 1), z = z, b = test_ds[:][3][mask_idx])
            X_ref_masked = X_ref_masked.detach().cpu()
        if i == 1: X_pred = X_pred[1].detach().cpu()
        if i > 0:
            X_pred_whole = torch.empty(X_ref_masked.shape, device = test_ds[:][0].device)
            X_pred_whole[:,model.opt.fixed_idx]  = X_ref_masked[:,model.opt.fixed_idx].to(test_ds[:][0].device)
            X_pred_whole[:,model.opt.gen_idx] = X_pred.to(test_ds[:][0].device)
            T_pred = FWD(X_pred_whole)
            T_comp_pred = torch.complex(T_pred[:,0], T_pred[:,1])
            tr_pred = torch.square(torch.abs(T_comp_pred))
            ph_pred = torch.angle(T_comp_pred)
            tr_mse = nn.L1Loss()(tr_pred, tr_true[mask_idx])
            ph_mse = nn.L1Loss()(ph_pred, ph_true[mask_idx])
            loss = nn.L1Loss()(T_pred, test_ds[:][1][mask_idx])
            print(f'Transmittance Range: {T_bound}, Phase Range: {P_bound}')
            print(f'Error (Transmission | Transmittance | Phase Shift): {loss:.2e} | {tr_mse:.2e} | {ph_mse:.2e}')

        N = X_gen_masked.shape[0]
        xmin = X_ref_masked[:,model.opt.gen_idx[0]].min()-0.5; xmax = X_ref_masked[:,model.opt.gen_idx[0]].max()+0.5
        ymin = X_ref_masked[:,model.opt.gen_idx[1]].min()-0.5; ymax = X_ref_masked[:,model.opt.gen_idx[1]].max()+0.5
        dx = (xmax - xmin) / 3
        dy = (ymax - ymin) / 3
        if exp['output_dim'] >= 3: 
            zmin = X_ref_masked[:,model.opt.gen_idx[2]].min(); zmax = X_ref_masked[:,model.opt.gen_idx[2]].max()
            dz = (zmax - zmin) / 3
        
        if i == 0:
            if exp['output_dim'] == 2:
                # t1 = ax1.scatter(xs = X_ref_masked[:,model.opt.gen_idx[0]], ys = X_ref_masked[:,model.opt.gen_idx[1]], zs = T_ref_masked.detach().cpu(), edgecolor = 'k',\
                #                         linewidths= 0.5,s = 50, c = T_ref_masked.detach().cpu(), cmap = 'Blues', label = 'Real')
                t1 = ax1.scatter(xs = X_ref_masked[:,model.opt.gen_idx[0]], ys = X_ref_masked[:,model.opt.gen_idx[1]], \
                                        zs = T_ref_masked.detach().cpu(), s = 10, label = 'Real', cmap = 'Blues', c = T_ref_masked.detach().cpu(), alpha = 0.7)
                ax1.set_xticks(np.arange(xmin, xmax + dx - 1e-8, dx))
                ax1.set_yticks(np.arange(ymin, ymax + dy - 1e-8, dy))
                ax1.set_zticks(np.arange(0.9, 1.0+0.025-1e-8, 0.025))
                ax1.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
                for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
                    # outward_factor를 0으로 만들면 바깥쪽 눈금이 사라집니다.
                    axis._axinfo['tick']['outward_factor'] = 0.2
                    axis._axinfo['tick']['inward_factor'] = 0.0
            elif exp['output_dim'] >= 3:
                t1 = ax1.scatter(xs = X_ref_masked[:,model.opt.gen_idx[0]], ys = X_ref_masked[:,model.opt.gen_idx[1]], zs = X_ref_masked[:,model.opt.gen_idx[2]], edgecolor = 'k', linewidths= 0.5,s = 50, c = T_ref_masked.detach().cpu(), cmap = 'Blues', label = 'Real')
        else: 
            if exp['output_dim'] == 2:
                ax1.set_xticks(np.arange(xmin, xmax + dx - 1e-8, dx))
                ax1.set_yticks(np.arange(ymin, ymax + dy - 1e-8, dy))
                ax1.set_zticks(np.arange(0.9, 1.0+0.025-1e-8, 0.025))
                ax1.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
                for axis in (ax1.xaxis, ax1.yaxis, ax1.zaxis):
                    # outward_factor를 0으로 만들면 바깥쪽 눈금이 사라집니다.
                    axis._axinfo['tick']['outward_factor'] = 0.2
                    axis._axinfo['tick']['inward_factor'] = 0.0
                # t1 = ax1.scatter(xs = X_pred[:N,0], ys = X_pred[:N,1], zs = T_gen_masked.detach().cpu(), s = 50, edgecolor = 'k',\
                #                 linewidths= 0.5, c = T_gen_masked.detach().cpu()[:N], cmap = 'Blues', label = 'Pred')  
                t1 = ax1.scatter(xs = X_pred[:N,0], ys = X_pred[:N,1], zs = T_gen_masked.detach().cpu(), s = 10, \
                                cmap = 'Blues', c = T_gen_masked.detach().cpu(), label = 'Pred', alpha = 0.7)       
            elif exp['output_dim'] >= 3:
                ax1.set_xticks(np.arange(xmin, xmax + dx - 1e-8, dx))
                ax1.set_yticks(np.arange(ymin, ymax + dy - 1e-8, dy))
                ax1.set_zticks(np.arange(zmin, zmax + dz - 1e-8, dz))
                t1 = ax1.scatter(xs = X_pred[:N,0], ys = X_pred[:N,1], zs = X_pred[:N,2], s = 50, edgecolor = 'k',\
                                linewidths= 0.5, c = T_gen_masked.detach().cpu()[:N], cmap = 'Blues', label = 'Pred')               
        # ax1.set_xlim([X_ref_masked[:,model.opt.gen_idx[0]].min()*0.95, X_ref_masked[:,model.opt.gen_idx[0]].max()*1.05])
        # ax1.set_ylim([X_ref_masked[:,model.opt.gen_idx[1]].min()*0.95, X_ref_masked[:,model.opt.gen_idx[1]].max()*1.05])
        ax1.set_xlim([xmin-0.02, xmax+0.02])
        ax1.set_ylim([ymin-0.02, ymax+0.02])
        if exp['output_dim'] == 2: ax1.set_zlim([T_bound[0]-0.01, T_bound[1]])
        elif exp['output_dim'] >= 3: ax1.set_zlim([zmin-0.02, zmax+0.02])
        # ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_zticks([])
        ax1.grid(False)
        fig1.tight_layout()
        os.makedirs('./experiments/test_dataset_generation/', exist_ok=True)
        if i == 0: fig1.savefig(f'./experiments/test_dataset_generation/ground_truth_transmittance_gen_{model.opt.gen_variables}_{model.opt.material["freq"]}k.png', dpi = 500, transparent = True)
        else: fig1.savefig(f'./experiments/test_dataset_generation/{model.opt.model_cfg.name.inv}_transmittance_gen_{model.opt.gen_variables}_{model.opt.material["freq"]}k.png', dpi = 500, transparent = True)
        
        if i == 0: 
            if exp['output_dim'] == 2:
                
                t2 = ax2.scatter(xs = X_ref_masked[:,model.opt.gen_idx[0]], ys = X_ref_masked[:,model.opt.gen_idx[1]], \
                                zs = P_ref_masked.detach().cpu()*180/np.pi, s = 10,  cmap = 'Reds', c = P_ref_masked.detach().cpu()*180/np.pi,\
                                    label = 'Real', alpha = 0.7)
                # t2 = ax2.scatter(xs = X_ref_masked[:,model.opt.gen_idx[0]], ys = X_ref_masked[:,model.opt.gen_idx[1]], \
                #                 zs = P_ref_masked.detach().cpu()*180/np.pi, s = 10,  c = 'White',\
                #                     label = 'Real', alpha = 0.7)
                ax2.set_xticks(np.arange(xmin, xmax + dx - 1e-8, dx))
                ax2.set_yticks(np.arange(ymin, ymax + dy - 1e-8, dy))
                ax2.set_zticks(np.arange(-180.0, 180.0+120.0-1e-8, 120.0))
                ax2.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
                for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
                    # outward_factor를 0으로 만들면 바깥쪽 눈금이 사라집니다.
                    axis._axinfo['tick']['outward_factor'] = 0.2
                    axis._axinfo['tick']['inward_factor'] = 0.0
            elif exp['output_dim'] >= 3:
                ax1.set_xticks(np.arange(xmin, xmax + dx - 1e-8, dx))
                ax1.set_yticks(np.arange(ymin, ymax + dy - 1e-8, dy))
                ax1.set_zticks(np.arange(zmin, zmax + dz - 1e-8, dz))
                t2 = ax2.scatter(xs = X_ref_masked[:,model.opt.gen_idx[0]], ys = X_ref_masked[:,model.opt.gen_idx[1]], zs = X_ref_masked[:,model.opt.gen_idx[2]], edgecolor = 'k', s = 50, linewidths= 0.5, c = P_ref_masked.detach().cpu()*180/np.pi, cmap = 'Reds', label = 'Real')
        else:
            if exp['output_dim'] == 2:
                # t2 = ax2.scatter(xs = X_pred[:N,0], ys = X_pred[:N,1], zs = P_ref_masked.detach().cpu()*180/np.pi,\
                #                     s = 10, c = 'white', alpha = 0.7, label = 'Pred')
                t2 = ax2.scatter(xs = X_pred[:N,0], ys = X_pred[:N,1], zs = P_ref_masked.detach().cpu()*180/np.pi,\
                                    s = 10, cmap = 'Reds', c = P_ref_masked.detach().cpu()*180/np.pi, alpha = 0.7, label = 'Pred')
                # t2 = ax2.scatter(xs = X_pred[:N,0], ys = X_pred[:N,1], zs = P_ref_masked.detach().cpu()*180/np.pi,\
                #                     s = 10, color = 'white', edgecolor = 'k', linewidths = 0.1, alpha = 0.7, label = 'Pred')
                ax2.set_xticks(np.arange(xmin, xmax + dx - 1e-8, dx))
                ax2.set_yticks(np.arange(ymin, ymax + dy - 1e-8, dy))
                ax2.set_zticks(np.arange(-180.0, 180.0+120.0-1e-8, 120.0))
                ax2.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
                for axis in (ax2.xaxis, ax2.yaxis, ax2.zaxis):
                    # outward_factor를 0으로 만들면 바깥쪽 눈금이 사라집니다.
                    axis._axinfo['tick']['outward_factor'] = 0.2
                    axis._axinfo['tick']['inward_factor'] = 0.0
            elif exp['output_dim'] >= 3:
                ax1.set_xticks(np.arange(xmin, xmax + dx - 1e-8, dx))
                ax1.set_yticks(np.arange(ymin, ymax + dy - 1e-8, dy))
                ax1.set_zticks(np.arange(zmin, zmax + dz - 1e-8, dz))
                t2 = ax2.scatter(xs = X_pred[:N,0], ys = X_pred[:N,1], zs = X_pred[:N,2], s = 50, edgecolor = 'k',\
                                linewidths= 0.5, c = P_ref_masked.detach().cpu()*180/np.pi, cmap = 'Reds', label = 'Pred')         
        # ax2.set_xlim([X_ref_masked[:,model.opt.gen_idx[0]].min()*0.95, X_ref_masked[:,model.opt.gen_idx[0]].max()*1.05])
        ax2.set_xlim([xmin-0.02, xmax+0.02])
        ax2.set_ylim([ymin-0.02, ymax+0.02])
        # ax2.set_ylim([X_ref_masked[:,model.opt.gen_idx[1]].min()*0.95, X_ref_masked[:,model.opt.gen_idx[1]].max()*1.05])
        if exp['output_dim'] == 2: ax2.set_zlim([-185, 185])
        elif exp['output_dim'] >= 3: ax2.set_zlim([zmin-0.02, zmax+0.02])
        # ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
        ax2.grid(False)
        fig2.tight_layout()
        # os.makedirs('./experiments/test_dataset_generation/', exist_ok=True)
        if i == 0: fig2.savefig(f'./experiments/test_dataset_generation/ground_truth_phase_gen_{model.opt.gen_variables}_{model.opt.material["freq"]}k.png', dpi = 500, transparent = True)
        else: fig2.savefig(f'./experiments/test_dataset_generation/{model.opt.model_cfg.name.inv}_phase_gen_{model.opt.gen_variables}_{model.opt.material["freq"]}k.png', dpi = 500, transparent = True)
    print(f'3d design space plot: ./experiments/test_dataset_generation/' + '\n')
    infeasible_idx1 = (X_pred[:,0].round(decimals = 2) == 0).detach().cpu().sum()
    infeasible_idx2 = (X_pred[:,1].round(decimals = 2) > 1.8).detach().cpu().sum()
    print(f'Infeasible designs in generated designs: n(l2 <= 0) = {infeasible_idx1} / {X_pred.shape[0]}, n(l5 > 1.8) = {infeasible_idx2} / {X_pred.shape[0]}')

@torch.no_grad()
def evaluate_design_wo_plot(model_name:str, models, test_ds, T_bound = [0.9, 1], P_bound = [-50, -30]):
    model, FWD = models  
    random_seed(model.opt.seed)

    T_complex = torch.complex(test_ds[:][1][:,0], test_ds[:][1][:,1])
    tr_true = torch.square(torch.abs(T_complex))
    ph_true = torch.angle(T_complex)

    T_lower_mask = torch.square(torch.abs(T_complex)) > T_bound[0]; T_upper_mask = torch.square(torch.abs(T_complex)) < T_bound[1]
    T_bound_mask = T_lower_mask & T_upper_mask

    P_lower_mask = torch.angle(T_complex)*180/torch.pi > P_bound[0]; P_upper_mask = torch.angle(T_complex)*180/torch.pi < P_bound[1]
    P_bound_mask = P_lower_mask & P_upper_mask
    mask_idx = T_bound_mask & P_bound_mask

    X_gen_masked = test_ds[:][0][mask_idx]; T_gen_masked = torch.square(torch.abs(T_complex[mask_idx])).detach().cpu(); P_gen_masked = torch.angle(T_complex[mask_idx]).detach().cpu()

    X_ref_masked = test_ds[:][0][mask_idx]; T_ref_masked = torch.square(torch.abs(T_complex[mask_idx])).detach().cpu(); P_ref_masked = torch.angle(T_complex[mask_idx]).detach().cpu()

    model.eval()
    exp = deepcopy(model.opt.model_cfg['psat'])
    cond1 = exp['input_dim'] == 2
    if (exp['output_dim'] == 8): cond2 = exp['input_dim'] == 2+exp['output_dim']
    else: cond2 = exp['input_dim'] == 2+(8-exp['output_dim'])

    for i in range(2):
        z = torch.normal(mean = 0, std = 1, size = (X_gen_masked.shape[0], model.opt.model_cfg.psat.latent_dim), device = X_gen_masked.device)
        with torch.no_grad():
            if i == 1:
                if cond1: 
                    X_pred = model(x = test_ds[:][1][mask_idx], z = z)
                elif cond2: 
                    if exp['output_dim'] == 8:
                        X_mean = torch.tensor([0.6/2, 4.1/2, 5/2, 9/2, 2.1/2, 4.1/2, 8.1/2, 1.6/2], device = X_gen_masked.device).repeat(X_gen_masked.shape[0], 1)
                        X_pred = model(x = torch.cat([X_mean, test_ds[:][1][mask_idx]], dim = 1), z = z, b = test_ds[:][3][mask_idx])
                    else:
                        X_pred = model(x = torch.cat([X_gen_masked[:,model.opt.fixed_idx],test_ds[:][1][mask_idx]], dim = 1), z = z, b = test_ds[:][3][mask_idx])
            X_ref_masked = X_ref_masked.detach().cpu()
        if i == 1: X_pred = X_pred[1].detach().cpu()
        if i > 0:
            X_pred_whole = torch.empty(X_ref_masked.shape, device = test_ds[:][0].device)
            X_pred_whole[:,model.opt.fixed_idx]  = X_ref_masked[:,model.opt.fixed_idx].to(test_ds[:][0].device)
            X_pred_whole[:,model.opt.gen_idx] = X_pred.to(test_ds[:][0].device)
            T_pred = FWD(X_pred_whole)
            T_comp_pred = torch.complex(T_pred[:,0], T_pred[:,1])
            tr_pred = torch.square(torch.abs(T_comp_pred))
            ph_pred = torch.angle(T_comp_pred)
            tr_mse = nn.L1Loss()(tr_pred, tr_true[mask_idx])
            ph_mse = nn.L1Loss()(ph_pred, ph_true[mask_idx])
            loss = nn.L1Loss()(T_pred, test_ds[:][1][mask_idx])
            print(f'Transmittance Range: {T_bound}, Phase Range: {P_bound}')
            print(f'Error (Transmission | Transmittance | Phase Shift): {loss:.2e} | {tr_mse:.2e} | {ph_mse:.2e}')

    infeasible_idx1 = (X_pred[:,0].round(decimals = 2) == 0).detach().cpu().sum()
    infeasible_idx2 = (X_pred[:,1].round(decimals = 2) > 1.8).detach().cpu().sum()
    print(f'Infeasible designs in generated designs: n(l2 <= 0) = {infeasible_idx1} / {X_pred.shape[0]}, n(l5 > 1.8) = {infeasible_idx2} / {X_pred.shape[0]}')

def plot_attention_maps(model, plot = True):
    random_seed(model.opt.seed)
    ticks_label = np.array(['$\l_{1}$', '$\l_{2}$', '$\l_{3}$', '$\l_{4}$', '$\l_{5}$', '$\l_{6}$', '$\l_{7}$', '$\l_{8}$', '$\ T_{real}$', '$\ T_{imag}$'])
    cross_attn_weights = torch.mean(model.cross_attn.attn_weights.detach().cpu()[:,:,:,-2:], dim = (0,1))
    cross_attn_weights1 = torch.mean(model.cross_attn.attn_weights.detach().cpu()[:,0,:,-2:], dim = 0)
    cross_attn_weights2 = torch.mean(model.cross_attn.attn_weights.detach().cpu()[:,1,:,-2:], dim = 0)

    if plot:
        fig, axes = plt.subplots(1,3, figsize = (12,2))
        t = axes[0].xaxis.set_major_locator(plt.MultipleLocator(0.5))
        t = axes[0].yaxis.set_major_locator(plt.MultipleLocator(0.5))
        t = axes[0].imshow(torch.mean(model.cross_attn.attn_weights.detach().cpu(), dim = (0,1)), cmap = 'YlOrRd') # output과 input 간 cross-attention
        axes[0].grid(True, alpha = 0.3)
        axes[0].set_xticks(np.arange(model.opt.model_cfg['psat']['input_dim']), np.concatenate([ticks_label[model.opt.fixed_idx], ticks_label[-2:]]))
        axes[0].set_yticks(np.arange(model.opt.model_cfg['psat']['output_dim']), ticks_label[model.opt.gen_idx])
        fig.colorbar(t, ax = axes[0], fraction = 0.01, aspect = 25)

        t1 = axes[1].xaxis.set_major_locator(plt.MultipleLocator(0.5))
        t1 = axes[1].yaxis.set_major_locator(plt.MultipleLocator(0.5))
        t1 = axes[1].imshow(torch.mean(model.cross_attn.attn_weights.detach().cpu()[:,:,:,-2:], dim = (0,1)), cmap = 'YlOrRd') # output과 input 간 cross-attention에서 T_real, T_imag 부분
        axes[1].grid(True, alpha = 0.3)
        axes[1].set_xticks(np.arange(2), ticks_label[-2:])
        axes[1].set_yticks(np.arange(model.opt.model_cfg['psat']['output_dim']), ticks_label[model.opt.gen_idx])
        fig.colorbar(t1, ax = axes[1], fraction = 0.01, aspect = 25)

        t2 = axes[2].xaxis.set_major_locator(plt.MultipleLocator(0.5))
        t2 = axes[2].yaxis.set_major_locator(plt.MultipleLocator(0.5))
        t2 = axes[2].imshow(torch.mean(model.out_self_attn.attn_weights.detach().cpu(), dim = (0,1)), cmap = 'YlOrRd') # output 간 self-attention
        axes[2].grid(True, alpha = 0.3)
        axes[2].set_xticks(np.arange(2), ticks_label[-2:])
        axes[2].set_yticks(np.arange(model.opt.model_cfg['psat']['output_dim']), ticks_label[model.opt.gen_idx])
        axes[2].set_xticks(np.arange(model.opt.model_cfg['psat']['output_dim']), ticks_label[model.opt.gen_idx])
        axes[2].set_yticks(np.arange(model.opt.model_cfg['psat']['output_dim']), ticks_label[model.opt.gen_idx])
        fig.colorbar(t2, ax = axes[2], fraction = 0.01, aspect = 25)
        fig.suptitle('Input (All) & Pseudo Design | Input (Transmission) & Pseudo Design | Within Pseudo Design')
        fig.tight_layout()
        plt.show()
    return cross_attn_weights, cross_attn_weights1, cross_attn_weights2

def refracting_phase_profile(refracted_angle, **kwargs): # radian
    dy = kwargs['dy']; k_t = kwargs['k_t'] # transmitted angle
    # delta_y = 0.01 # Ly
    n_cells = kwargs.get('n_cells')
    # k0 = 116.5
    dphi = dy * k_t * np.sin(refracted_angle)
    profile = np.arange(0, n_cells*dphi, dphi)
    complex_value = np.exp(1j*profile)
    profile_converted = np.angle(complex_value)
    return profile_converted

def phase_grad2refracted_angle(phase_grad, **kwargs): # dphi/dy
    # k0 = 116.5
    k0 = kwargs['k0']; theta_i = kwargs['theta_i']
    # theta_i = 0
    refracted_angle = np.arcsin(np.sin(theta_i)+phase_grad/k0)
    return refracted_angle

def focusing_phase_profile(y, **kwargs):
    k0 = kwargs['k0']; f_l = kwargs['f_l']
    lens = k0*(np.sqrt(f_l**2 + y**2) - f_l)
    lens = np.exp(1j*lens)
    lens = np.angle(lens)
    return lens

def generate_design_practical_case(trainer, profile, type:str):
    random_seed(trainer.best_psaat.PSAT.opt.seed)
    gen_model, FWD = trainer.best_psaat.PSAT, trainer.best_FWD
    gen_model.eval(); FWD.eval()
    n_cells = len(profile); N_trans = 11; N_repeat = 100
    Transmittance = [0.9 + 0.01*j for j in range(N_trans)]
    design = trainer.test_ds[:][0][0:1,:].repeat(n_cells*N_trans*N_repeat,1).detach().clone() # Forward model input
    fixed_design = design[0:1,gen_model.opt.fixed_idx].detach().clone()
    b = torch.tensor([[0,0,1]], device = gen_model.opt.device)
    design_candidate = []; loss_list = []
    infeasible_cnt = 0
    start = time.time()
    for ph_idx, phase in enumerate(profile):
        for tr_idx, transmittance in enumerate(Transmittance):
            T_real = np.sqrt(transmittance)*np.cos(phase)
            T_imag = np.sqrt(transmittance)*np.sin(phase)
            T_true = torch.tensor([T_real, T_imag], dtype = torch.float32).unsqueeze(0).to(gen_model.opt.device)
            input = torch.cat([fixed_design, T_true], dim = 1).repeat(repeats = (N_repeat, 1))
            z = torch.normal(mean=0, std=1, size=(N_repeat, gen_model.opt.model_cfg.psat.latent_dim), device = gen_model.opt.device)
            _, gen_design = gen_model(input, z, b.repeat(N_repeat,1))
            gen_design = gen_design.round(decimals = 2)
            while ((gen_design[:,0] == 0).sum() > 0) or ((gen_design[:,1] > 1.8).sum() > 0):
                infeasible_idx = torch.where((gen_design[:,0] == 0) | (gen_design[:,1] > 1.8))[0]
                infeasible_cnt += len(infeasible_idx)
                z[infeasible_idx] = torch.normal(mean=0, std=1, size=(infeasible_idx.shape[0], gen_model.opt.model_cfg.psat.latent_dim), device = gen_model.opt.device)
                _, gen_design[infeasible_idx] = gen_model(input[infeasible_idx], z[infeasible_idx], b.repeat(N_repeat,1)[infeasible_idx])
                gen_design = gen_design.round(decimals = 2)
            design[(ph_idx*N_trans + tr_idx) * N_repeat: (ph_idx*N_trans+tr_idx+1)*N_repeat, gen_model.opt.gen_idx] = gen_design.detach().clone()
            T_pred = FWD(design[(ph_idx*N_trans + tr_idx) * N_repeat: (ph_idx*N_trans+tr_idx+1)*N_repeat]).detach().clone()
            tr_pred = torch.square(torch.abs(torch.complex(T_pred[:,0], T_pred[:,1])))
            ph_pred = torch.angle(torch.complex(T_pred[:,0], T_pred[:,1]))*180/np.pi
            loss = ((T_true.repeat(N_repeat,1) - T_pred)**2).mean(dim = 1)
            min_loss_idx = torch.where(loss == loss.min())[0][0]
            design_candidate.append(design[(ph_idx*N_trans + tr_idx) * N_repeat + min_loss_idx].cpu().tolist())
            loss_list.append(loss[min_loss_idx].item())
            print(f'Design Candidate: {np.round(np.array(design_candidate[-1]), decimals = 2)[[1,4]]} | Transmittance (True|Pred): {transmittance:.2f} | {tr_pred[min_loss_idx].item():.2f}, Phase (True|Pred): {phase*180/np.pi:.1f} | {ph_pred[min_loss_idx].item():.2f} | Loss: {loss_list[-1]:.2e}')    
    end = time.time()
    design_candidate = torch.tensor(design_candidate, dtype = torch.float64).round(decimals = 2)
    loss_list = torch.tensor(loss_list)

    best_design = []
    for i in range(n_cells):
        min_idx = torch.where(loss_list[N_trans*i:N_trans*(i+1)] == loss_list[N_trans*i:N_trans*(i+1)].min())[0]
        if min_idx.numel() > 1: min_idx = torch.tensor([min_idx[0]])
        best_design.extend(design_candidate[N_trans*i + min_idx, :].tolist())
        print(f'Best design: {np.array(best_design[-1])[[1,4]]} | Loss: {loss_list[N_trans*i + min_idx].item():.2e}')
    best_design = torch.tensor(best_design)
    
    print(f'Generation Time for Total {n_cells*N_trans*N_repeat} Designs: {end - start:.2f} sec')

    print(f'\n# of Physical Violations for Total {n_cells*N_trans*N_repeat/1000:.1f}k Generations: n(l2 <= 0) + n(l5 > 1.8) = {infeasible_cnt}')

    plt.figure(figsize = (8,4))
    os.makedirs(f'./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells', exist_ok = True)
    os.makedirs(f'./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells', exist_ok = True)

    n_cells = len(profile)
    if n_cells % 2 == 0: l_bound = -(n_cells//2 - 1); u_bound = (n_cells//2 + 1)
    else: l_bound = -(n_cells//2); u_bound = (n_cells//2) + 1
    profile[profile < 0] += 2*np.pi

    if type == 'refracting':
        plt.plot(np.arange(l_bound*gen_model.opt.material['H'], u_bound*gen_model.opt.material['H'], gen_model.opt.material['H']), profile*180/np.pi, 'k', alpha = 1, linewidth = 2)
        plt.scatter(np.arange(l_bound*gen_model.opt.material['H'], u_bound*gen_model.opt.material['H'], gen_model.opt.material['H']), profile*180/np.pi, c = 'none', edgecolors = 'b', linewidths = 2, marker = 'o', s = 100)
        plt.xticks()
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
        plt.savefig(f'./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells/{gen_model.opt.model_cfg.name.inv}_refracting_{gen_model.opt.refracted_angle*180/np.pi:.1f}deg_phase_profile.png', dpi = 500, transparent = True)
        spio.savemat(f'./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells/{gen_model.opt.model_cfg.name.inv}_refracting_{gen_model.opt.refracted_angle*180/np.pi:.1f}deg_design_candidate.mat', {'design_candidate': design_candidate.detach().cpu().numpy()})
        spio.savemat(f'./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells/{gen_model.opt.model_cfg.name.inv}_refracting_{gen_model.opt.refracted_angle*180/np.pi:.1f}deg_best_design.mat', {'best_design': best_design.detach().cpu().numpy()})
        
    elif type == 'focusing':
        plt.plot(np.arange(l_bound*gen_model.opt.material['H'], u_bound*gen_model.opt.material['H'], gen_model.opt.material['H']), profile*180/np.pi, 'k', alpha = 1, linewidth = 2)
        plt.scatter(np.arange(l_bound*gen_model.opt.material['H'], u_bound*gen_model.opt.material['H'], gen_model.opt.material['H']), profile*180/np.pi, c = 'none', edgecolors = 'b', linewidths = 2, marker = 'o', s = 100, label = 'Target phase')
        plt.savefig(f'./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells/{gen_model.opt.model_cfg.name.inv}_focusing_fl{trainer.opt.material["f_l"]}m_phase_profile.png', dpi = 500, transparent = True)
        spio.savemat(f'./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells/{gen_model.opt.model_cfg.name.inv}_focusing_fl{trainer.opt.material["f_l"]}m_design_candidate.mat', {'design_candidate': design_candidate.detach().cpu().numpy()})
        spio.savemat(f'./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells/{gen_model.opt.model_cfg.name.inv}_focusing_fl{trainer.opt.material["f_l"]}m_best_design.mat', {'best_design': best_design.detach().cpu().numpy()})
    print(f'\nGenerated designs saved in ./experiments/practical_case/{gen_model.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells/')
    
def optimize_design_practical_case(trainer, profile, type: str):
    random_seed(trainer.opt.seed)
    FWD = trainer.best_FWD
    FWD.eval()

    for param in FWD.parameters():
        param.requires_grad = False

    # ── config ──────────────────────────────────────────────────────
    gen_idx  = trainer.opt.gen_idx
    n_gen    = len(gen_idx)
    device   = trainer.opt.device
    n_points = trainer.opt.model_cfg.optim.n_points
    max_eval = getattr(trainer.opt.model_cfg.optim, 'max_eval', 1000)
    method   = trainer.opt.model_cfg.optim.method
    lr       = trainer.opt.model_cfg.optim.lr

    if trainer.opt.gen_variables == 'l2l5':
        lb = np.array([0.1, 0.1])
        ub = np.array([5.0, 1.8])

    lb_t = torch.tensor(lb, dtype=torch.float32, device=device)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=device)

    n_cells = len(profile)
    N_trans = 11
    Transmittance = [0.9 + 0.01 * j for j in range(N_trans)]

    fixed_design = trainer.test_ds[:][0][0:1, trainer.opt.fixed_idx].detach().clone()
    b = torch.tensor([[0, 0, 1]], device=device)

    # ── single target 최적화 helper ─────────────────────────────────
    def optimize_single_target(X_template, T_true, B_target):
        """
        n_points개 시작점으로 Adam/LBFGS 최적화 → best 설계 변수 반환
        """
        n_pts = n_points

        rng   = np.random.default_rng(seed=42)
        inits = rng.uniform(lb, ub, size=(n_pts, n_gen))

        X_gen = torch.tensor(inits, dtype=torch.float32, device=device)
        X_gen = nn.Parameter(X_gen)

        X_base  = X_template.expand(n_pts, -1).clone()
        T_rep   = T_true.expand(n_pts, -1)
        B_rep   = B_target.expand(n_pts, *[-1] * (B_target.dim() - 1))

        best_loss = float('inf')
        best_gen  = X_gen.data[0].clone()

        if method == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                [X_gen], lr=lr, max_iter=1000,
                tolerance_grad=1e-7, tolerance_change=1e-9, history_size=10,
            )
            n_outer = max(max_eval // 20, 5)

            for _ in range(n_outer):
                def closure():
                    optimizer.zero_grad()
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = FWD(X_full)
                    loss = nn.functional.mse_loss(T_pred, T_rep, reduction='sum')
                    loss.backward()
                    return loss

                optimizer.step(closure)

                with torch.no_grad():
                    X_gen.data.clamp_(lb_t, ub_t)
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = FWD(X_full)
                    losses = nn.functional.mse_loss(
                        T_pred, T_rep, reduction='none'
                    ).mean(dim=1)

                    min_loss, min_idx = losses.min(dim=0)
                    if min_loss.item() < best_loss:
                        best_loss = min_loss.item()
                        best_gen  = X_gen.data[min_idx].clone()

        elif method == 'adam':
            optimizer = torch.optim.Adam([X_gen], lr=lr)

            for step in range(max_eval):
                optimizer.zero_grad()
                X_full = X_base.clone()
                X_full[:, gen_idx] = X_gen
                T_pred = FWD(X_full)
                loss = nn.functional.mse_loss(T_pred, T_rep, reduction='sum')
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    X_gen.data.clamp_(lb_t, ub_t)
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = FWD(X_full)
                    losses = nn.functional.mse_loss(
                        T_pred, T_rep, reduction='none'
                    ).mean(dim=1)

                    min_loss, min_idx = losses.min(dim=0)
                    if min_loss.item() < best_loss:
                        best_loss = min_loss.item()
                        best_gen  = X_gen.data[min_idx].clone()

        # best 설계 조립
        best_X = X_template.clone()
        best_X[0, gen_idx] = best_gen

        # best에 대한 forward 평가
        with torch.no_grad():
            T_pred_best = FWD(best_X)
            T_comp = torch.complex(T_pred_best[0, 0], T_pred_best[0, 1])
            tr_pred = torch.square(torch.abs(T_comp)).item()
            ph_pred = (torch.angle(T_comp) * 180 / np.pi).item()

        return best_X, best_loss, tr_pred, ph_pred

    # ── 메인 루프 ───────────────────────────────────────────────────
    design_candidate = []
    loss_list = []

    start = time.time()

    pbar = tqdm(profile, desc = 'Profile #', position = 0, leave = True)
    # for ph_idx, phase in enumerate(profile):
    for ph_idx, phase in enumerate(pbar):
        for tr_idx, transmittance in enumerate(Transmittance):
            T_real = np.sqrt(transmittance) * np.cos(phase)
            T_imag = np.sqrt(transmittance) * np.sin(phase)
            T_true = torch.tensor([[T_real, T_imag]], dtype=torch.float32, device=device)

            # template: fixed 변수는 test_ds 첫 샘플에서 가져옴
            X_template = trainer.test_ds[:][0][0:1, :].detach().clone().to(device)

            best_X, best_loss, tr_pred, ph_pred = optimize_single_target(
                X_template, T_true, b,
            )

            # 소수 둘째자리 반올림 (생성 모델과 동일하게)
            best_X = best_X.round(decimals=2)
            design_candidate.append(best_X[0].cpu().tolist())
            loss_list.append(best_loss)

            print(
                f'Design Candidate: {np.round(np.array(design_candidate[-1]), decimals=2)[gen_idx]} | '
                f'Transmittance (True|Pred): {transmittance:.2f} | {tr_pred:.2f}, '
                f'Phase (True|Pred): {phase * 180 / np.pi:.1f} | {ph_pred:.2f} | '
                f'Loss: {loss_list[-1]:.2e}'
            )

    end = time.time()

    design_candidate = torch.tensor(design_candidate, dtype=torch.float64).round(decimals=2)
    loss_list = torch.tensor(loss_list)

    # ── cell별 best 선택 ────────────────────────────────────────────
    best_design = []
    for i in range(n_cells):
        min_idx = torch.where(
            loss_list[N_trans * i: N_trans * (i + 1)] == loss_list[N_trans * i: N_trans * (i + 1)].min()
        )[0]
        if min_idx.numel() > 1:
            min_idx = torch.tensor([min_idx[0]])
        best_design.extend(design_candidate[N_trans * i + min_idx, :].tolist())
        print(f'Best design: {np.array(best_design[-1])[gen_idx]} | Loss: {loss_list[N_trans * i + min_idx].item():.2e}')
    best_design = torch.tensor(best_design)

    print(f'Optimization Time for Total {n_cells * N_trans} Targets ({method}, n_points={n_points}): {end - start:.2f} sec')

    # ── plotting & saving ───────────────────────────────────────────
    plt.figure(figsize=(8, 4))
    method_name = f'grad_{method}'
    os.makedirs(f'./experiments/practical_case/{method_name}/gen_designs/{type}_{n_cells}cells', exist_ok=True)
    os.makedirs(f'./experiments/practical_case/{method_name}/comsol_results/{type}_{n_cells}cells', exist_ok=True)

    if n_cells % 2 == 0:
        l_bound = -(n_cells // 2 - 1)
        u_bound = (n_cells // 2 + 1)
    else:
        l_bound = -(n_cells // 2)
        u_bound = (n_cells // 2) + 1

    profile[profile < 0] += 2 * np.pi

    H = trainer.opt.material['H']
    x_positions = np.arange(l_bound * H, u_bound * H, H)

    if type == 'refracting':
        plt.plot(x_positions, profile * 180 / np.pi, 'k', alpha=1, linewidth=2)
        plt.scatter(x_positions, profile * 180 / np.pi, c='none', edgecolors='b',
                    linewidths=2, marker='o', s=100)
        plt.xticks()
        plt.tick_params(axis='both', direction='in', labelbottom=False, labelleft=False)
        angle_str = f'{trainer.opt.refracted_angle * 180 / np.pi:.1f}'
        plt.savefig(
            f'./experiments/practical_case/{method_name}/gen_designs/{type}_{n_cells}cells/{method_name}_refracting_{angle_str}deg_phase_profile.png',
            dpi=500, transparent=True,
        )
        spio.savemat(
            f'./experiments/practical_case/{method_name}/gen_designs/{type}_{n_cells}cells/{method_name}_refracting_{angle_str}deg_design_candidate.mat',
            {'design_candidate': design_candidate.detach().cpu().numpy()},
        )
        spio.savemat(
            f'./experiments/practical_case/{method_name}/gen_designs/{type}_{n_cells}cells/{method_name}_refracting_{angle_str}deg_best_design.mat',
            {'best_design': best_design.detach().cpu().numpy()},
        )

    elif type == 'focusing':
        plt.plot(x_positions, profile * 180 / np.pi, 'k', alpha=1, linewidth=2)
        plt.scatter(x_positions, profile * 180 / np.pi, c='none', edgecolors='b',
                    linewidths=2, marker='o', s=100, label='Target phase')
        fl_str = f'{trainer.opt.material["f_l"]}'
        plt.savefig(
            f'./experiments/practical_case/{method_name}/gen_designs/{type}_{n_cells}cells/{method_name}_focusing_fl{fl_str}m_phase_profile.png',
            dpi=500, transparent=True,
        )
        spio.savemat(
            f'./experiments/practical_case/{method_name}/gen_designs/{type}_{n_cells}cells/{method_name}_focusing_fl{fl_str}m_design_candidate.mat',
            {'design_candidate': design_candidate.detach().cpu().numpy()},
        )
        spio.savemat(
            f'./experiments/practical_case/{method_name}/gen_designs/{type}_{n_cells}cells/{method_name}_focusing_fl{fl_str}m_best_design.mat',
            {'best_design': best_design.detach().cpu().numpy()},
        )

    print(f'\nGenerated designs saved in ./experiments/practical_case/{method_name}/gen_designs/{type}_{n_cells}cells/')
    
def evaluate_design_practical_case(trainer, profile, type): # 생성한 디자인의 COMSOL 결과 확인
    random_seed(trainer.opt.seed)
    n_cells = len(profile)
    profile[profile < 0] += 2*np.pi
    if n_cells % 2 == 0: l_bound = -(n_cells//2 - 1); u_bound = (n_cells//2 + 1)
    else: l_bound = -(n_cells//2); u_bound = (n_cells//2) + 1
    
    os.makedirs(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/gen_designs/{type}_{n_cells}cells', exist_ok = True)
    os.makedirs(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells', exist_ok = True)
    
    xmin = l_bound*trainer.opt.material['H'] - trainer.opt.material['H']/2
    xmax = u_bound*trainer.opt.material['H']
    dx = (xmax - xmin) / 7
    if type == 'refracting':
        T_gen = spio.loadmat(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_best_design_comsol_results.mat')['transmission']
        tr_gen = np.square(np.abs(T_gen))
        ph_gen = np.angle(T_gen)
        ph_gen[ph_gen < 0] += 2*np.pi
        ph_gen[np.abs(ph_gen - 2*np.pi) < 0.1] = 0.0
        plt.figure(figsize = (8,4))
        plt.hlines(xmin = -180, xmax = 180, y = 0.9, color = 'b', linestyle = '--', label = 'Transmittance (Target)')
        plt.scatter(x = np.arange(l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H'], trainer.opt.material['H']), y = tr_gen, color = 'r', marker = '*', s = 150, label = 'Transmittance (Generated)')
        plt.yticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
        plt.ylim([-0.02, 1.02])
        plt.xlim([l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H']])
        plt.xticks(np.arange(xmin, xmax + dx, dx))
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
        # plt.grid()
        # plt.legend()
        plt.savefig(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_transmittance_wo_ticks.png', dpi = 500, transparent = True)
        print(f'Transmittance Results Saved in ./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_transmittance_wo_ticks.png')
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = True, labelleft = True)
        plt.savefig(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_transmittance_with_ticks.png', dpi = 500, transparent = True)
        print(f'Transmittance Results Saved in ./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_transmittance_with_ticks.png')

        plt.figure(figsize = (8,4))
        plt.scatter(x = np.arange(l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H'], trainer.opt.material['H']), y = ph_gen*180/np.pi, color = 'r', marker = '*', s = 150, label = 'Phase (Generated)')
        plt.plot(np.arange(l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H'], trainer.opt.material['H']), profile*180/np.pi, color = 'b', linestyle = '--', linewidth = 2, label = 'Phase (Target)')
        plt.xlim([l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H']])
        # plt.ylim([-185, 185])
        plt.yticks([0, 90, 180, 270, 360])
        plt.ylim([-5, 365])
        plt.xticks(np.arange(xmin, xmax + dx, dx))
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
        # plt.grid()
        # plt.show()
        plt.savefig(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_phase_shift_wo_ticks.png', dpi = 500, transparent = True)
        print(f'Phase Shift Results Saved in ./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_phase_shift_wo_ticks.png')
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = True, labelleft = True)
        plt.savefig(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_phase_shift_with_ticks.png', dpi = 500, transparent = True)
        print(f'Phase Shift Results Saved in ./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_{np.rad2deg(trainer.opt.refracted_angle):.1f}deg_phase_shift_with_ticks.png')

    elif type == 'focusing':
        T_gen = spio.loadmat(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_best_design_comsol_results.mat')['transmission']
        tr_gen = np.square(np.abs(T_gen))
        ph_gen = np.angle(T_gen)
        ph_gen[ph_gen < 0] += 2*np.pi
        ph_gen[np.abs(ph_gen - 2*np.pi) < 0.1] = 0.0
        plt.figure(figsize = (8,4))
        plt.hlines(xmin = -180, xmax = 180, y = 0.9, color = 'b', linestyle = '--', label = 'Transmittance (Target)')
        plt.scatter(x = np.arange(l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H'], trainer.opt.material['H']), y = tr_gen, color = 'r', marker = '*', s = 150, label = 'Transmittance (Generated)')
        plt.ylim([-0.02, 1.02])
        plt.yticks([0, 0.25, 0.5, 0.75, 0.9, 1.0])
        plt.xticks(np.arange(xmin, xmax + dx, dx))
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
        plt.xlim([l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H']])
        # plt.grid()
        plt.savefig(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_transmittance_wo_ticks.png', dpi = 500, transparent = True)
        print(f'Transmittance Results Saved in ./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_transmittance_wo_ticks.png')
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = True, labelleft = True)
        plt.savefig(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_transmittance_with_ticks.png', dpi = 500, transparent = True)
        print(f'Transmittance Results Saved in ./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_transmittance_with_ticks.png')

        plt.figure(figsize = (8,4))
        plt.scatter(x = np.arange(l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H'], trainer.opt.material['H']), y = ph_gen*180/np.pi, color = 'r', marker = '*', s = 150, label = 'Phase (Generated)')
        plt.plot(np.arange(l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H'], trainer.opt.material['H']), profile*180/np.pi, color = 'b', linestyle = '--',  linewidth = 2, label = 'Phase (Target)')
        # plt.ylim([-185, 185])
        plt.xticks(np.arange(xmin, xmax + dx, dx))
        plt.yticks([0, 90, 180, 270, 360])
        plt.ylim([-5, 365])
        plt.xlim([l_bound*trainer.opt.material['H'], u_bound*trainer.opt.material['H']])
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
        # plt.grid()
        # plt.show()
        plt.savefig(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_phase_shift_wo_ticks.png', dpi = 500, transparent = True)
        print(f'Phase Shift Results Saved in ./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_phase_shift_wo_ticks.png')
        plt.tick_params(axis = 'both', direction = 'in', labelbottom = True, labelleft = True)
        plt.savefig(f'./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_phase_shift_with_ticks.png', dpi = 500, transparent = True)
        print(f'Phase Shift Results Saved in ./experiments/practical_case/{trainer.opt.model_cfg.name.inv}/comsol_results/{type}_{n_cells}cells/{trainer.opt.model_cfg.name.inv}_{type}_fl{trainer.opt.material["f_l"]}m_phase_shift_with_ticks.png')
    
    # breakpoint()
    tr_gen[tr_gen >= 0.9] = 0.9
    # breakpoint()
    tr_true = 0.9 * torch.ones(ph_gen.shape[0],1)
    ph_true = torch.tensor(profile).unsqueeze(1)
    tr_gen = torch.tensor(tr_gen)
    ph_gen = torch.tensor(ph_gen)
    T_true = torch.cat([tr_true*torch.cos(ph_true), tr_true*torch.sin(ph_true)], dim = 1)
    T_gen = torch.cat([tr_gen*torch.cos(ph_gen), tr_gen*torch.sin(ph_gen)], dim = 1)
    
    T_error = torch.nn.L1Loss()(T_true, T_gen)    
    ph_error = torch.nn.L1Loss()(ph_true, ph_gen)
    tr_error = torch.nn.L1Loss()(tr_true, tr_gen)
    print(f'Transmittance Error: {tr_error.item():.2e} | Phase Error: {ph_error.item():.2e} | Total T Error: {T_error.item():.2e}')

def physical_analysis_attention(trainer):
    random_seed(trainer.best_psaat.PSAT.opt.seed)
    phase_interval = 10
    cross_attn_weights_list = []; cross_attn_weights1_list = []; cross_attn_weights2_list = []
    for phase in np.arange(-180, 180, phase_interval):
        evaluate_design_wo_plot(model_name = 'psaat', models = [trainer.best_psaat.PSAT, trainer.best_FWD], test_ds = trainer.test_ds, T_bound = [0.90, 1.01], P_bound = [phase, phase + phase_interval])
        cross_attn_weights, cross_attn_weights1, cross_attn_weights2 = plot_attention_maps(model = trainer.best_psaat.PSAT, plot = False)
        cross_attn_weights_list.append(cross_attn_weights.tolist())
        cross_attn_weights1_list.append(cross_attn_weights1.tolist())
        cross_attn_weights2_list.append(cross_attn_weights2.tolist())
    
    cross_attn = torch.tensor(cross_attn_weights_list)
    # ymax_0 = cross_attn[:,:,0].sum(dim = 1).max()
    # ymin_0 = cross_attn[:,:,0].sum(dim = 1).min()
    # ymax_1 = cross_attn[:,:,1].sum(dim = 1).max()
    # ymin_1 = cross_attn[:,:,1].sum(dim = 1).min()
    # dy_0 = (ymax_0 - ymin_0)/4
    # dy_1 = (ymax_1 - ymin_1)/4
    
    nan_idx = torch.isnan(cross_attn[:,0,0])
    ymax_0 = cross_attn[~nan_idx,:,0].sum(dim = 1).max()
    ymin_0 = cross_attn[~nan_idx,:,0].sum(dim = 1).min()
    ymax_1 = cross_attn[~nan_idx,:,1].sum(dim = 1).max()
    ymin_1 = cross_attn[~nan_idx,:,1].sum(dim = 1).min()
    ymean_0 = cross_attn[~nan_idx,:,0].sum(dim = 1).mean() / cross_attn.shape[1]
    ymean_1 = cross_attn[~nan_idx,:,1].sum(dim = 1).mean() / cross_attn.shape[1]
    cross_attn[nan_idx,:,0] = ymean_0
    cross_attn[nan_idx,:,1] = ymean_1
    dy_0 = (ymax_0 - ymin_0)/4
    dy_1 = (ymax_1 - ymin_1)/4
    
    os.makedirs(f'./experiments/physical_analysis_attention', exist_ok = True)

    fig1, ax1 = plt.subplots(1,1, figsize = (6,3)); fig2, ax2 = plt.subplots(1,1, figsize = (6,3))
    ax1.plot(np.arange(-175, 185, phase_interval), cross_attn[:,:,0].sum(dim = 1), color = 'b', marker = 'o', label = 'T (real)') # T (real)
    markerline1, stemlines1, baseline1 = ax1.stem(np.arange(-175, 185, phase_interval), cross_attn[:,:,0].sum(dim = 1), basefmt = 'b', linefmt = 'b--', markerfmt = 'o', bottom = cross_attn[:,:,0].sum(dim = 1).mean(), label = 'T (real)') # T (real)
    markerline1.set_markerfacecolor('none')
    ax1.set_xticks(np.arange(-180, 180+90-1e-8, 90))
    ax1.set_yticks(np.arange(ymin_0, ymax_0 + dy_0-1e-8, dy_0))
    ax1.set_xlim([-185, 185]); 
    ax1.set_ylim([ymin_0-0.01, ymax_0+0.01]); 
    ax1.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
    fig1.tight_layout()
    fig1.savefig(f'./experiments/physical_analysis_attention/attention_score_I(real)_gen_{trainer.opt.gen_variables}_{trainer.opt.material["freq"]}k_wo_ticks.png', dpi = 500, transparent = True)

    ax2.plot(np.arange(-175, 185, phase_interval), cross_attn[:,:,1].sum(dim = 1), color = 'r', marker = 'o', label = 'T (imag)') # T (imag)
    markerline2, stemlines2, baseline2 = ax2.stem(np.arange(-175, 185, phase_interval), cross_attn[:,:,1].sum(dim = 1), basefmt = 'r', linefmt = 'r--', markerfmt = 'D', bottom = cross_attn[:,:,1].sum(dim = 1).mean(), label = 'T (imag)') # T (imag)
    markerline2.set_markerfacecolor('none')
    ax2.set_xticks(np.arange(-180, 180+90-1e-8, 90))
    ax2.set_yticks(np.arange(ymin_1, ymax_1 + dy_1-1e-8, dy_1))
    ax2.set_xlim([-185, 185])
    ax2.set_ylim([ymin_1-0.01, ymax_1+0.01])
    ax2.tick_params(axis = 'both', direction = 'in', labelbottom = False, labelleft = False)
    fig2.tight_layout()
    fig2.savefig(f'./experiments/physical_analysis_attention/attention_score_I(imag)_gen_{trainer.opt.gen_variables}_{trainer.opt.material["freq"]}k_wo_ticks.png', dpi = 500, transparent = True)
    # axes[0].grid(True, axis = 'x')
    # axes[1].grid(True, axis = 'x')
    # axes[0].set_xticks([]); axes[1].set_xticks([])
    # axes[0].set_yticks([]); axes[1].set_yticks([])
    # axes[0].tick_params(axis = 'x', bottom = False, labelbottom = False)
    # axes[1].tick_params(axis = 'x', bottom = False, labelbottom = False)

    ax1.tick_params(axis = 'both', direction = 'in', labelbottom = True, labelleft = True)
    ax2.tick_params(axis = 'both', direction = 'in', labelbottom = True, labelleft = True)
    ax1.set_xticks(np.arange(-180, 180+90-1e-8, 90))
    ax2.set_xticks(np.arange(-180, 180+90-1e-8, 90))
    ax1.set_yticks(np.arange(ymin_0, ymax_0 + dy_0-1e-8, dy_0))
    ax2.set_yticks(np.arange(ymin_1, ymax_1 + dy_1-1e-8, dy_1))
    ax1.set_xlim([-185, 185]); ax2.set_xlim([-185, 185])
    ax1.set_ylim([ymin_0-0.01, ymax_0+0.01]); ax2.set_ylim([ymin_1-0.01, ymax_1+0.01])
    ax1.grid(True, axis = 'x')
    ax2.grid(True, axis = 'x')
    fig1.tight_layout(); fig2.tight_layout();
    fig1.savefig(f'./experiments/physical_analysis_attention/attention_score_I(real)_gen_{trainer.opt.gen_variables}_{trainer.opt.material["freq"]}k_with_ticks.png', dpi = 500, transparent = True)
    fig2.savefig(f'./experiments/physical_analysis_attention/attention_score_I(imag)_gen_{trainer.opt.gen_variables}_{trainer.opt.material["freq"]}k_with_ticks.png', dpi = 500, transparent = True)
    print(f'Figure saved to ./experiments/physical_analysis_attention/')
    
def mma_based_optim(trainer):
    fwd = trainer.best_FWD
    fwd.eval()
    
    for param in fwd.parameters():
        param.requires_grad = False
    
    n_points = trainer.opt.model_cfg.optim.n_points
    gen_idx = trainer.opt.gen_idx
    n_gen = len(gen_idx)
    device = trainer.opt.device
    max_eval = getattr(trainer.opt.model_cfg.optim, 'max_eval', 1000)
    
    if trainer.opt.gen_variables == 'l2l5': 
        lb = np.array([0.1, 0.1])
        ub = np.array([5.0, 1.8])
    def make_objective(X_template, T_target, B_target):
        x_fixed = X_template.squeeze(0).detach().clone()
        def objective(X_gen, grad):
            X_gen_t = torch.tensor(X_gen, dtype=torch.float32,
                                requires_grad=True, device=device)
            vals = list(x_fixed)
            for i, idx in enumerate(gen_idx):
                vals[idx] = X_gen_t[i]
            X_full = torch.stack(vals).unsqueeze(0)
            
            with torch.enable_grad():
                T_pred = fwd(X_full)
                loss =  nn.functional.mse_loss(T_pred, T_target)
            
            if grad.size > 0:
                loss.backward()
                grad[:] = X_gen_t.grad.detach().cpu().numpy()
            return loss.item()
        return objective
    
    start = time.time()
    X_opt_list = []
    T_mae_list = []
    tr_mae_list = []
    ph_mae_list = []
    T_tgt_list, B_tgt_list = [], []
    
    for X, T, _, B in trainer.test_dl:
        X, T, B = X.to(device), T.to(device), B.to(device)
        T_tgt_list.append(T); B_tgt_list.append(B)
        pbar1 = tqdm(range(1, X.shape[0]+1), desc = 'Optim', unit = 'sample', ncols = 150, position = 0, leave = True)
        for i in pbar1:
            X_template = X[i-1:i]
            T_target = T[i-1:i]
            B_target = B[i-1:i]
            obj_fn = make_objective(X_template, T_target, B_target)
            
            rng = np.random.default_rng(seed=i)
            inits = rng.uniform(lb, ub, size = (n_points, n_gen))
            best_f, best_X = float('inf'), X_template.clone()
            
            # if i % 100 == 0: print(i)
            
            # pbar2 = tqdm(range(1, n_points+1), desc = 'Multi-sample', unit = 'sample', ncols = 150, position = 1, leave = False)
            # for j in pbar2:
            for j in range(n_points):
                opt = nlopt.opt(nlopt.LD_MMA, n_gen)
                opt.set_lower_bounds(lb.tolist())
                opt.set_upper_bounds(ub.tolist())
                opt.set_min_objective(obj_fn)
                opt.set_maxeval(max_eval)
                opt.set_xtol_rel(1e-3)
                try:
                    X_gen_opt = opt.optimize(inits[j].tolist())
                    f_opt = opt.last_optimum_value()
                except Exception:
                    continue
                
                if f_opt < best_f:
                    best_f = f_opt
                    X_best = X_template.clone()
                    X_best[:,gen_idx] = torch.tensor(X_gen_opt, dtype = torch.float32, device = device)
                    best_X = X_best
            pbar1.set_postfix({
                    'X(init)':  str([round(v, 2).item() for v in inits[0]]),
                    'X(optim)': str([round(v, 2) for v in best_X[:,gen_idx].detach().cpu().squeeze().tolist()])
})
            X_opt_list.append(best_X)
            
    end = time.time()
    
    with torch.no_grad():
        X_opt_torch = torch.cat(X_opt_list, dim = 0)
        T_tgt_torch = torch.cat(T_tgt_list, dim = 0)
        B_tgt_torch = torch.cat(B_tgt_list, dim = 0)
        T_pred = fwd(X_opt_torch)
        T_comp_pred = torch.complex(T_pred[:,0], T_pred[:,1])
        tr_pred = torch.square(torch.abs(T_comp_pred))
        ph_pred = torch.angle(T_comp_pred)
        
        T_comp_tgt = torch.complex(T_tgt_torch[:,0], T_tgt_torch[:,1])
        tr_tgt = torch.square(torch.abs(T_comp_tgt))
        ph_tgt = torch.angle(T_comp_tgt)
        
        T_l1 = nn.L1Loss()(T_pred, T_tgt_torch).item()
        tr_l1 = nn.L1Loss()(tr_pred, tr_tgt).item()
        ph_l1 = nn.L1Loss()(ph_pred, ph_tgt).item()
            
    print(f"Time: {end - start:.2f}s")
    print('MAE of Transmission Coeff. | Transmittance | Phase Shift :', f'{T_l1:.2e} | {tr_l1:.2e} | {ph_l1:.2e}\n')
    X_opt = torch.stack(X_opt_list)
    return X_opt

# def grad_based_optim(trainer):
#     fwd = trainer.best_FWD
#     fwd.eval()
 
#     for param in fwd.parameters():
#         param.requires_grad = False
 
#     # ── config ──────────────────────────────────────────────────────
#     n_points = trainer.opt.model_cfg.optim.n_points
#     gen_idx  = trainer.opt.gen_idx
#     n_gen    = len(gen_idx)
#     device   = trainer.opt.device
#     max_eval = getattr(trainer.opt.model_cfg.optim, 'max_eval', 1000)
 
#     if trainer.opt.gen_variables == 'l2l5':
#         lb = np.array([0.1, 0.1])
#         ub = np.array([5.0, 1.8])
 
#     lb_t = torch.tensor(lb, dtype=torch.float32, device=device)
#     ub_t = torch.tensor(ub, dtype=torch.float32, device=device)
 
#     # ── helper: 단일 샘플에 대해 n_points 시작점을 배치로 최적화 ──
#     def optimize_single_sample(X_template, T_target, B_target, inits):
#         """
#         X_template : (1, n_feat)
#         T_target   : (1, ...)
#         B_target   : (1, ...)
#         inits      : (n_points, n_gen)  numpy
#         Returns    : best_X (1, n_feat), best_loss (float)
#         """
#         n_pts = inits.shape[0]
 
#         # 시작점을 GPU 텐서로 (n_points, n_gen)
#         X_gen = torch.tensor(inits, dtype=torch.float32, device=device)
#         X_gen = nn.Parameter(X_gen)
 
#         # template / target을 n_points 만큼 expand
#         X_base = X_template.expand(n_pts, -1).clone()        # (n_points, n_feat)
#         T_rep  = T_target.expand(n_pts, *[-1] * (T_target.dim() - 1))
#         B_rep  = B_target.expand(n_pts, *[-1] * (B_target.dim() - 1))

#         best_loss = float('inf')
#         best_gen  = X_gen.data.clone()
        
#         if trainer.opt.model_cfg.optim.method == 'lbfgs':
#             optimizer = torch.optim.LBFGS(
#                 [X_gen],
#                 lr=trainer.opt.model_cfg.optim.lr,
#                 max_iter=20,           # LBFGS 내부 line-search 반복
#                 tolerance_grad=1e-7,
#                 tolerance_change=1e-9,
#                 history_size=10,
#             )
    
#             # L-BFGS는 closure를 여러 번 호출하므로 outer loop로 감싸서 max_eval 제어
#             n_outer = max(max_eval // 20, 5)

#             for _ in range(n_outer):
#                 def closure():
#                     optimizer.zero_grad()
#                     # gen 변수를 X_base에 삽입
#                     X_full = X_base.clone()
#                     X_full[:, gen_idx] = X_gen
#                     T_pred = fwd(X_full, B_rep)
#                     # 각 시작점별 MSE를 합산 (개별 loss의 합 → 각각 독립 최적화 효과)
#                     loss = nn.functional.mse_loss(T_pred, T_rep, reduction='sum')
#                     loss.backward()
#                     return loss
    
#                 optimizer.step(closure)
    
#                 # bound clamp & 현재 best 갱신 (시작점별 개별 loss 비교)
#                 with torch.no_grad():
#                     X_gen.data.clamp_(lb_t, ub_t)
#                     X_full = X_base.clone()
#                     X_full[:, gen_idx] = X_gen
#                     T_pred = fwd(X_full, B_rep)
#                     losses = nn.functional.mse_loss(
#                         T_pred, T_rep, reduction='none'
#                     ).view(n_pts, -1).mean(dim=1)           # (n_points,)
#                     min_loss, min_idx = losses.min(dim=0)
#                     if min_loss.item() < best_loss:
#                         best_loss = min_loss.item()
#                         best_gen  = X_gen.data[min_idx].clone()
        
#         elif trainer.opt.model_cfg.optim.method == 'adam':
            
#             optimizer = torch.optim.Adam([X_gen], lr = trainer.opt.model_cfg.optim.lr)
            
#             pbar_inner = tqdm(range(1, max_eval+1), desc = 'Multi', ncols = 150, position = 1, leave = True)
#             for step in pbar_inner:
#                 optimizer.zero_grad()
    
#                 X_full = X_base.clone()
#                 X_full[:, gen_idx] = X_gen
#                 T_pred = fwd(X_full, B_rep)
#                 loss = nn.functional.mse_loss(T_pred, T_rep, reduction='sum')
#                 loss.backward()
#                 optimizer.step()
                
#                 # bound clamp & best 갱신
#                 with torch.no_grad():
#                     X_gen.data.clamp_(lb_t, ub_t)
#                     losses = nn.functional.mse_loss(
#                         T_pred, T_rep, reduction='none'
#                     ).view(n_pts, -1).mean(dim=1)
#                     min_loss, min_idx = losses.min(dim=0)
#                     if min_loss.item() < best_loss:
#                         best_loss = min_loss.item()
#                         best_gen  = X_gen.data[min_idx].clone()
                        
#                         T_comp_pred = torch.complex(T_pred[:, 0], T_pred[:, 1])
#                         tr_pred    = torch.square(torch.abs(T_comp_pred))
#                         ph_pred    = torch.angle(T_comp_pred)
                        
#                         T_comp_tgt = torch.complex(T_rep[:, 0], T_rep[:, 1])
#                         tr_tgt     = torch.square(torch.abs(T_comp_tgt))
#                         ph_tgt     = torch.angle(T_comp_tgt)
    
#                         T_l1  = nn.L1Loss()(T_pred, T_rep).item()
#                         tr_l1 = nn.L1Loss()(tr_pred, tr_tgt).item()
#                         ph_l1 = nn.L1Loss()(ph_pred, ph_tgt).item()
                
#                 pbar_inner.set_postfix({
#                     'Trans. Coeff.': f'{T_l1:.2e}',
#                     'Transmittance': f'{tr_l1:.2e}',
#                     'Phase Shift': f'{ph_l1:.2e}'
#                 })
                        
#         # best_X 조립
#         best_X = X_template.clone()
#         best_X[0, gen_idx] = best_gen
#         return best_X, best_loss
 
#     # ── 메인 루프 ───────────────────────────────────────────────────
#     X_opt_list  = []
#     T_tgt_list  = []
#     B_tgt_list  = []
 
#     for X, T, _, B in trainer.test_dl:
#         X, T, B = X.to(device), T.to(device), B.to(device)
#         T_tgt_list.append(T)
#         B_tgt_list.append(B)
 
#         n_samples = X.shape[0]
#         pbar = tqdm(
#             range(n_samples), desc='Optim',
#             unit='sample', ncols=150, position=0, leave=True,
#         )
#         for i in pbar:
#             X_template = X[i : i + 1]
#             T_target   = T[i : i + 1]
#             B_target   = B[i : i + 1]
 
#             rng   = np.random.default_rng(seed=i + 1)
#             inits = rng.uniform(lb, ub, size=(n_points, n_gen))
 
#             best_X, best_f = optimize_single_sample(
#                 X_template, T_target, B_target, inits,
#             )
 
#             pbar.set_postfix({
#                 'X(init)':  [round(v, 2) for v in inits[0].tolist()],
#                 'X(optim)': [round(v, 2) for v in
#                              best_X[0, gen_idx].detach().cpu().tolist()],
#                 'loss': f'{best_f:.2e}',
#             })
#             X_opt_list.append(best_X)
 
#     # ── 평가 ────────────────────────────────────────────────────────
#     with torch.no_grad():
#         X_opt_torch = torch.cat(X_opt_list, dim=0)
#         T_tgt_torch = torch.cat(T_tgt_list, dim=0)
#         B_tgt_torch = torch.cat(B_tgt_list, dim=0)
 
#         T_pred     = fwd(X_opt_torch, B_tgt_torch)
#         T_comp_pred = torch.complex(T_pred[:, 0], T_pred[:, 1])
#         tr_pred    = torch.square(torch.abs(T_comp_pred))
#         ph_pred    = torch.angle(T_comp_pred)
 
#         T_comp_tgt = torch.complex(T_tgt_torch[:, 0], T_tgt_torch[:, 1])
#         tr_tgt     = torch.square(torch.abs(T_comp_tgt))
#         ph_tgt     = torch.angle(T_comp_tgt)
 
#         T_l1  = nn.L1Loss()(T_pred, T_tgt_torch).item()
#         tr_l1 = nn.L1Loss()(tr_pred, tr_tgt).item()
#         ph_l1 = nn.L1Loss()(ph_pred, ph_tgt).item()
 
#     print(
#         'MAE of Transmission Coeff. | Transmittance | Phase Shift :',
#         f'{T_l1:.2e} | {tr_l1:.2e} | {ph_l1:.2e}\n',
#     )
 
#     X_opt = torch.stack(X_opt_list)
#     return X_opt

def grad_based_optim(trainer):
    fwd = trainer.best_FWD
    fwd.eval()

    for param in fwd.parameters():
        param.requires_grad = False

    # ── config ──────────────────────────────────────────────────────
    n_points = trainer.opt.model_cfg.optim.n_points
    gen_idx  = trainer.opt.gen_idx
    n_gen    = len(gen_idx)
    device   = trainer.opt.device
    max_eval = getattr(trainer.opt.model_cfg.optim, 'max_eval', 1000)
    save_dir = getattr(trainer.opt, 'save_dir', './results')
    os.makedirs(save_dir, exist_ok=True)

    if trainer.opt.gen_variables == 'l2l5':
        lb = np.array([0.1, 0.1])
        ub = np.array([5.0, 1.8])

    lb_t = torch.tensor(lb, dtype=torch.float32, device=device)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=device)

    # ── helper: 단일 샘플에 대해 n_points 시작점을 배치로 최적화 ──
    def optimize_single_sample(X_template, T_target, B_target, inits):
        n_pts = inits.shape[0]

        X_gen = torch.tensor(inits, dtype=torch.float32, device=device)
        X_gen = nn.Parameter(X_gen)

        X_base = X_template.expand(n_pts, -1).clone()
        T_rep  = T_target.expand(n_pts, *[-1] * (T_target.dim() - 1))
        B_rep  = B_target.expand(n_pts, *[-1] * (B_target.dim() - 1))

        best_loss = float('inf')
        best_gen  = X_gen.data.clone()
        best_idx  = 0

        # ── trajectory 기록: (n_points, n_steps, n_gen) ──
        trajectories = [[inits[j].tolist()] for j in range(n_pts)]
        loss_histories = [[] for _ in range(n_pts)]

        if trainer.opt.model_cfg.optim.method == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                [X_gen],
                lr=trainer.opt.model_cfg.optim.lr,
                max_iter=20,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=10,
            )
            n_outer = max(max_eval // 20, 5)

            for _ in range(n_outer):
                def closure():
                    optimizer.zero_grad()
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = fwd(X_full)
                    loss = nn.functional.mse_loss(T_pred, T_rep, reduction='sum')
                    loss.backward()
                    return loss

                optimizer.step(closure)

                with torch.no_grad():
                    X_gen.data.clamp_(lb_t, ub_t)
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = fwd(X_full)
                    losses = nn.functional.mse_loss(
                        T_pred, T_rep, reduction='none'
                    ).view(n_pts, -1).mean(dim=1)

                    # trajectory 기록
                    current_pos = X_gen.data.detach().cpu().numpy()
                    current_losses = losses.detach().cpu().numpy()
                    for j in range(n_pts):
                        trajectories[j].append(current_pos[j].tolist())
                        loss_histories[j].append(current_losses[j])

                    min_loss, min_idx_t = losses.min(dim=0)
                    if min_loss.item() < best_loss:
                        best_loss = min_loss.item()
                        best_gen  = X_gen.data[min_idx_t].clone()
                        best_idx  = min_idx_t.item()

        elif trainer.opt.model_cfg.optim.method == 'adam':
            optimizer = torch.optim.Adam([X_gen], lr=trainer.opt.model_cfg.optim.lr)

            pbar_inner = tqdm(range(1, max_eval + 1), desc='Multi', ncols=150, position=1, leave=False)
            for step in pbar_inner:
                optimizer.zero_grad()

                X_full = X_base.clone()
                X_full[:, gen_idx] = X_gen
                T_pred = fwd(X_full)
                loss = nn.functional.mse_loss(T_pred, T_rep, reduction='sum')
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    X_gen.data.clamp_(lb_t, ub_t)
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = fwd(X_full)
                    losses = nn.functional.mse_loss(
                        T_pred, T_rep, reduction='none'
                    ).view(n_pts, -1).mean(dim=1)

                    # trajectory 기록
                    current_pos = X_gen.data.detach().cpu().numpy()
                    current_losses = losses.detach().cpu().numpy()
                    for j in range(n_pts):
                        trajectories[j].append(current_pos[j].tolist())
                        loss_histories[j].append(current_losses[j])

                    min_loss, min_idx_t = losses.min(dim=0)
                    if min_loss.item() < best_loss:
                        best_loss = min_loss.item()
                        best_gen  = X_gen.data[min_idx_t].clone()
                        best_idx  = min_idx_t.item()

                        T_comp_pred = torch.complex(T_pred[:, 0], T_pred[:, 1])
                        tr_pred = torch.square(torch.abs(T_comp_pred))
                        ph_pred = torch.angle(T_comp_pred)

                        T_comp_tgt = torch.complex(T_rep[:, 0], T_rep[:, 1])
                        tr_tgt = torch.square(torch.abs(T_comp_tgt))
                        ph_tgt = torch.angle(T_comp_tgt)

                        T_l1  = nn.L1Loss()(T_pred, T_rep).item()
                        tr_l1 = nn.L1Loss()(tr_pred, tr_tgt).item()
                        ph_l1 = nn.L1Loss()(ph_pred, ph_tgt).item()

                pbar_inner.set_postfix({
                    'Trans. Coeff.': f'{T_l1:.2e}',
                    'Transmittance': f'{tr_l1:.2e}',
                    'Phase Shift': f'{ph_l1:.2e}'
                })

        # numpy 변환
        trajectories  = [np.array(t) for t in trajectories]   # list of (n_steps, n_gen)
        loss_histories = [np.array(l) for l in loss_histories]  # list of (n_steps,)

        best_X = X_template.clone()
        best_X[0, gen_idx] = best_gen
        return best_X, best_loss, trajectories, loss_histories, best_idx

    # ── plotting ────────────────────────────────────────────────────
    def plot_trajectories(trajectories, loss_histories, best_idx, sample_idx, save_dir):
        """
        trajectories  : list of (n_steps, 2) arrays,  len = n_points
        loss_histories : list of (n_steps,) arrays
        best_idx       : index of best starting point
        """
        n_pts = len(trajectories)
        cmap = plt.cm.tab20 if n_pts > 10 else plt.cm.tab10
        colors = [cmap(i % cmap.N) for i in range(n_pts)]

        gen_labels = [f'gen_idx[{g}]' for g in gen_idx]

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # ── (a) 2D path plot ───────────────────────────────────────
        ax = axes[0]
        # boundary box
        ax.add_patch(plt.Rectangle(
            (lb[0], lb[1]), ub[0] - lb[0], ub[1] - lb[1],
            fill=False, edgecolor='gray', linestyle='--', linewidth=1.5, label='Bounds',
        ))

        for j in range(n_pts):
            traj = trajectories[j]  # (n_steps, 2)
            alpha = 1.0 if j == best_idx else 0.35
            lw    = 2.5 if j == best_idx else 1.0
            zorder = 10  if j == best_idx else 1

            ax.plot(traj[:, 0], traj[:, 1],
                    color=colors[j], alpha=alpha, lw=lw, zorder=zorder,
                    label=f'pt {j} (best)' if j == best_idx else None)
            # start marker
            ax.scatter(traj[0, 0], traj[0, 1],
                       color=colors[j], marker='o', s=50, edgecolors='k',
                       alpha=alpha, zorder=zorder + 1)
            # end marker
            ax.scatter(traj[-1, 0], traj[-1, 1],
                       color=colors[j], marker='*', s=150, edgecolors='k',
                       alpha=alpha, zorder=zorder + 1)

        ax.set_xlabel(gen_labels[0], fontsize=12)
        ax.set_ylabel(gen_labels[1], fontsize=12)
        ax.set_title(f'Optimization Paths  (sample {sample_idx})', fontsize=13)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim(lb[0] - 0.2, ub[0] + 0.2)
        ax.set_ylim(lb[1] - 0.1, ub[1] + 0.1)
        ax.grid(True, alpha=0.3)

        # ── (b) loss convergence ───────────────────────────────────
        ax2 = axes[1]
        for j in range(n_pts):
            alpha = 1.0 if j == best_idx else 0.3
            lw    = 2.5 if j == best_idx else 1.0
            ax2.plot(loss_histories[j],
                     color=colors[j], alpha=alpha, lw=lw,
                     label=f'pt {j} (best)' if j == best_idx else None)

        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('MSE Loss', fontsize=12)
        ax2.set_title(f'Loss Convergence  (sample {sample_idx})', fontsize=13)
        ax2.set_yscale('log')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        opt_num = f'{max_eval // 1000}k' if max_eval >= 1000 else f'{max_eval}'
        path = os.path.join(save_dir, f'{trainer.opt.model_cfg.optim.method}_path_sample_{sample_idx}_opt_{opt_num}.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path

    # ── 메인 루프 ───────────────────────────────────────────────────
    X_opt_list  = []
    T_tgt_list  = []
    B_tgt_list  = []

    plot_interval = getattr(trainer.opt.model_cfg.optim, 'plot_interval', trainer.opt.data_cfg.test.n_data // 5)
    # plot_interval = None → 마지막 샘플만, int → 매 N번째 샘플마다 plot

    sample_counter = 0

    for X, T, _, B in trainer.test_dl:
        X, T, B = X.to(device), T.to(device), B.to(device)
        T_tgt_list.append(T)
        B_tgt_list.append(B)

        n_samples = X.shape[0]
        pbar = tqdm(
            range(n_samples), desc='Optim',
            unit='sample', ncols=150, position=0, leave=True,
        )
        for i in pbar:
            X_template = X[i : i + 1]
            T_target   = T[i : i + 1]
            B_target   = B[i : i + 1]

            rng   = np.random.default_rng(seed=i + 1)
            inits = rng.uniform(lb, ub, size=(n_points, n_gen))

            best_X, best_f, trajs, loss_hists, best_pt_idx = optimize_single_sample(
                X_template, T_target, B_target, inits,
            )

            # plot 저장 조건
            should_plot = (
                (plot_interval is not None and sample_counter % plot_interval == 0)
                or (plot_interval is None and i == n_samples - 1)
            )
            if should_plot:
                plot_trajectories(trajs, loss_hists, best_pt_idx, sample_counter, save_dir)

            sample_counter += 1

            pbar.set_postfix({
                'X(init)':  [round(v, 2) for v in inits[0].tolist()],
                'X(optim)': [round(v, 2) for v in
                             best_X[0, gen_idx].detach().cpu().tolist()],
                'loss': f'{best_f:.2e}',
            })
            X_opt_list.append(best_X)

    # ── 평가 ────────────────────────────────────────────────────────
    with torch.no_grad():
        X_opt_torch = torch.cat(X_opt_list, dim=0)
        T_tgt_torch = torch.cat(T_tgt_list, dim=0)
        B_tgt_torch = torch.cat(B_tgt_list, dim=0)

        T_pred      = fwd(X_opt_torch)
        T_comp_pred = torch.complex(T_pred[:, 0], T_pred[:, 1])
        tr_pred     = torch.square(torch.abs(T_comp_pred))
        ph_pred     = torch.angle(T_comp_pred)

        T_comp_tgt = torch.complex(T_tgt_torch[:, 0], T_tgt_torch[:, 1])
        tr_tgt     = torch.square(torch.abs(T_comp_tgt))
        ph_tgt     = torch.angle(T_comp_tgt)

        T_l1  = nn.L1Loss()(T_pred, T_tgt_torch).item()
        tr_l1 = nn.L1Loss()(tr_pred, tr_tgt).item()
        ph_l1 = nn.L1Loss()(ph_pred, ph_tgt).item()

    print(
        'MAE of Transmission Coeff. | Transmittance | Phase Shift :',
        f'{T_l1:.2e} | {tr_l1:.2e} | {ph_l1:.2e}\n',
    )

    X_opt = torch.stack(X_opt_list)
    return X_opt
        
# def grad_based_optim_practical_case(fwd, profile):
def grad_based_optim(trainer):
    fwd = trainer.best_FWD
    fwd.eval()
 
    for param in fwd.parameters():
        param.requires_grad = False
 
    # ── config ──────────────────────────────────────────────────────
    n_points = trainer.opt.model_cfg.optim.n_points
    gen_idx  = trainer.opt.gen_idx
    n_gen    = len(gen_idx)
    device   = trainer.opt.device
    max_eval = getattr(trainer.opt.model_cfg.optim, 'max_eval', 1000)
    save_dir = getattr(trainer.opt, 'save_dir', './results')
    os.makedirs(save_dir, exist_ok=True)
 
    if trainer.opt.gen_variables == 'l2l5':
        lb = np.array([0.1, 0.1])
        ub = np.array([5.0, 1.8])
 
    lb_t = torch.tensor(lb, dtype=torch.float32, device=device)
    ub_t = torch.tensor(ub, dtype=torch.float32, device=device)
 
    # ── helper: 단일 샘플에 대해 n_points 시작점을 배치로 최적화 ──
    def optimize_single_sample(X_template, T_target, B_target, inits):
        n_pts = inits.shape[0]
 
        X_gen = torch.tensor(inits, dtype=torch.float32, device=device)
        X_gen = nn.Parameter(X_gen)
 
        X_base = X_template.expand(n_pts, -1).clone()
        T_rep  = T_target.expand(n_pts, *[-1] * (T_target.dim() - 1))
        B_rep  = B_target.expand(n_pts, *[-1] * (B_target.dim() - 1))
 
        best_loss = float('inf')
        best_gen  = X_gen.data.clone()
        best_idx  = 0
 
        # ── trajectory 기록 ──
        trajectories = [[inits[j].tolist()] for j in range(n_pts)]
        loss_histories = [[] for _ in range(n_pts)]
 
        if trainer.opt.model_cfg.optim.method == 'lbfgs':
            optimizer = torch.optim.LBFGS(
                [X_gen],
                lr=trainer.opt.model_cfg.optim.lr,
                max_iter=20,
                tolerance_grad=1e-7,
                tolerance_change=1e-9,
                history_size=10,
            )
            n_outer = max(max_eval // 20, 5)
 
            for _ in range(n_outer):
                def closure():
                    optimizer.zero_grad()
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = fwd(X_full)
                    loss = nn.functional.mse_loss(T_pred, T_rep, reduction='sum')
                    loss.backward()
                    return loss
 
                optimizer.step(closure)
 
                with torch.no_grad():
                    X_gen.data.clamp_(lb_t, ub_t)
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = fwd(X_full)
                    losses = nn.functional.mse_loss(
                        T_pred, T_rep, reduction='none'
                    ).view(n_pts, -1).mean(dim=1)
 
                    current_pos = X_gen.data.detach().cpu().numpy()
                    current_losses = losses.detach().cpu().numpy()
                    for j in range(n_pts):
                        trajectories[j].append(current_pos[j].tolist())
                        loss_histories[j].append(current_losses[j])
 
                    min_loss, min_idx_t = losses.min(dim=0)
                    if min_loss.item() < best_loss:
                        best_loss = min_loss.item()
                        best_gen  = X_gen.data[min_idx_t].clone()
                        best_idx  = min_idx_t.item()
 
        elif trainer.opt.model_cfg.optim.method == 'adam':
            optimizer = torch.optim.Adam([X_gen], lr=trainer.opt.model_cfg.optim.lr)
 
            pbar_inner = tqdm(range(1, max_eval + 1), desc='Multi', ncols=150, position=1, leave=False)
            for step in pbar_inner:
                optimizer.zero_grad()
 
                X_full = X_base.clone()
                X_full[:, gen_idx] = X_gen
                T_pred = fwd(X_full)
                loss = nn.functional.mse_loss(T_pred, T_rep, reduction='sum')
                loss.backward()
                optimizer.step()
 
                with torch.no_grad():
                    X_gen.data.clamp_(lb_t, ub_t)
                    X_full = X_base.clone()
                    X_full[:, gen_idx] = X_gen
                    T_pred = fwd(X_full)
                    losses = nn.functional.mse_loss(
                        T_pred, T_rep, reduction='none'
                    ).view(n_pts, -1).mean(dim=1)
 
                    current_pos = X_gen.data.detach().cpu().numpy()
                    current_losses = losses.detach().cpu().numpy()
                    for j in range(n_pts):
                        trajectories[j].append(current_pos[j].tolist())
                        loss_histories[j].append(current_losses[j])
 
                    min_loss, min_idx_t = losses.min(dim=0)
                    if min_loss.item() < best_loss:
                        best_loss = min_loss.item()
                        best_gen  = X_gen.data[min_idx_t].clone()
                        best_idx  = min_idx_t.item()
 
                        T_comp_pred = torch.complex(T_pred[:, 0], T_pred[:, 1])
                        tr_pred = torch.square(torch.abs(T_comp_pred))
                        ph_pred = torch.angle(T_comp_pred)
 
                        T_comp_tgt = torch.complex(T_rep[:, 0], T_rep[:, 1])
                        tr_tgt = torch.square(torch.abs(T_comp_tgt))
                        ph_tgt = torch.angle(T_comp_tgt)
 
                        T_l1  = nn.L1Loss()(T_pred, T_rep).item()
                        tr_l1 = nn.L1Loss()(tr_pred, tr_tgt).item()
                        ph_l1 = nn.L1Loss()(ph_pred, ph_tgt).item()
 
                pbar_inner.set_postfix({
                    'Trans. Coeff.': f'{T_l1:.2e}',
                    'Transmittance': f'{tr_l1:.2e}',
                    'Phase Shift': f'{ph_l1:.2e}'
                })
 
        trajectories   = [np.array(t) for t in trajectories]
        loss_histories = [np.array(l) for l in loss_histories]
 
        best_X = X_template.clone()
        best_X[0, gen_idx] = best_gen
        return best_X, best_loss, trajectories, loss_histories, best_idx
 
    # ── landscape 계산 ──────────────────────────────────────────────
    def compute_landscape(X_template, B_target, grid_res=80):
        """
        설계 변수 공간을 grid로 나눠 forward model 평가 →
        각 grid 점에서의 transmittance, phase shift 반환
        """
        g0 = np.linspace(lb[0], ub[0], grid_res)
        g1 = np.linspace(lb[1], ub[1], grid_res)
        G0, G1 = np.meshgrid(g0, g1)                      # (grid_res, grid_res)
 
        # grid points → flat (grid_res^2, n_gen)
        grid_flat = np.stack([G0.ravel(), G1.ravel()], axis=1)
        n_grid = grid_flat.shape[0]
 
        with torch.no_grad():
            X_base = X_template.expand(n_grid, -1).clone()
            X_base[:, gen_idx] = torch.tensor(grid_flat, dtype=torch.float32, device=device)
            B_rep = B_target.expand(n_grid, *[-1] * (B_target.dim() - 1))
 
            T_pred = fwd(X_base)
            T_comp = torch.complex(T_pred[:, 0], T_pred[:, 1])
            tr  = torch.square(torch.abs(T_comp)).cpu().numpy()    # transmittance
            ph  = torch.angle(T_comp).cpu().numpy()                # phase shift
 
        TR = tr.reshape(grid_res, grid_res)
        PH = ph.reshape(grid_res, grid_res)
        return G0, G1, TR, PH
 
    # ── plotting ────────────────────────────────────────────────────
    def plot_trajectories(trajectories, loss_histories, best_idx,
                          sample_idx, save_dir, G0, G1, TR, PH):
        n_pts = len(trajectories)
 
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        gen_labels = [f'gen_idx[{g}]' for g in gen_idx]
 
        # ── 좌측: Transmittance landscape (blue) + paths ──────────
        ax = axes[0]
        cf = ax.contourf(G0, G1, TR, levels=50, cmap='Blues', alpha=0.85)
        ax.contour(G0, G1, TR, levels=15, colors='steelblue', linewidths=0.4, alpha=0.5)
        cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label('Transmittance', fontsize=11)
 
        _draw_paths(ax, trajectories, best_idx, n_pts)
 
        ax.set_xlabel(gen_labels[0], fontsize=12)
        ax.set_ylabel(gen_labels[1], fontsize=12)
        ax.set_title(f'Transmittance Landscape  (sample {sample_idx})', fontsize=13)
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
 
        # ── 우측: Phase Shift landscape (red) + paths ─────────────
        ax2 = axes[1]
        cf2 = ax2.contourf(G0, G1, PH, levels=50, cmap='Reds', alpha=0.85)
        ax2.contour(G0, G1, PH, levels=15, colors='firebrick', linewidths=0.4, alpha=0.5)
        cbar2 = fig.colorbar(cf2, ax=ax2, shrink=0.85, pad=0.02)
        cbar2.set_label('Phase Shift (rad)', fontsize=11)
 
        _draw_paths(ax2, trajectories, best_idx, n_pts)
 
        ax2.set_xlabel(gen_labels[0], fontsize=12)
        ax2.set_ylabel(gen_labels[1], fontsize=12)
        ax2.set_title(f'Phase Shift Landscape  (sample {sample_idx})', fontsize=13)
        ax2.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax2.set_xlim(lb[0], ub[0])
        ax2.set_ylim(lb[1], ub[1])
 
        plt.tight_layout()
        opt_num = f'{max_eval // 1000}k' if max_eval >= 1000 else f'{max_eval}'
        path = os.path.join(
            save_dir,
            f'{trainer.opt.model_cfg.optim.method}_landscape_sample_{sample_idx}_opt_{opt_num}.png',
        )
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return path
 
    def _draw_paths(ax, trajectories, best_idx, n_pts):
        """경로를 배경 위에 잘 보이도록 그리기"""
        for j in range(n_pts):
            traj = trajectories[j]
            is_best = (j == best_idx)
            alpha  = 1.0  if is_best else 0.5
            lw     = 3.0  if is_best else 1.2
            zorder = 10   if is_best else 5
            color  = 'lime' if is_best else 'white'
            edge   = 'black'
 
            # path line — 검은 테두리 + 색상 선 (가시성 확보)
            ax.plot(traj[:, 0], traj[:, 1],
                    color='black', lw=lw + 1.5, alpha=alpha * 0.6, zorder=zorder - 1)
            ax.plot(traj[:, 0], traj[:, 1],
                    color=color, lw=lw, alpha=alpha, zorder=zorder,
                    label=f'pt {j} (best)' if is_best else (f'pt {j}' if n_pts <= 10 else None))
 
            # start marker (○)
            ax.scatter(traj[0, 0], traj[0, 1],
                       color=color, marker='o', s=60, edgecolors=edge,
                       linewidths=1.2, alpha=alpha, zorder=zorder + 1)
            # end marker (★)
            ax.scatter(traj[-1, 0], traj[-1, 1],
                       color=color, marker='*', s=200, edgecolors=edge,
                       linewidths=1.0, alpha=alpha, zorder=zorder + 1)
 
    # ── 메인 루프 ───────────────────────────────────────────────────
    X_opt_list  = []
    T_tgt_list  = []
    B_tgt_list  = []
 
    plot_interval = getattr(trainer.opt.model_cfg.optim, 'plot_interval',
                            trainer.opt.data_cfg.test.n_data // 5)
 
    sample_counter = 0
 
    for X, T, _, B in trainer.test_dl:
        X, T, B = X.to(device), T.to(device), B.to(device)
        T_tgt_list.append(T)
        B_tgt_list.append(B)
 
        n_samples = X.shape[0]
        pbar = tqdm(
            range(n_samples), desc='Optim',
            unit='sample', ncols=150, position=0, leave=True,
        )
        for i in pbar:
            X_template = X[i : i + 1]
            T_target   = T[i : i + 1]
            B_target   = B[i : i + 1]
 
            rng   = np.random.default_rng(seed=i + 1)
            inits = rng.uniform(lb, ub, size=(n_points, n_gen))
 
            best_X, best_f, trajs, loss_hists, best_pt_idx = optimize_single_sample(
                X_template, T_target, B_target, inits,
            )
 
            # plot 저장 조건
            should_plot = (
                (plot_interval is not None and sample_counter % plot_interval == 0)
                or (plot_interval is None and i == n_samples - 1)
            )
            if should_plot:
                G0, G1, TR, PH = compute_landscape(X_template, B_target)
                plot_trajectories(trajs, loss_hists, best_pt_idx,
                                  sample_counter, save_dir, G0, G1, TR, PH)
 
            sample_counter += 1
 
            pbar.set_postfix({
                'X(init)':  [round(v, 2) for v in inits[0].tolist()],
                'X(optim)': [round(v, 2) for v in
                             best_X[0, gen_idx].detach().cpu().tolist()],
                'loss': f'{best_f:.2e}',
            })
            X_opt_list.append(best_X)
 
    # ── 평가 ────────────────────────────────────────────────────────
    with torch.no_grad():
        X_opt_torch = torch.cat(X_opt_list, dim=0)
        T_tgt_torch = torch.cat(T_tgt_list, dim=0)
        B_tgt_torch = torch.cat(B_tgt_list, dim=0)
 
        T_pred      = fwd(X_opt_torch)
        T_comp_pred = torch.complex(T_pred[:, 0], T_pred[:, 1])
        tr_pred     = torch.square(torch.abs(T_comp_pred))
        ph_pred     = torch.angle(T_comp_pred)
 
        T_comp_tgt = torch.complex(T_tgt_torch[:, 0], T_tgt_torch[:, 1])
        tr_tgt     = torch.square(torch.abs(T_comp_tgt))
        ph_tgt     = torch.angle(T_comp_tgt)
 
        T_l1  = nn.L1Loss()(T_pred, T_tgt_torch).item()
        tr_l1 = nn.L1Loss()(tr_pred, tr_tgt).item()
        ph_l1 = nn.L1Loss()(ph_pred, ph_tgt).item()
 
    print(
        'MAE of Transmission Coeff. | Transmittance | Phase Shift :',
        f'{T_l1:.2e} | {tr_l1:.2e} | {ph_l1:.2e}\n',
    )
 
    X_opt = torch.stack(X_opt_list)
    return X_opt
 