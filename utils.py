import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from itertools import product
import numpy as np
import torch
from sklearn.decomposition import PCA
import functools
import torch
import torch.nn as nn
from copy import deepcopy

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def plot_learning_curve(hist:dict, xlabel=None, ylabel=None, save_path = None, ylim=None, colors = None):
    labels = list(hist.keys())
    for i in range(len(hist)):
        if torch.is_tensor(hist[labels[i]]):
            hist[labels[i]] = hist[labels[i]].detach().cpu().numpy()
    for i in range(len(hist)):
        if colors == None:
            plt.plot(range(len(hist[labels[i]])), hist[labels[i]], color = plt.cm.YlOrBr((i+2)*50), label = labels[i], alpha = 0.8)
        else:
            plt.plot(range(len(hist[labels[i]])), hist[labels[i]], color = colors[i], label = labels[i], alpha = 0.8)
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)
    if ylim: plt.ylim(ylim)
    # plt.legend(loc='upper right')
    plt.tick_params(axis='both', which='major', direction='in', labelbottom=False, labelleft=False)
    plt.tick_params(axis='both', which='minor', bottom=False, left=False)
    # plt.tick_params(axis = 'both', which = 'both', direction = 'in', labelbottom = False, labelleft = False)
    plt.tight_layout()
    
    if save_path is not None:
        print(f'Learning curve plot: {save_path}')
        plt.savefig(save_path, dpi = 500, transparent = True)
    plt.close('all')

def named_apply(module, fn, prefix=""):
    for name, child in module.named_children():
        fn(child, prefix + name)
        named_apply(child, fn, prefix + name + ".")

class Color:
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    END = '\033[0m'
    
def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    
    # 1. Linear, Conv 레이어 (직교 초기화 대상)
    if classname.find('Linear') != -1 or classname.find('Conv') != -1:
        if hasattr(m, 'weight_orig'): # Spectral Norm 적용 시
            nn.init.orthogonal_(m.weight_orig.data, gain=1.0)
        elif hasattr(m, 'weight'):
            nn.init.orthogonal_(m.weight.data, gain=1.0)
            
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
            
    # 2. LayerNorm 레이어 (Identity 초기화 대상)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.constant_(m.weight.data, 1.0) # Gamma = 1
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0) # Beta = 0

def weights_init(m):

    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight.data, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

def refracting_phase_profile(refracted_angle, **kwargs): # radian
    dy = kwargs['dy']; k_t = kwargs['k_t'] # transmitted angle
    # delta_y = 0.01 # Ly
    # k0 = 116.5
    dphi = dy * k_t * np.sin(refracted_angle)
    # phases = np.arange(-np.pi, np.pi, phase_unitcell)
    profile = np.arange(-26*dphi, 28*dphi, dphi)
    return profile

def phase_grad2refracted_angle(phase_grad, **kwargs): # dphi/dy
    # k0 = 116.5
    k0 = kwargs['k0']; theta_i = kwargs['theta_i']
    # theta_i = 0
    refracted_angle = np.arcsin(np.sin(theta_i)+phase_grad/k0)
    return refracted_angle

def focusing_phase_profile(y, **kwargs):
    k0 = kwargs['k0']; f_l = kwargs['f_l']
    lens = k0*(np.sqrt(f_l**2 + y**2) - f_l)
    lens_leftside = np.linspace(lens[0]/2, 0, 5)
    lens_rightside = np.linspace(0, lens[-1]/2, 5)
    lens = np.concatenate([lens_leftside, lens, lens_rightside])
    return lens

def create_grid_representation(n_grid, **design_params):
    Lx, Ly, l1, l2, l3, l4, l5, l6, l7, l8 = design_params.values()
    whole_grid = np.meshgrid(np.linspace(-Lx/2, Lx/2, n_grid), np.linspace(-Ly/2, Ly/2, n_grid))
    # 좌측 상단 slot
    slot1_x = (whole_grid[0] > -l3/2-2*l8-2*l1) & (whole_grid[0] < -l3/2-2*l8-l1)
    slot1_y = (whole_grid[1] > Ly/4-l2/2) & (whole_grid[1] < Ly/4+l2/2)
    # 좌측 하단 slot
    slot2_x = slot1_x
    slot2_y = (whole_grid[1] > -Ly/4-l2/2) & (whole_grid[1] < -Ly/4+l2/2)
    # 좌측 가운데 slot
    slot3_x = (whole_grid[0] > -l3/2-l8-l1) & (whole_grid[0] < -l3/2-l8)
    slot3_y = (whole_grid[1] > -l2/2) & (whole_grid[1] < l2/2)
    # 우측 가운데 slot
    slot4_x = (whole_grid[0] > l3/2+l8+l1) & (whole_grid[0] < l3/2+l8+2*l1)
    slot4_y = (whole_grid[1] > -l2/2) & (whole_grid[1] < l2/2)
    # 우측 상단 slot
    slot5_x = (whole_grid[0] > l3/2+2*l8+l1) & (whole_grid[0] < l3/2+2*l8+2*l1)
    slot5_y = (whole_grid[1] > Ly/4-l2/2) & (whole_grid[1] < Ly/4+l2/2)
    # 우측 하단 slot
    slot6_x = slot5_x
    slot6_y = (whole_grid[1] > -Ly/4-l2/2) & (whole_grid[1] < -Ly/4+l2/2)
    # 가운데 스프링 공백 부분
    spring_empty_x = (whole_grid[0] > -l3/2) & (whole_grid[0] < l3/2)
    spring_empty_y = (whole_grid[1] > -l4/2) & (whole_grid[1] < l4/2)
    # 가운데 스프링 부분
    spring_x = (whole_grid[0] >= -l6/2) & (whole_grid[0] <= l6/2)
    spring_y = (whole_grid[1] >= -l7/2) & (whole_grid[1] <= l7/2)
    # 스프링 연결 부분
    spring_connect_x = (whole_grid[0] >= -l5/2) & (whole_grid[0] <= l5/2)
    spring_connect_y = ((whole_grid[1] >= l7/2) & (whole_grid[1] <= l4/2)) | ((whole_grid[1] >= -l4/2) & (whole_grid[1] <= -l7/2))
    grid_idx = ~((slot1_x & slot1_y) | (slot2_x & slot2_y) | (slot3_x & slot3_y) | (slot4_x & slot4_y) | (slot5_x & slot5_y) | (slot6_x & slot6_y) | (spring_empty_x & spring_empty_y)) | (spring_x & spring_y) | (spring_connect_x & spring_connect_y)
    grid_representation = np.zeros_like(whole_grid[0])
    grid_representation[grid_idx] = 1
    # grid_position = np.column_stack((whole_grid[0][grid_idx], whole_grid[1][grid_idx]))
    grid_position = np.stack(whole_grid, axis= -1)[grid_idx]
    return np.stack(whole_grid, axis = -1), grid_position, grid_representation