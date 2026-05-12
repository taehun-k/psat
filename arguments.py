import torch
import argparse
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def get_args(): #

    # ------------------------------------------------------------------------------------------------------------------
    # 개별 변수
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=device, help='cuda or mps or cpu')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    # ------------------------------------------------------------------------------------------------------------------
    opt = parser.parse_args([])
    
    # Aluminum
    opt.material = {
        'rho': 2700, # Density
        'poisson': 0.33, # Poisson's ratio
        'E': 70e9, # Young's Modulus
        'H': 0.01, # Height of metasurface unit cell
        'W': 0.006, # Width of metasurface unit cell
        'd': 0.15,
    }
    
    return opt