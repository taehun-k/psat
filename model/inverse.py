import torch
import torch.nn as nn
from model.attention import PositionalEncoding, MultiHeadAttention
from torch.nn.utils import spectral_norm
from model.forward import Model_blocks

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class PSAAT(nn.Module):
    def __init__(self, opt):
        super(PSAAT, self).__init__()
        self.opt = opt
        self.PSAT = PSAT(opt = self.opt)
        self.D = Discriminator(opt = self.opt)

class PSAT(nn.Module):
    def __init__(self, opt):
        super(PSAT, self).__init__()
        self.opt = opt
        self.input_dim = self.opt.model_cfg['psat']['input_dim']
        self.hidden_dim = self.opt.model_cfg['psat']['hidden_dim']
        self.output_dim = self.opt.model_cfg['psat']['output_dim']
        self.num_heads = self.opt.model_cfg['psat']['num_heads']
        self.context_dim = self.opt.model_cfg['psat']['context_dim']
        self.model = nn.Sequential()
        
        meff_idx = [3,4,5,6]; keff_idx = [0,1,2,7]; 
        if int(self.opt.data_cfg.n_design_var) == 8:
            self.embed_meff_idx = meff_idx
            self.embed_keff_idx = keff_idx
        else:
            self.embed_meff_idx = [self.opt.fixed_idx.index(i) for i in meff_idx if i in self.opt.fixed_idx]
            self.embed_keff_idx = [self.opt.fixed_idx.index(i) for i in keff_idx if i in self.opt.fixed_idx]
        
        # 1. Embedding module
        self.embedding_meff = Model_blocks.positional_encoding_block_shared(context_dim = self.context_dim, seq_len = len(self.embed_meff_idx))
        self.embedding_keff = Model_blocks.positional_encoding_block_shared(context_dim = self.context_dim, seq_len = len(self.embed_keff_idx))
        self.embedding_transmission = Model_blocks.positional_encoding_block_shared(context_dim = self.context_dim, seq_len = 2)

        # 2. Self attention module (input)
        self.in_self_attn = MultiHeadAttention(input_dim = self.input_dim, output_dim = self.input_dim, context_dim = self.context_dim,\
                                                            num_heads = self.num_heads)
        # 3. Feed-forward neural network
        self.ffnn = nn.Sequential()
        if self.opt.model_cfg.psat.conditional:
            # self.ffnn.fc1 = Model_blocks.fc_block(in_channel = self.context_dim + self.opt.model_cfg.psat.latent_dim + 3,\
            #                                                     out_channel = self.hidden_dim[0], norm = self.opt.model_cfg.psat.norm)
            self.ffnn.fc1 = Model_blocks.fc_block(in_channel = self.context_dim + self.opt.model_cfg.psat.latent_dim + 1,\
                                                                out_channel = self.hidden_dim[0], norm = self.opt.model_cfg.psat.norm)
        else:
            self.ffnn.fc1 = Model_blocks.fc_block(in_channel = self.context_dim + self.opt.model_cfg.psat.latent_dim,\
                                                        out_channel = self.hidden_dim[0], norm = self.opt.model_cfg.psat.norm)
        for i in range(len(self.hidden_dim)-1):
            self.ffnn.add_module(f'fc{i+2}', Model_blocks.fc_block(in_channel = self.hidden_dim[i],\
                                                                   out_channel = self.hidden_dim[i+1], norm = opt.model_cfg.psat.norm))
        self.ffnn.add_module(f'fc{i+3}', nn.Sequential(nn.Flatten(), nn.Linear(self.hidden_dim[-1]*self.input_dim, self.output_dim, bias = False)))

        # 4. Embedding module for output
        self.embedding_out = Model_blocks.positional_encoding_block_sep(context_dim = self.context_dim, seq_len = self.output_dim)
        
        # 5. Self attention module (output)
        self.out_self_attn = MultiHeadAttention(input_dim = self.output_dim, output_dim = self.output_dim, context_dim = self.context_dim,\
                                                            num_heads = self.num_heads)
        # 6. Cross attention module
        self.cross_attn = MultiHeadAttention(input_dim = self.input_dim, output_dim = self.output_dim, context_dim = self.context_dim,\
                                                            num_heads = self.num_heads) # after 250116 12:45

        # 7. Refine network
        self.rfnn = nn.Sequential() # refine network
        if len(self.opt.model_cfg['psat']['refine_dim']) == 0:
            self.rfnn.fc1 = nn.Linear(self.context_dim, 1, bias = False) # 250309
        else:
            for i in range(len(self.opt.model_cfg['psat']['refine_dim'])-1):
                self.rfnn.add_module(f'fc{i+1}', Model_blocks.fc_block(in_channel = self.context_dim if i == 0 else self.opt.model_cfg['psat']['refine_dim'][i-1], 
                                                                     out_channel = self.opt.model_cfg['psat']['refine_dim'][i], norm = self.opt.model_cfg.psat.norm, activation = 'gelu'))

        print(f"Generator parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x, z, b): # z: (B, Z), b: (B, 3)
        # 1. Embedding for input
        if len(x.shape) == 2: x = x.unsqueeze(2) # (B, I) -> (B, I, 1)
        embed_meff = self.embedding_meff(x[:,self.embed_meff_idx]) # (B, (I-2) // 2, C)
        embed_keff = self.embedding_keff(x[:,self.embed_keff_idx]) # (B, (I-2) // 2, C)
        embed_transmission = self.embedding_transmission(x[:,-2:]) # (B, 2, C)
        embed_in = torch.empty(x.shape[0], x.shape[1], self.context_dim, device = device)
        embed_in[:,self.embed_meff_idx] = embed_meff
        embed_in[:,self.embed_keff_idx] = embed_keff
        embed_in[:,-2:] = embed_transmission
        
        # 2. Self-attention for input
        attn = self.in_self_attn(x = embed_in, y = embed_in).view(-1, self.input_dim, self.context_dim) # (B, I, C)

        # 3. Feed-forward neural network
        if self.opt.model_cfg.psat.conditional:

            # Transmission 실수부, 허수부에 동일한 z 복사 / 설계변수에는 0
            z = z.unsqueeze(1).repeat(1, 2, 1)
            z = torch.cat([torch.zeros(z.shape[0], self.input_dim - 2, z.shape[2], device = device), z], dim = 1) # (B, I, Z)

            # Bandgap Trial 2 (251006_0910) -> Best
            b = b.unsqueeze(1).repeat(1, 2, 1) # (B, BA) -> (B, 1, BA) -> (B, 2, BA)
            b = torch.cat([torch.zeros(b.shape[0], self.input_dim - 2, b.shape[2], device = device), b], dim = 1) # (B, I, BA)
            pred = self.ffnn(torch.cat([attn, z, b], dim = 2)) # (B, I, C+Z+BA) -> (B, I, H) -> (B, O)
        else:
            # Transmission 실수부, 허수부에 동일한 z 복사 / 설계변수에는 0
            z = z.unsqueeze(1).repeat(1, 2, 1)
            z = torch.cat([torch.zeros(z.shape[0], self.input_dim - 2, z.shape[2], device = device), z], dim = 1) # (B, I, Z)
            pred = self.ffnn(torch.cat([attn, z], dim = 2)) # (B, I, C+Z) -> (B, I, H) -> (B, O)

        # 4. Embedding for output
        pred_pseudo = pred
        embed_out = self.embedding_out(pred.unsqueeze(2))# (B, O, C)
        
        # 5. Self-attention for output
        pred_self_attn = self.out_self_attn(x = embed_out, y = embed_out).view(-1, self.output_dim, self.context_dim) # (B, O, C)

        # 6. Cross attention module
        pred_attn = self.cross_attn(x = attn, y = pred_self_attn).view(-1, self.output_dim, self.context_dim) # (B, O, C)

        # 7. Refine network
        pred_final = self.rfnn(pred_attn).squeeze() + pred
        # return nn.Softplus()(pred_pseudo), nn.Softplus()(pred_final)
        return nn.ReLU()(pred_pseudo), nn.ReLU()(pred_final)
    
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.opt = opt
        self.input_dim = self.opt.model_cfg['discriminator']['input_dim']
        self.hidden_dim = self.opt.model_cfg['discriminator']['hidden_dim']
        self.output_dim = self.opt.model_cfg['discriminator']['output_dim']
        self.num_heads = self.opt.model_cfg['discriminator']['num_heads']
        self.context_dim = self.opt.model_cfg['discriminator']['context_dim']
        self.meff_idx = [3,4,5,6]; self.keff_idx = [0,1,2,7]

        # 1. Embedding module
        self.embedding_meff = Model_blocks.positional_encoding_block_shared(context_dim = self.context_dim, seq_len = len(self.meff_idx))
        self.embedding_keff = Model_blocks.positional_encoding_block_shared(context_dim = self.context_dim, seq_len = len(self.keff_idx))
        self.embedding_transmission = Model_blocks.positional_encoding_block_shared(context_dim = self.context_dim, seq_len = 2)

        # 2. Self attention module (input)
        self.in_self_attn = MultiHeadAttention(input_dim = self.input_dim, output_dim = self.input_dim, context_dim = self.context_dim,\
                                                            num_heads = self.num_heads)

        # 3. Feed-forward neural network
        self.ffnn = nn.Sequential()
        if self.opt.model_cfg.discriminator.conditional:
            # self.ffnn.fc1 = Model_blocks.fc_block(in_channel = self.input_dim * self.context_dim + 3,\
            #                                       out_channel = self.hidden_dim[0], norm = self.opt.model_cfg.discriminator.norm) # 250
            self.ffnn.fc1 = Model_blocks.fc_block(in_channel = self.input_dim * self.context_dim + 1,\
                                                  out_channel = self.hidden_dim[0], norm = self.opt.model_cfg.discriminator.norm) # 250
        else:
            self.ffnn.fc1 = Model_blocks.fc_block(in_channel = self.input_dim * self.context_dim,\
                                                  out_channel = self.hidden_dim[0], norm = self.opt.model_cfg.discriminator.norm)        
        for i in range(len(self.hidden_dim)-1):
            self.ffnn.add_module(f'fc{i+2}', Model_blocks.fc_block(in_channel = self.hidden_dim[i],\
                                                                   out_channel = self.hidden_dim[i+1], norm = self.opt.model_cfg.discriminator.norm))
        self.ffnn.add_module(f'fc{i+3}', nn.Linear(self.hidden_dim[-1], self.output_dim, bias = False))
        if self.opt.model_cfg.name in ['sngan', 'psaat']:
            self = apply_spectral_norm(self)
        print(f"Discriminator parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}")

    def forward(self, x, b): # b: bandgap (1) or not (0)
        # 1. Embedding for input
        if len(x.shape) == 2: x = x.unsqueeze(2)
        embed_meff = self.embedding_meff(x[:,self.meff_idx]) # (B, 4, C)
        embed_keff = self.embedding_keff(x[:,self.keff_idx]) # (B, 4, C)
        embed_transmission = self.embedding_transmission(x[:,-2:]) # (B, 2, C)
        embed_in = torch.empty(x.shape[0], x.shape[1], self.context_dim, device = device) # (B, I, C)
        embed_in[:,self.meff_idx] = embed_meff
        embed_in[:,self.keff_idx] = embed_keff
        embed_in[:,-2:] = embed_transmission

        # 2. Self-attention for input
        attn = self.in_self_attn(x = embed_in, y = embed_in)

        # 3. Feed-forward neural network
        if self.opt.model_cfg.discriminator.conditional:
            logit = self.ffnn(torch.cat([attn, b], dim = 1))
        else:
            logit = self.ffnn(attn)
        return logit

def apply_spectral_norm(model):
    for name, child in model.named_children():
        if isinstance(child, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if not hasattr(child, 'weight_orig'):
                spectral_norm(child)
        else: 
            apply_spectral_norm(child)
    return model