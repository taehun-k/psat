import torch
import torch.nn as nn
from model.attention import MultiHeadAttention, PositionalEncoding
import functools

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

class AttnFWD(nn.Module):
    def __init__(self, opt):
        super(AttnFWD, self).__init__()
        self.opt = opt
        self.input_dim = opt.model_cfg['forward']['input_dim']
        self.hidden_dim = opt.model_cfg['forward']['attn_fwd']['hidden_dim']
        self.output_dim = opt.model_cfg['forward']['output_dim']
        self.context_dim = opt.model_cfg['forward']['attn_fwd']['context_dim']
        self.num_heads = opt.model_cfg['forward']['attn_fwd']['num_heads']
        self.num_attn = opt.model_cfg['forward']['attn_fwd']['num_attn']

        self.meff_idx = [3,4,5,6]; self.keff_idx = [0,1,2,7]
        self.embedding_meff = Model_blocks.positional_encoding_block_shared(context_dim = self.context_dim, seq_len = len(self.meff_idx))
        self.embedding_keff = Model_blocks.positional_encoding_block_shared(context_dim = self.context_dim, seq_len = len(self.keff_idx))
        # self.in_self_attn = MultiHeadAttention(input_dim = self.input_dim, output_dim = self.input_dim, context_dim = self.context_dim,\
        #                                                     num_heads = self.num_heads)
        self.in_self_attn = nn.ModuleList()
        self.in_self_attn.append(MultiHeadAttention(input_dim = self.input_dim, output_dim = self.input_dim, context_dim = self.context_dim,\
                                                            num_heads = self.num_heads))
        for _ in range(self.num_attn-1):
            self.in_self_attn.append(nn.Unflatten(1,(self.input_dim, self.context_dim)))
            self.in_self_attn.append(MultiHeadAttention(input_dim = self.input_dim, output_dim = self.input_dim, context_dim = self.context_dim,\
                                                        num_heads = self.num_heads))
        
        self.ffnn = nn.Sequential()
        # self.ffnn.flatten = nn.Flatten()
        self.ffnn.fc1 = Model_blocks.fc_block(self.input_dim * self.context_dim, self.hidden_dim[0], norm = self.opt.model_cfg['forward']['norm'])
        
        if len(self.hidden_dim) >= 2:
            for i in range(len(self.hidden_dim)-1):
                self.ffnn.add_module(f'fc{i+2}', Model_blocks.fc_block(in_channel = self.hidden_dim[i],\
                                                                    out_channel = self.hidden_dim[i+1], norm = self.opt.model_cfg['forward']['norm']))
                # setattr(self.ffnn, f'fc{i+2}', Model_blocks.fc_block(self.hidden_dim[i], self.hidden_dim[i+1], norm = self.opt.norm))

            self.ffnn.add_module(f'fc{i+3}', nn.Linear(self.hidden_dim[-1], self.output_dim, bias = False))
        elif len(self.hidden_dim) == 1: self.ffnn.add_module(f'fc2', nn.Linear(self.hidden_dim[-1], self.output_dim, bias = False))

    def forward(self, x):
        if len(x.shape) == 2: x = x.unsqueeze(2)
        embed_meff = self.embedding_meff(x[:,self.meff_idx]) # (B, 4, C)
        embed_keff = self.embedding_keff(x[:,self.keff_idx]) # (B, 4, C)
        embed_in = torch.empty(x.shape[0], x.shape[1], self.context_dim, device = device)
        embed_in[:,self.meff_idx] = embed_meff
        embed_in[:,self.keff_idx] = embed_keff
        # attn = self.in_self_attn(x = embed_in, y = embed_in) # (B, 8, C)
        attn = self.in_self_attn[0](x = embed_in, y = embed_in) # (B, 8, C)
        for layer in self.in_self_attn[1:]:
            if isinstance(layer, MultiHeadAttention): attn = layer(x = attn, y = attn)
            else: attn = layer(attn)
        logit = self.ffnn(attn)
        return logit
    
class BaseFWD(nn.Module):
    def __init__(self, opt):
        super(BaseFWD, self).__init__()
        self.opt = opt
        self.input_dim = opt.model_cfg['forward']['input_dim']
        self.hidden_dim = opt.model_cfg['forward']['base_fwd']['hidden_dim']
        self.output_dim = opt.model_cfg['forward']['output_dim']
        self.ffnn = nn.Sequential()
        self.ffnn.fc1 = Model_blocks.fc_block(self.input_dim, self.hidden_dim[0], norm = self.opt.model_cfg['forward']['norm'])

        for i in range(len(self.hidden_dim)-1):
            self.ffnn.add_module(f'fc{i+2}', Model_blocks.fc_block(in_channel = self.hidden_dim[i],\
                                                                   out_channel = self.hidden_dim[i+1], norm = self.opt.model_cfg['forward']['norm']))
            # setattr(self.ffnn, f'fc{i+2}', Model_blocks.fc_block(self.hidden_dim[i], self.hidden_dim[i+1], norm = self.opt.norm))

        self.ffnn.add_module(f'fc{i+3}', nn.Linear(self.hidden_dim[-1], self.output_dim, bias = False))

    def forward(self, x):
        logit = self.ffnn(x)
        return logit
    
class Model_blocks:

    def fc_block(in_channel, out_channel, norm, activation = 'relu'):
        net = nn.Sequential()
        net.add_module('1', nn.Linear(in_channel, out_channel, bias = False))

        if norm == 'LN': net.add_module('2', nn.LayerNorm(normalized_shape = out_channel))
        elif norm == 'BN': net.add_module('2', nn.BatchNorm1d(num_features = out_channel))
        elif norm == 'IN': net.add_module('2', nn.InstanceNorm1d(num_features = out_channel))
        # if self.exp['norm_G'] == None: setattr(net, '2', nn.LeakyReLU(True))
        if norm == None: 
            if activation == 'relu':
                net.add_module('2', nn.ReLU(True))
            elif activation == 'gelu':
                net.add_module('2', nn.GELU())
        # else: setattr(net, '3', nn.LeakyReLU(True))
        # else: net.add_module('3', nn.ReLU(True))
        else: 
            if activation == 'relu':
                net.add_module('3', nn.ReLU(True))
            elif activation == 'gelu':
                net.add_module('3', nn.GELU())
        return net
    
    class positional_encoding_block_sep(nn.Module): # sequence에 대해 각각 다른 embedding 적용
        def __init__(self, context_dim, seq_len):
            super().__init__()
            self.context_dim = context_dim
            self.seq_len = seq_len
            self.layer = nn.Sequential()
            self.layer.add_module('1', nn.ModuleList([nn.Linear(1, self.context_dim, bias = False) for _ in range(self.seq_len)]))
            self.layer.add_module('2', PositionalEncoding(embed_dim = self.context_dim, seq_len = self.seq_len))

        def forward(self, x):
            embed = torch.stack([layer(x[:, i, :]) for i, layer in enumerate(self.layer[0])], dim = 1)
            output = self.layer[1](embed)
            return output
        
    class positional_encoding_block_shared(nn.Module): # sequence에 대해 같은 embedding 적용
        def __init__(self, context_dim, seq_len):
            super().__init__()
            self.context_dim = context_dim
            self.seq_len = seq_len
            self.layer = nn.Sequential()
            self.layer.add_module('1', nn.Linear(1, self.context_dim, bias = False))
            self.layer.add_module('2', PositionalEncoding(embed_dim = self.context_dim, seq_len = self.seq_len))

        def forward(self, x):
            embed = self.layer[0](x)
            output = self.layer[1](embed)
            return output

    def residual_block(func): # func의 input과 output의 dimension이 같을 때만 사용 가능
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            input = args[0]
            residual = func(*args, **kwargs)
            output = input + residual
            return output
        return wrapper