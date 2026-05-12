import torch
import torch.nn as nn
import math

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, output_dim, context_dim, num_heads = 4):
        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = context_dim // num_heads
        # breakpoint()
        assert context_dim % num_heads == 0, "context_dim must be divisible by num_heads"
        bias = False

        # Multi-head attention projections
        self.query_proj = nn.Linear(self.context_dim, self.context_dim, bias = bias)
        self.key_proj = nn.Linear(self.context_dim, self.context_dim, bias = bias)
        self.value_proj = nn.Linear(self.context_dim, self.context_dim, bias = bias)
        
        # Output projection (마지막 linear projection)
        self.output_proj = nn.Linear(self.context_dim, self.context_dim, bias = bias)

        self.softmax = nn.Softmax(dim = -1)
        self.norm_layer = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(self.context_dim * self.output_dim)
        )
        # self.norm_layer = nn.Sequential(
        #     nn.LayerNorm(self.context_dim),
        #     nn.Flatten()
        # )
        
        # self.norm_layer = nn.LayerNorm(self.context_dim)
        # self.norm_layer = nn.LayerNorm(self.context_idm*self.output_dim)


    def attentionblock(self, x, y): # x: (batch, input_dim, context_dim), y: (batch, output_dim, context_dim)
        batch_size = x.size(0)
        
        # Linear projections
        query = self.query_proj(y)  # (batch, output_dim, context_dim)
        key = self.key_proj(x)      # (batch, input_dim, context_dim)
        value = self.value_proj(x)  # (batch, input_dim, context_dim)
        
        # Reshape for multi-head attention
        # (batch, output_dim, context_dim) -> (batch, output_dim, num_heads, head_dim) -> (batch, num_heads, output_dim, head_dim)
        query = query.view(batch_size, self.output_dim, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, input_dim, context_dim) -> (batch, input_dim, num_heads, head_dim) -> (batch, num_heads, input_dim, head_dim)
        key = key.view(batch_size, self.input_dim, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, self.input_dim, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (batch, num_heads, output_dim, input_dim)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # attn_scores = torch.matmul(query, key.transpose(-2, -1)) / scale
        attn_weights = self.softmax(attn_scores)
        
        # Apply attention to values: (batch, num_heads, output_dim, head_dim)
        attn_output = torch.matmul(attn_weights, value)
        
        # Concatenate heads: (batch, output_dim, context_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, self.output_dim, self.context_dim
        )
        
        # Final linear projection (마지막 linear projection)
        output = self.output_proj(attn_output)
        
        # Store for debugging/visualization
        self.scores = attn_scores  # (batch, num_heads, output_dim, input_dim)
        self.attn_weights = attn_weights  # (batch, num_heads, output_dim, input_dim)
        self.multi_head_outputs = output  # (batch, output_dim, context_dim)
        
        return output

    def forward(self, x, y): # x: (batch, input_dim)
        attn = self.attentionblock(x = x, y = y) # (batch, output_dim, context_dim)
        res = y + attn
        output = self.norm_layer(res) # (batch, output_dim, context_dim) -> (batch, output_dim * context_dim)
        # output = self.norm_layer(nn.Flatten()(res))
        
        # attn = self.attentionblock(x = x, y =self.norm_layer(y))
        # output = (y + attn).view(-1, self.output_dim * self.context_dim)
        return output
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, seq_len): # embedding_dim: d_model이라고도 함, seq_len: max sequence length
        super(PositionalEncoding, self).__init__() # super().__init__()과 동일한데, super().__init()은 python3에서만 사용 가능
        # self.dropout = nn.Dropout(p = dropout)
        position = torch.arange(0, seq_len, dtype = torch.float).unsqueeze(1) # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))  # (embed_dim/2)

        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term) # (seq_len, embed_dim/2)
        pe[:, 1::2] = torch.cos(position * div_term) # (seq_len, embed_dim/2)
        pe = pe.unsqueeze(0) # (seq_len, embed_dim) -> (1, seq_len, embed_dim)
        
        self.register_buffer('pe', pe) # pe는 학습되는 parameter가 아니므로, register_buffer로 등록

    def forward(self, x):
        # return self.dropout(x + self.pe) # (1, seq_len, embed_dim)
        return x + self.pe
