import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Key, Query, Value projections
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Project and reshape to multi-head format
        q = self.query(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.proj(out)
        
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.net(x)
        
class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, n_heads, dropout)
        self.ff = FeedForward(dim, dim * 4, dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x, mask=None):
        # Pre-LayerNorm transformer block
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x
        
class GPT2Model(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        dim: int = 768, 
        n_layers: int = 12, 
        n_heads: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(dim, n_heads, dropout) for _ in range(n_layers)]
        )
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)
        
        # Output head
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        
        # Tie weights between token embedding and lm head
        self.token_embedding.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
    def get_causal_mask(self, seq_len):
        # Create a lower triangular mask for causal attention
        mask = torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len)
        return mask
            
    def forward(
        self, 
        input_ids: torch.LongTensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        return_logits: bool = True
    ):
        batch_size, seq_len = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # Get token and position embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids)
        
        # Add embeddings and apply dropout
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask if attention_mask is not provided
        if attention_mask is None:
            attention_mask = self.get_causal_mask(seq_len).to(input_ids.device)
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attention_mask)
            
        # Apply final layer norm
        x = self.ln_f(x)
        
        if return_logits:
            logits = self.lm_head(x)
            return logits
            
        return x
        
def create_model_from_config(config: Dict[str, Any]) -> GPT2Model:
    """Create a GPT2 model from a configuration dictionary."""
    model_config = config.get("model", {})
    
    model = GPT2Model(
        vocab_size=model_config.get("vocab_size", 50257),
        dim=model_config.get("dim", 768),
        n_layers=model_config.get("n_layers", 12),
        n_heads=model_config.get("n_heads", 12),
        max_seq_len=model_config.get("max_seq_len", 1024),
        dropout=model_config.get("dropout", 0.1)
    )
    
    return model 