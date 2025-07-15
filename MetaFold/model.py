# =================================================================================
# MetaFold-RNA: MetaLearnerNet v3.0

import torch
import torch.nn as nn
import torch.nn.functional as F

# =================================================================================
# Section 0: Relative Position Encoding
# ================================================================================= 
class RelativePositionEncoding(nn.Module):
    def __init__(self, bin_num, c_z):
        super().__init__()
        self.bins = torch.linspace(-32, 32, steps=bin_num)  # shape: [N_bins]
        self.linear = nn.Linear(len(self.bins), c_z)

    def forward(self, L, device):
        """
        residue_idx: [B, L]
        Output: [B, L, L, c_z]
        """
        residue_idx = torch.arange(L, device=device)
        residue_idx = residue_idx.unsqueeze(0) # [1, L]
        # B, L = residue_idx.shape
        diff = residue_idx.unsqueeze(2) - residue_idx.unsqueeze(1)  # [B, L, L]
        diff = diff.clamp(min=self.bins[0].item(), max=self.bins[-1].item())  # clip to bin range
        one_hot = one_hot_with_bins(diff, self.bins.to(residue_idx.device))   # [B, L, L, N_bins]
        return self.linear(one_hot)  # [B, L, L, c_z]

def one_hot_with_bins(x, bins):
    """
    x: [B, L, L], integer diffs
    bins: [N_bins], e.g., [-32, ..., 32]
    Output: one-hot encoding: [B, L, L, N_bins]
    """
    x = x.unsqueeze(-1)  # [B, L, L, 1]
    dist = torch.abs(x - bins.view(1, 1, 1, -1))  # [B, L, L, N_bins]
    min_idx = torch.argmin(dist, dim=-1)  # [B, L, L]
    one_hot = torch.nn.functional.one_hot(min_idx, num_classes=bins.shape[0])  # [B, L, L, N_bins]
    return one_hot.float()

# =================================================================================
# Section 1: 2D feture extraction blocks
# =================================================================================

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        squeezed_channels = max(1, int(in_channels * se_ratio))
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(in_channels, squeezed_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(squeezed_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution Block"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25, dropout_prob=0.2):
        super(MBConvBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.InstanceNorm2d(hidden_dim),
                nn.SiLU()
            ])
        
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.InstanceNorm2d(hidden_dim),
            nn.SiLU()
        ])
        
        layers.append(SEBlock(hidden_dim, se_ratio))
        
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        if self.use_residual:
            return self.dropout(x + self.block(x))
        else:
            return self.dropout(self.block(x))


# =================================================================================
# Section 2: Co-evolution Components
# =================================================================================

class OuterProductMean(nn.Module):
    """
    1D -> 2D outer product mean.
    """
    def __init__(self, in_channels, hidden_dim=32, out_channels=32):
        super().__init__()
        self.linear = nn.Linear(in_channels, hidden_dim)
        # self.linear2 = nn.Linear(in_channels, hidden_dim)
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * hidden_dim),
            nn.Linear(hidden_dim * hidden_dim, out_channels)
        )

    def forward(self, x):
        # x: [B, L, in_channels]
        a = self.linear(x) # -> [B, L, hidden_dim]

        outer = torch.einsum('bid,bje->bijde', a, a)
        outer = outer.flatten(start_dim=-2)
        pair_repr = self.projection(outer) # -> [B, L, L, out_channels]
        return pair_repr

class PairwiseBiasAttention(nn.Module):
    """
    Pairwise Bias Attention mechanism.
    This module computes attention scores based on pairwise representations.
    """
    def __init__(self, embed_dim, num_heads, pair_dim):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        
        self.bias_projection = nn.Linear(pair_dim, num_heads, bias=False)

    def forward(self, seq_repr, pair_repr):
        # seq_repr: [B, L, embed_dim]
        # pair_repr: [B, L, L, pair_dim]
        B, L, _ = seq_repr.shape
        
        pair_bias = self.bias_projection(pair_repr) # -> [B, L, L, num_heads]
        pair_bias = pair_bias.permute(0, 3, 1, 2)   # -> [B, num_heads, L, L]

        q, k, v = self.qkv_projection(seq_repr).chunk(3, dim=-1)
        
        q = q.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, L, D_h]
        k = k.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, L, D_h]
        v = v.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # [B, H, L, D_h]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_scores = attn_scores + pair_bias

        attn_weights = F.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_weights, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        
        return self.out_projection(context)

class CoEvolutionBlock(nn.Module):
    """
    Co-evolution block that combines 1D and 2D representations.
    """
    def __init__(self, seq_dim=64, pair_dim=32, num_heads=4, dropout=0.1):
        super().__init__()

        # --- 1D (Transformer-like) ---
        self.attn_norm = nn.LayerNorm(seq_dim)
        self.pair_bias_attention = PairwiseBiasAttention(seq_dim, num_heads, pair_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # --- 2D (CNN-like) ---
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_processor = MBConvBlock(
            in_channels=pair_dim, out_channels=pair_dim, 
            kernel_size=3, stride=1, expand_ratio=2
        )

        # --- outer product mean ---
        self.outer_product_mean = OuterProductMean(in_channels=seq_dim, out_channels=pair_dim)

    def forward(self, seq_repr, pair_repr):
        # seq_repr: [B, L, seq_dim]
        # pair_repr: [B, L, L, pair_dim]
        
        # 1. 1D -> 2D
        op_update = self.outer_product_mean(self.attn_norm(seq_repr)) # -> [B, L, L, pair_dim]
        pair_repr = pair_repr + op_update
        
        pair_repr_norm = self.pair_norm(pair_repr)
        pair_repr_norm = pair_repr_norm.permute(0, 3, 1, 2) # -> [B, pair_dim, L, L]
        pair_repr = pair_repr + self.pair_processor(pair_repr_norm).permute(0, 2, 3, 1)

        # 3. & 4. 2D -> 1D 
        attn_output = self.pair_bias_attention(self.attn_norm(seq_repr), self.pair_norm(pair_repr))
        seq_repr = seq_repr + self.attn_dropout(attn_output)
        
        return seq_repr, pair_repr


# =================================================================================
# Section 3: (MetaLearnerNet v3.0)
# =================================================================================

class MetaLearnerNet_v3(nn.Module):
    """
    MetaLearnerNet v3.0
    """
    def __init__(self, num_blocks=8, num_categories=5, seq_dim=64, pair_dim=32, predicts_channels=4, pair_feature_channels=17):
        super().__init__()

        # --- Stem ---
        self.seq_embedding = nn.Embedding(num_embeddings=num_categories, embedding_dim=seq_dim)
        self.rel_pos = RelativePositionEncoding(bin_num=32*2+1, c_z=pair_dim)
        
        total_pair_channels = pair_feature_channels + predicts_channels
        self.pair_stem = nn.Conv2d(total_pair_channels, pair_dim, kernel_size=1)
        
        # --- Co-evolution blocks ---
        self.co_evolution_blocks = nn.ModuleList([
            CoEvolutionBlock(seq_dim, pair_dim) for _ in range(num_blocks)
        ])
        
        # --- Output head ---
        self.output_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 1)
        )

    def forward(self, predicts, seq_encoded, pair_feature):
        # seq_encoded: [B, L]
        # pair_feature: [B, C1, L, L]
        # predicts: [B, C2, L, L]

        # 1. input stem
        seq_repr = self.seq_embedding(seq_encoded) # -> [B, L, seq_dim]
        
        initial_pair_repr = torch.cat([pair_feature, predicts], dim=1)
        pair_repr = self.pair_stem(initial_pair_repr) # -> [B, pair_dim, L, L]
        pair_repr = pair_repr.permute(0, 2, 3, 1)    # -> [B, L, L, pair_dim]

        rel_bias = self.rel_pos(seq_repr.size(1), seq_repr.device)  # [B, L, L, pair_dim]
        pair_repr = pair_repr + rel_bias

        # 2. Co-evolution blocks
        for block in self.co_evolution_blocks:
            seq_repr, pair_repr = block(seq_repr, pair_repr)
            
        # 3. output head
        output = self.output_head(pair_repr).squeeze(-1) # -> [B, L, L]
        
        output = output.unsqueeze(1)

        # symmetric output
        output = (output + output.transpose(-2, -1)) / 2
        return output

# =================================================================================
# Section 4: test case
# =================================================================================
if __name__ == '__main__':
    # --- 1. hyperparameters ---
    seq_len = 64
    batch_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model parameters
    NUM_BLOCKS = 4
    SEQ_DIM = 128
    PAIR_DIM = 64
    PREDICTS_CHANNELS = 4
    PAIR_FEATURE_CHANNELS = 17
    NUM_CATEGORIES = 4 # (A, U, C, G)

    print(f"device: {device}")
    
    # --- 2. 实例化新模型 ---
    model = MetaLearnerNet_v3(
        num_blocks=NUM_BLOCKS,
        num_categories=NUM_CATEGORIES,
        seq_dim=SEQ_DIM,
        pair_dim=PAIR_DIM,
        predicts_channels=PREDICTS_CHANNELS,
        pair_feature_channels=PAIR_FEATURE_CHANNELS
    ).to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"MetaLearnerNet_v3 ")
    print(f"The model has {num_params:,} trainable parameters.")

    # --- 3. create dummy input data ---
    dummy_seq_encoded = torch.randint(0, NUM_CATEGORIES, (batch_size, seq_len), dtype=torch.long).to(device)
    dummy_pair_feature = torch.randn(batch_size, PAIR_FEATURE_CHANNELS, seq_len, seq_len).to(device)
    dummy_predicts = torch.randn(batch_size, PREDICTS_CHANNELS, seq_len, seq_len).to(device)

    print("\ndummy input data shapes:")
    print(f"  - seq_encoded:    {dummy_seq_encoded.shape}")
    print(f"  - pair_feature:   {dummy_pair_feature.shape}")
    print(f"  - predicts:       {dummy_predicts.shape}")

    # --- 4. run the model ---
    with torch.no_grad():
        output = model(dummy_seq_encoded, dummy_pair_feature, dummy_predicts)

    print(f"\nmodel output shape: {output.shape}")
    expected_shape = (batch_size, 1, seq_len, seq_len)
    if output.shape == expected_shape:
        print(f"\n✅ model ran successfully, input-output dimension match! (expected: {expected_shape})")
    else:
        print(f"\n❌ [Warning] model output dimension is incorrect! (expected: {expected_shape})")