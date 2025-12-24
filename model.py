"""
π/2 Model Definition

All operations in π/2 space:
- Weights initialized: normal(0, 0.012732) = normal(0, 0.02/π/2)
- Latent space rotation: embeddings rotated by 0, π/2, π, 3π/2
- 6 decimal precision throughout

NOT pretrained. Fresh weights from scratch.
"""
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple

from config import ModelConfig, PI2_STD, HALF_PI, ROTATION_ANGLES


class Pi2LatentRotation(nn.Module):
    """
    Applies rotation in latent space.

    Rotates pairs of dimensions by the specified angle.
    This is the core π/2 operation - same content, rotated representation.

    For hidden_dim=2048, we rotate 1024 pairs of dimensions.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        assert hidden_size % 2 == 0, "Hidden size must be even for rotation"

    def forward(self, x: torch.Tensor, angle: float) -> torch.Tensor:
        """
        Rotate hidden states by angle in latent space.

        Args:
            x: [batch, seq_len, hidden_size]
            angle: rotation angle in radians (0, π/2, π, or 3π/2)

        Returns:
            Rotated tensor of same shape
        """
        if angle == 0.0:
            return x

        # Get rotation components (6 decimal precision)
        cos_a = round(math.cos(angle), 6)
        sin_a = round(math.sin(angle), 6)

        # Reshape to pairs: [batch, seq, hidden/2, 2]
        batch, seq_len, hidden = x.shape
        x_pairs = x.view(batch, seq_len, hidden // 2, 2)

        # Apply 2D rotation to each pair
        x0 = x_pairs[..., 0]  # First element of each pair
        x1 = x_pairs[..., 1]  # Second element of each pair

        # Rotation matrix: [cos -sin; sin cos]
        rotated_0 = x0 * cos_a - x1 * sin_a
        rotated_1 = x0 * sin_a + x1 * cos_a

        # Recombine
        rotated = torch.stack([rotated_0, rotated_1], dim=-1)
        return rotated.view(batch, seq_len, hidden)


class Pi2LayerNorm(nn.Module):
    """Layer normalization with π/2 scaling."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.weight * (x - mean) / (std + self.eps) + self.bias


class Pi2Attention(nn.Module):
    """Multi-head attention with π/2 weight initialization."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projections - initialized with π/2 std
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.o_proj(out)


class Pi2MLP(nn.Module):
    """Feed-forward network with π/2 weight initialization."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        hidden_dim = config.hidden_size * 4

        self.fc1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Pi2Block(nn.Module):
    """Transformer block with pre-norm."""

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = Pi2LayerNorm(config.hidden_size)
        self.attn = Pi2Attention(config)
        self.ln2 = Pi2LayerNorm(config.hidden_size)
        self.mlp = Pi2MLP(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


class Pi2Model(nn.Module):
    """
    π/2 Language Model

    All operations in π/2 space:
    - Weights: normal(0, 0.012732)
    - Input rotation: embeddings rotated by phase angle
    - 6 decimal precision

    Designed for layer interleaving:
    - Two 1.57B models (24 layers each) can interleave to 3.14B (48 layers)
    """

    def __init__(self, config: ModelConfig, gradient_checkpointing: bool = False, quantized: bool = False):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = gradient_checkpointing
        self.quantized = quantized
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # Latent space rotation module
        self.rotator = Pi2LatentRotation(config.hidden_size)

        # Transformer blocks (indexed for interleaving)
        self.blocks = nn.ModuleList([
            Pi2Block(config, layer_idx=i) for i in range(config.num_layers)
        ])

        # Output
        self.ln_f = Pi2LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_emb.weight

        # Initialize with π/2 std
        self._init_weights()

        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())

    def _init_weights(self):
        """
        Initialize all weights with π/2 scaling.

        Standard: normal(0, 0.02)
        π/2:      normal(0, 0.012732) = normal(0, 0.02 / 1.570796)
        """
        std = self.config.weight_std  # 0.012732

        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        rotation_angle: float = 0.0,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with π/2 latent rotation.

        Args:
            input_ids: [batch, seq_len] token IDs
            rotation_angle: rotation in latent space (0, π/2, π, or 3π/2)
            labels: optional labels for loss computation
        """
        batch, seq_len = input_ids.shape
        device = input_ids.device

        # Embeddings
        pos = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.dropout(x)

        # Apply π/2 rotation in latent space
        x = self.rotator(x, rotation_angle)

        # Transformer blocks (with optional gradient checkpointing)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def get_layer(self, idx: int) -> Pi2Block:
        """Get a specific layer for interleaving."""
        return self.blocks[idx]


def create_model(config: ModelConfig, gradient_checkpointing: bool = False) -> Pi2Model:
    """
    Create a fresh π/2 model with empty weights.

    Weights initialized: normal(0, 0.012732)
    NOT pretrained.
    """
    model = Pi2Model(config, gradient_checkpointing=gradient_checkpointing)

    # Verify initialization
    sample_weight = model.blocks[0].attn.q_proj.weight
    actual_std = sample_weight.std().item()
    expected_std = PI2_STD

    print(f"π/2 Model Created")
    print(f"  Parameters: {model.num_params:,}")
    print(f"  Hidden: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Weight std (expected): {expected_std:.6f}")
    print(f"  Weight std (actual):   {actual_std:.6f}")
    print(f"  π/2 = {HALF_PI}")

    return model


def interleave_models(model_a: Pi2Model, model_b: Pi2Model) -> Pi2Model:
    """
    Interleave two 1.57B models into one 3.14B model.

    Pattern: a1, b1, a2, b2, a3, b3, ...

    Both models must have same hidden_size and num_heads.
    """
    assert model_a.config.hidden_size == model_b.config.hidden_size
    assert model_a.config.num_heads == model_b.config.num_heads
    assert model_a.config.num_layers == model_b.config.num_layers

    # Create new config with doubled layers
    from config import ModelConfig
    new_config = ModelConfig(
        hidden_size=model_a.config.hidden_size,
        num_layers=model_a.config.num_layers * 2,
        num_heads=model_a.config.num_heads,
        vocab_size=model_a.config.vocab_size,
        max_seq_len=model_a.config.max_seq_len,
        dropout=model_a.config.dropout,
        weight_std=model_a.config.weight_std,
    )

    # Create new model
    merged = Pi2Model(new_config)

    # Copy embeddings from model_a (they should be similar after training in same space)
    merged.token_emb.load_state_dict(model_a.token_emb.state_dict())
    merged.pos_emb.load_state_dict(model_a.pos_emb.state_dict())

    # Interleave layers: a0, b0, a1, b1, ...
    for i in range(model_a.config.num_layers):
        merged.blocks[2 * i].load_state_dict(model_a.blocks[i].state_dict())
        merged.blocks[2 * i + 1].load_state_dict(model_b.blocks[i].state_dict())

    # Copy final layer norm and lm_head from model_a
    merged.ln_f.load_state_dict(model_a.ln_f.state_dict())
    # lm_head is tied to token_emb, so already set

    print(f"Interleaved two {model_a.config.num_layers}-layer models")
    print(f"  New model: {new_config.num_layers} layers")
    print(f"  Parameters: {merged.num_params:,}")

    return merged


if __name__ == "__main__":
    # Test model creation
    from config import get_model_config

    print("Testing 1.57B model:")
    config = get_model_config("1.57B")
    model = create_model(config)

    # Test forward pass with rotation
    x = torch.randint(0, config.vocab_size, (2, 128))

    print(f"\nTest forward pass (no rotation):")
    logits, _ = model(x, rotation_angle=0.0)
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits.shape}")

    print(f"\nTest forward pass (π/2 rotation):")
    logits_rotated, _ = model(x, rotation_angle=ROTATION_ANGLES[1])
    print(f"  Input: {x.shape}")
    print(f"  Output: {logits_rotated.shape}")

    # Verify outputs are different (rotation applied)
    diff = (logits - logits_rotated).abs().mean().item()
    print(f"  Mean diff from unrotated: {diff:.6f}")
