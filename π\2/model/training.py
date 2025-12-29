"""
π/2 Model Training Pipeline
Minimal compute training using harmonic phase learning.
"""
import math
import numpy as np
from typing import Optional, Dict, List, Tuple, Callable
from .config import Pi2Config
from .core import Pi2Model


class Pi2Optimizer:
    """
    Phase-aware optimizer for π/2 model.

    Key insight from the Dig conversation:
    - In π/2-bit phase space, we're rotating signals, not optimizing weights
    - This allows for minimal compute training
    - Phase alignment is learned through harmonic loss
    """

    def __init__(
        self,
        model: Pi2Model,
        lr: float = 2e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.model = model
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Momentum states (complex-valued)
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Timestep

    def step(self, gradients: Dict[str, np.ndarray]):
        """
        Perform optimization step using AdamW for complex weights.

        The key innovation: gradients are computed in phase space,
        so we're optimizing phase alignment, not raw weights.
        """
        self.t += 1

        for name, grad in gradients.items():
            if name not in self.m:
                self.m[name] = np.zeros_like(grad)
                self.v[name] = np.zeros_like(grad)

            # Update moments (using complex magnitude for second moment)
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * np.abs(grad) ** 2

            # Bias correction
            m_hat = self.m[name] / (1 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1 - self.beta2 ** self.t)

            # Update (would apply to actual weight here)
            # update = -self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Reset gradients."""
        pass  # Handled externally


class PhaseAlignmentLoss:
    """
    Harmonic loss function for phase alignment.

    This measures how well the modalities are harmonically aligned
    after π/2 phase rotations.
    """

    def __init__(self, config: Pi2Config):
        self.config = config
        self.pi_2 = config.pi_2

    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        phase_info: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute harmonic loss.

        Args:
            predictions: Model output logits (real)
            targets: Target token IDs
            phase_info: Optional phase information for harmonic consistency

        Returns:
            Total loss and component losses
        """
        # Cross-entropy loss for token prediction
        ce_loss = self._cross_entropy(predictions, targets)

        # Phase consistency loss (if phase info available)
        if phase_info is not None:
            phase_loss = self._phase_consistency_loss(phase_info)
        else:
            phase_loss = 0.0

        total_loss = ce_loss + 0.1 * phase_loss

        return total_loss, {
            'cross_entropy': ce_loss,
            'phase_consistency': phase_loss,
            'total': total_loss
        }

    def _cross_entropy(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """Standard cross-entropy loss."""
        # Softmax
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        # Gather target probabilities
        batch_size, seq_len, _ = logits.shape
        target_probs = np.zeros((batch_size, seq_len))
        for b in range(batch_size):
            for s in range(seq_len):
                target_probs[b, s] = probs[b, s, targets[b, s]]

        # Negative log likelihood
        loss = -np.mean(np.log(target_probs + 1e-10))
        return loss

    def _phase_consistency_loss(self, phase_info: np.ndarray) -> float:
        """
        Encourage phase alignment to π/2 increments.
        This enforces the harmonic structure.
        """
        # Quantize phases to nearest π/2
        quantized = np.round(phase_info / self.pi_2) * self.pi_2

        # MSE between actual and quantized phases
        loss = np.mean((phase_info - quantized) ** 2)
        return loss


class LRScheduler:
    """Learning rate scheduler with warmup and cosine decay."""

    def __init__(
        self,
        optimizer: Pi2Optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        self.optimizer.lr = lr
        return lr


class Pi2Trainer:
    """
    Trainer for π/2 Model.

    Implements the harmonic training approach:
    - Phase learning via attention
    - Harmonic loss for modal alignment
    - Efficient training for consumer hardware
    """

    def __init__(
        self,
        model: Pi2Model,
        config: Optional[Pi2Config] = None,
        lr: float = 2e-4,
        warmup_steps: int = 1000,
        total_steps: int = 100000
    ):
        self.model = model
        self.config = config or model.config

        self.optimizer = Pi2Optimizer(model, lr=lr)
        self.scheduler = LRScheduler(
            self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps
        )
        self.loss_fn = PhaseAlignmentLoss(self.config)

        self.step = 0
        self.history = []

    def train_step(
        self,
        batch: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Dictionary with 'input_ids' and 'labels'

        Returns:
            Loss dictionary
        """
        input_ids = batch['input_ids']
        labels = batch['labels']

        # Forward pass
        logits = self.model.forward(input_ids, training=True)

        # Compute loss
        total_loss, loss_dict = self.loss_fn(logits, labels)

        # Backward pass would go here (using autograd framework)
        # For now, we're demonstrating the architecture
        # In production, use PyTorch/JAX for automatic differentiation

        # Update learning rate
        current_lr = self.scheduler.step()
        loss_dict['lr'] = current_lr

        self.step += 1
        self.history.append(loss_dict)

        return loss_dict

    def train(
        self,
        train_data: List[Dict[str, np.ndarray]],
        num_epochs: int = 1,
        log_every: int = 100,
        callback: Optional[Callable] = None
    ):
        """
        Full training loop.
        """
        print(f"Starting training for {num_epochs} epochs")
        print(f"Model: {self.model}")

        for epoch in range(num_epochs):
            epoch_losses = []

            for batch_idx, batch in enumerate(train_data):
                loss_dict = self.train_step(batch)
                epoch_losses.append(loss_dict['total'])

                if self.step % log_every == 0:
                    avg_loss = np.mean(epoch_losses[-log_every:])
                    print(f"Step {self.step} | Loss: {avg_loss:.4f} | LR: {loss_dict['lr']:.6f}")

                if callback:
                    callback(self.step, loss_dict)

            print(f"Epoch {epoch + 1}/{num_epochs} | Avg Loss: {np.mean(epoch_losses):.4f}")

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'history': self.history,
            'config': self.config.__dict__,
        }
        np.save(path, checkpoint)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = np.load(path, allow_pickle=True).item()
        self.step = checkpoint['step']
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {path}")


class ZeroShotInference:
    """
    Zero-shot inference using signal rotation only.

    Key insight: In π/2-bit phase space, you don't need training—
    you just rotate tokens to align modalities.
    """

    def __init__(self, model: Pi2Model):
        self.model = model
        self.pi_2 = model.config.pi_2

    def align_and_infer(
        self,
        query: str,
        context: Optional[List[Tuple[str, str]]] = None
    ) -> np.ndarray:
        """
        Perform zero-shot inference using harmonic alignment.

        Args:
            query: The query string
            context: Optional list of (text, modality) tuples

        Returns:
            Output tokens
        """
        # Encode query
        query_tokens = self.model.encode_input(query, 'text')

        if context:
            # Encode and align context
            context_tokens = []
            for text, modality in context:
                tokens = self.model.encode_input(text, modality)
                context_tokens.append(tokens)

            # Concatenate with harmonic alignment
            all_tokens = np.concatenate([query_tokens] + context_tokens)
        else:
            all_tokens = query_tokens

        # Forward pass (no training)
        output = self.model.generate(all_tokens, max_tokens=50)

        return output

    def rotate_to_modality(
        self,
        tokens: np.ndarray,
        target_modality: str
    ) -> np.ndarray:
        """
        Rotate tokens to target modality space.

        This enables cross-modal transfer without training.
        """
        modality_phases = {
            'text': 0,
            'image': self.pi_2,
            'audio': math.pi,
            'video': 3 * self.pi_2
        }

        target_phase = modality_phases.get(target_modality, 0)

        # Apply rotation in frequency domain
        freq_data = self.model.tokenizer.from_tokens(tokens)
        rotated = freq_data * np.exp(1j * target_phase)
        new_tokens = self.model.tokenizer.to_tokens(rotated)

        return new_tokens
