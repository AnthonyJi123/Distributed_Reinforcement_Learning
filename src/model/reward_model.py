import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from src.model.transformer import GPT2Model

class RewardModel(nn.Module):
    """
    Reward model that estimates the quality of a text sequence.
    Used in RLHF to provide a reward signal for the PPO training.
    """
    def __init__(
        self,
        base_model: GPT2Model,
        dropout: float = 0.1
    ):
        super().__init__()
        self.base_model = base_model
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Value head on top of the transformer
        self.value_head = nn.Sequential(
            nn.Linear(base_model.dim, base_model.dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_model.dim, 1)
        )
        
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None
    ) -> torch.FloatTensor:
        """
        Forward pass of the reward model.
        
        Args:
            input_ids: Tensor of token ids
            attention_mask: Attention mask
            position_ids: Position ids
            
        Returns:
            Scalar reward values for each sequence in the batch
        """
        # Get hidden states from the base model (don't return logits)
        hidden_states = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_logits=False
        )
        
        # Apply value head to the last hidden state of the sequence
        rewards = self.value_head(hidden_states[:, -1, :])
        return rewards.squeeze(-1)
        
def create_reward_model_from_base(
    base_model: GPT2Model,
    config: Dict[str, Any] = None
) -> RewardModel:
    """
    Creates a reward model from a base GPT2 model.
    
    Args:
        base_model: Base GPT2 model to use
        config: Configuration dictionary
        
    Returns:
        Reward model
    """
    reward_model = RewardModel(
        base_model=base_model,
        dropout=config.get("dropout", 0.1) if config else 0.1
    )
    
    return reward_model
    
def train_reward_model(
    reward_model: RewardModel,
    preference_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    Trains a reward model on preference data.
    
    Args:
        reward_model: The reward model to train
        preference_data: Tuple of (chosen_ids, rejected_ids, attention_masks)
        optimizer: Optimizer to use for training
        device: Device to train on
        
    Returns:
        Loss value
    """
    reward_model.train()
    
    chosen_ids, rejected_ids, attention_masks = preference_data
    
    # Move data to device
    chosen_ids = chosen_ids.to(device)
    rejected_ids = rejected_ids.to(device)
    attention_masks = attention_masks.to(device)
    
    # Get rewards for chosen and rejected sequences
    chosen_rewards = reward_model(chosen_ids, attention_masks)
    rejected_rewards = reward_model(rejected_ids, attention_masks)
    
    # Compute the preference loss (chosen should have higher reward than rejected)
    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item() 