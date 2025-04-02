import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from tqdm import tqdm

from src.model.transformer import GPT2Model
from src.model.reward_model import RewardModel

class PPOTrainer:
    """
    Trainer for PPO (Proximal Policy Optimization) reinforcement learning with language models.
    Used in RLHF to fine-tune the policy model with a reward model.
    """
    
    def __init__(
        self,
        policy_model: GPT2Model,
        reference_model: GPT2Model,
        reward_model: RewardModel,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the PPO trainer.
        
        Args:
            policy_model: The policy model to train (will be updated)
            reference_model: A frozen copy of the initial policy model (for KL penalty)
            reward_model: The reward model to use for computing rewards
            config: Configuration dictionary with PPO parameters
            device: Device to use for training
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.reward_model = reward_model
        self.config = config
        self.device = device
        
        # Move models to device
        self.policy_model.to(device)
        self.reference_model.to(device)
        self.reward_model.to(device)
        
        # Freeze the reference model
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        # Freeze the reward model
        for param in self.reward_model.parameters():
            param.requires_grad = False
            
        # Create value head for policy model
        self.value_head = nn.Sequential(
            nn.Linear(policy_model.dim, policy_model.dim),
            nn.ReLU(),
            nn.Linear(policy_model.dim, 1)
        ).to(device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            list(self.policy_model.parameters()) + list(self.value_head.parameters()),
            lr=config.get("learning_rate", 1e-6)
        )
        
        # PPO hyperparameters
        self.ppo_epochs = config.get("ppo_epochs", 4)
        self.clip_range = config.get("clip_range", 0.2)
        self.value_loss_coef = config.get("value_loss_coef", 0.1)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.kl_coef = config.get("kl_coef", 0.1)
        
    def generate_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.LongTensor, List[float]]:
        """
        Generate sequences from the policy model and compute rewards.
        
        Args:
            input_ids: Input prompt token ids
            attention_mask: Attention mask for the prompt
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            
        Returns:
            Tuple of (input_ids, attention_mask, generated_ids, rewards)
        """
        batch_size, seq_len = input_ids.shape
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Store log probs and values for the generated tokens
        all_log_probs = []
        all_values = []
        
        # Clone input_ids for generation
        curr_ids = input_ids.clone()
        
        # Generate tokens auto-regressively
        self.policy_model.eval()
        
        with torch.no_grad():
            for i in range(max_length - seq_len):
                # Get logits from the policy model
                logits = self.policy_model(curr_ids, attention_mask)
                
                # Get the logits for the next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a mask based on sorted indices
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Compute log prob of the selected token
                log_prob = F.log_softmax(next_token_logits, dim=-1).gather(1, next_token).squeeze(1)
                all_log_probs.append(log_prob)
                
                # Compute value of the current state
                hidden_states = self.policy_model(curr_ids, attention_mask, return_logits=False)
                value = self.value_head(hidden_states[:, -1, :]).squeeze(-1)
                all_values.append(value)
                
                # Update attention mask and append new token
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=1)
                curr_ids = torch.cat([curr_ids, next_token], dim=1)
        
        # Compute rewards using the reward model
        with torch.no_grad():
            rewards = self.reward_model(curr_ids, attention_mask)
            
        # Convert log probs and values to tensors
        all_log_probs = torch.stack(all_log_probs, dim=1)
        all_values = torch.stack(all_values, dim=1)
        
        return curr_ids, attention_mask, all_log_probs, all_values, rewards
    
    def train_on_batch(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.FloatTensor,
        log_probs: torch.FloatTensor,
        values: torch.FloatTensor,
        rewards: torch.FloatTensor
    ) -> Dict[str, float]:
        """
        Train the policy model on a batch of rollouts.
        
        Args:
            input_ids: Full sequence token ids (prompt + generation)
            attention_mask: Attention mask for the full sequence
            log_probs: Log probs of generated tokens from the policy model
            values: Value predictions for each generated token
            rewards: Rewards for the generated sequences
            
        Returns:
            Dictionary of training metrics
        """
        batch_size, seq_len = input_ids.shape
        prompt_len = seq_len - log_probs.shape[1]
        
        # Move inputs to device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        log_probs = log_probs.to(self.device)
        values = values.to(self.device)
        rewards = rewards.to(self.device)
        
        # Use rewards as advantages for now (simple case)
        advantages = rewards.unsqueeze(1).expand_as(log_probs) - values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Train for multiple PPO epochs
        self.policy_model.train()
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl_div": 0.0,
            "total_loss": 0.0
        }
        
        for _ in range(self.ppo_epochs):
            # Compute logits for the entire sequence
            logits = self.policy_model(input_ids, attention_mask)
            
            # Extract logits for the generated part (excluding prompt)
            gen_logits = logits[:, prompt_len-1:-1, :]
            
            # Compute log probs of the generated tokens
            gen_log_probs = F.log_softmax(gen_logits, dim=-1)
            
            # Get the next token ids for the generated part
            next_tokens = input_ids[:, prompt_len:].unsqueeze(-1)
            
            # Get the log probs of the actual next tokens
            curr_log_probs = torch.gather(gen_log_probs, 2, next_tokens).squeeze(-1)
            
            # Compute the ratio between new and old policy
            ratio = torch.exp(curr_log_probs - log_probs)
            
            # Compute the PPO (clipped) policy loss
            policy_loss1 = -advantages * ratio
            policy_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            policy_loss = torch.max(policy_loss1, policy_loss2).mean()
            
            # Compute entropy bonus
            entropy = -torch.sum(torch.exp(gen_log_probs) * gen_log_probs, dim=-1).mean()
            
            # Compute KL divergence between new and reference policy
            with torch.no_grad():
                ref_logits = self.reference_model(input_ids, attention_mask)
                ref_gen_logits = ref_logits[:, prompt_len-1:-1, :]
                ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
                
            kl_div = F.kl_div(
                F.log_softmax(gen_logits, dim=-1),
                F.softmax(ref_gen_logits, dim=-1),
                reduction='batchmean'
            )
            
            # Compute value loss
            hidden_states = self.policy_model(input_ids, attention_mask, return_logits=False)
            gen_hidden_states = hidden_states[:, prompt_len-1:-1, :]
            curr_values = self.value_head(gen_hidden_states).squeeze(-1)
            value_loss = F.mse_loss(curr_values, values + advantages.detach())
            
            # Compute the total loss
            total_loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy + self.kl_coef * kl_div
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Update metrics
            metrics["policy_loss"] += policy_loss.item() / self.ppo_epochs
            metrics["value_loss"] += value_loss.item() / self.ppo_epochs
            metrics["entropy"] += entropy.item() / self.ppo_epochs
            metrics["kl_div"] += kl_div.item() / self.ppo_epochs
            metrics["total_loss"] += total_loss.item() / self.ppo_epochs
            
        return metrics 