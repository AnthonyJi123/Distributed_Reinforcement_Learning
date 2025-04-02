import os
import yaml
import torch
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.air.config import RunConfig
from ray.air.config import CheckpointConfig
from typing import Dict, Any, Optional, List, Tuple
import time
import logging
from functools import partial
import numpy as np

from src.model.transformer import GPT2Model, create_model_from_config
from src.model.reward_model import RewardModel, create_reward_model_from_base
from src.trainer.ppo_trainer import PPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RayRLHFTrainer:
    """
    Distributed RLHF trainer using Ray.
    Handles the distributed training of language models using RLHF.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the distributed trainer.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
        
        # Initialize Ray if not already done
        if not ray.is_initialized():
            ray_config = self.config.get("cluster", {})
            head_address = ray_config.get("head_address", None)
            
            ray.init(
                address=head_address if head_address != "localhost:6379" else None,
                runtime_env={"pip": ["torch", "transformers", "numpy", "pydantic", "tqdm"]}
            )
            logger.info(f"Ray initialized with {ray.cluster_resources()} resources")
            
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
        
    def _setup_checkpoint_config(self) -> CheckpointConfig:
        """Set up checkpoint configuration for Ray training."""
        fault_tolerance = self.config.get("fault_tolerance", {})
        return CheckpointConfig(
            num_to_keep=5,
            checkpoint_score_attribute="reward"
        )
        
    def _get_scaling_config(self) -> ScalingConfig:
        """Get scaling configuration for Ray training."""
        cluster_config = self.config.get("cluster", {})
        
        return ScalingConfig(
            num_workers=cluster_config.get("num_workers", 4),
            use_gpu=True,
            resources_per_worker={
                "CPU": cluster_config.get("resources_per_worker", {}).get("CPU", 1),
                "GPU": cluster_config.get("resources_per_worker", {}).get("GPU", 1)
            }
        )
        
    def initialize_models(self) -> Tuple[GPT2Model, RewardModel]:
        """
        Initialize the policy and reward models.
        
        Returns:
            Tuple of (policy_model, reward_model)
        """
        model_config = self.config.get("model", {})
        
        # Create base model
        policy_model = create_model_from_config({"model": model_config})
        
        # Create reference model (for KL divergence)
        reference_model = create_model_from_config({"model": model_config})
        
        # Copy parameters from policy model to reference model
        reference_model.load_state_dict(policy_model.state_dict())
        
        # Create reward model
        reward_model = create_reward_model_from_base(
            base_model=create_model_from_config({"model": model_config}),
            config=model_config
        )
        
        return policy_model, reference_model, reward_model
        
    def _train_step(self, policy_model, reference_model, reward_model, inputs, config):
        """
        Single training step for RLHF.
        This function runs on each worker.
        
        Args:
            policy_model: The policy model to train
            reference_model: Reference model for KL divergence
            reward_model: Reward model
            inputs: Input data
            config: Training configuration
            
        Returns:
            Dictionary of metrics
        """
        # Setup PPO trainer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ppo_config = config.get("rlhf", {}).get("ppo", {})
        
        ppo_trainer = PPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            reward_model=reward_model,
            config=ppo_config,
            device=device
        )
        
        # Extract inputs
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Generate batch with policy
        generated_ids, gen_attention_mask, log_probs, values, rewards = ppo_trainer.generate_batch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=config.get("model", {}).get("max_seq_len", 1024),
            temperature=0.8,
            top_k=50,
            top_p=0.95
        )
        
        # Train on generated batch
        metrics = ppo_trainer.train_on_batch(
            input_ids=generated_ids,
            attention_mask=gen_attention_mask,
            log_probs=log_probs,
            values=values,
            rewards=rewards
        )
        
        # Add the average reward to metrics
        metrics["reward"] = rewards.mean().item()
        
        # Report metrics
        train.report(metrics)
        
        return metrics
        
    def train_supervised(self, dataset, model_save_path: str, epochs: int = 3):
        """
        Supervised fine-tuning (SFT) phase of RLHF.
        Trains the policy model on demonstration data in a simpler way without Ray datasets.
        
        Args:
            dataset: PyTorch dataset with demonstration data
            model_save_path: Path to save the trained model
            epochs: Number of training epochs
        """
        # Create model on local process
        logger.info("Creating model for SFT...")
        model = create_model_from_config({"model": self.config.get("model", {})})
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        
        # Create optimizer
        sft_config = self.config.get("rlhf", {}).get("sft", {})
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=sft_config.get("learning_rate", 1e-5),
            weight_decay=0.01
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=sft_config.get("batch_size", 16),
            shuffle=True
        )
        
        logger.info(f"Starting SFT training for {epochs} epochs...")
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device) if "labels" in batch else input_ids
                
                # Forward pass
                logits = model(input_ids, attention_mask)
                
                # Compute loss
                loss_fct = torch.nn.CrossEntropyLoss()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Log epoch results
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save the model
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_save_path, "model.pt"))
        logger.info(f"SFT model saved to {model_save_path}")
        
        return model
    
    def train_reward_model(self, preference_dataset, model_save_path: str, epochs: int = 3):
        """
        Train the reward model on preference data using a simpler approach without Ray datasets.
        
        Args:
            preference_dataset: Dataset of preference pairs
            model_save_path: Path to save the reward model
            epochs: Number of training epochs
        """
        # Create base model
        logger.info("Creating reward model...")
        model_config = self.config.get("model", {})
        base_model = create_model_from_config({"model": model_config})
        
        # Create reward model
        reward_model = create_reward_model_from_base(
            base_model=base_model,
            config=model_config
        )
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        reward_model.to(device)
        
        # Create optimizer
        reward_config = self.config.get("rlhf", {}).get("reward_model", {})
        optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=reward_config.get("learning_rate", 5e-6),
            weight_decay=0.01
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            preference_dataset,
            batch_size=reward_config.get("batch_size", 16),
            shuffle=True
        )
        
        logger.info(f"Starting reward model training for {epochs} epochs...")
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                chosen_ids = batch["chosen_ids"].to(device)
                rejected_ids = batch["rejected_ids"].to(device)
                chosen_mask = batch["chosen_mask"].to(device)
                rejected_mask = batch["rejected_mask"].to(device)
                
                # Get rewards for chosen and rejected sequences
                chosen_rewards = reward_model(chosen_ids, chosen_mask)
                rejected_rewards = reward_model(rejected_ids, rejected_mask)
                
                # Compute the preference loss (chosen should have higher reward than rejected)
                loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # Log epoch results
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        
        # Save the model
        os.makedirs(model_save_path, exist_ok=True)
        torch.save(reward_model.state_dict(), os.path.join(model_save_path, "model.pt"))
        logger.info(f"Reward model saved to {model_save_path}")
        
        return reward_model
    
    def train_rlhf(self, dataset, policy_model_path: str, reward_model_path: str, output_path: str, max_iterations: int = 10000):
        """
        Train using RLHF (PPO) with a simpler approach without Ray datasets.
        
        Args:
            dataset: Dataset with prompts
            policy_model_path: Path to the pre-trained policy model
            reward_model_path: Path to the pre-trained reward model
            output_path: Path to save the final model
            max_iterations: Maximum number of training iterations
        """
        logger.info("Initializing models for RLHF training...")
        
        # Initialize or load models
        policy_model, reference_model, reward_model = self.initialize_models()
        
        # Try to load pre-trained models if they exist
        if os.path.exists(os.path.join(policy_model_path, "model.pt")):
            logger.info(f"Loading pre-trained policy model from {policy_model_path}")
            policy_model.load_state_dict(torch.load(os.path.join(policy_model_path, "model.pt")))
            # Copy to reference model
            reference_model.load_state_dict(policy_model.state_dict())
            
        if os.path.exists(os.path.join(reward_model_path, "model.pt")):
            logger.info(f"Loading pre-trained reward model from {reward_model_path}")
            # Need to get the base model from the reward model
            reward_base_model = reward_model.base_model
            reward_base_model.load_state_dict(torch.load(os.path.join(reward_model_path, "model.pt")))
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        policy_model.to(device)
        reference_model.to(device)
        reward_model.to(device)
        
        # Create PPO trainer
        ppo_config = self.config.get("rlhf", {}).get("ppo", {})
        ppo_trainer = PPOTrainer(
            policy_model=policy_model,
            reference_model=reference_model,
            reward_model=reward_model,
            config=ppo_config,
            device=device
        )
        
        # Create data loader
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            dataset,
            batch_size=ppo_config.get("batch_size", 8),
            shuffle=True
        )
        
        logger.info(f"Starting RLHF training for max {max_iterations} iterations...")
        
        # Training loop
        iteration = 0
        best_reward = float('-inf')
        
        while iteration < max_iterations:
            for batch in train_loader:
                if iteration >= max_iterations:
                    break
                    
                # Get inputs
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                
                # Generate batch with policy
                generated_ids, gen_attention_mask, log_probs, values, rewards = ppo_trainer.generate_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.get("model", {}).get("max_seq_len", 1024),
                    temperature=0.8,
                    top_k=50,
                    top_p=0.95
                )
                
                # Train on generated batch
                metrics = ppo_trainer.train_on_batch(
                    input_ids=generated_ids,
                    attention_mask=gen_attention_mask,
                    log_probs=log_probs,
                    values=values,
                    rewards=rewards
                )
                
                # Add the average reward to metrics
                avg_reward = rewards.mean().item()
                metrics["reward"] = avg_reward
                
                # Log progress
                if iteration % 10 == 0:
                    logger.info(f"Iteration {iteration}, Reward: {avg_reward:.4f}, "
                                f"Policy Loss: {metrics['policy_loss']:.4f}, "
                                f"Value Loss: {metrics['value_loss']:.4f}, "
                                f"KL Div: {metrics['kl_div']:.4f}")
                
                # Save best model
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    # Save checkpoint
                    os.makedirs(output_path, exist_ok=True)
                    torch.save({
                        "policy_model": policy_model.state_dict(),
                        "value_head": ppo_trainer.value_head.state_dict(),
                        "iteration": iteration,
                        "reward": avg_reward
                    }, os.path.join(output_path, "model.pt"))
                    logger.info(f"New best model saved (reward: {avg_reward:.4f})")
                
                iteration += 1
                
                # Save periodic checkpoint
                if iteration % 100 == 0:
                    checkpoint_path = os.path.join(output_path, f"checkpoint_{iteration}")
                    os.makedirs(checkpoint_path, exist_ok=True)
                    torch.save({
                        "policy_model": policy_model.state_dict(),
                        "value_head": ppo_trainer.value_head.state_dict(),
                        "iteration": iteration,
                        "reward": avg_reward
                    }, os.path.join(checkpoint_path, "model.pt"))
                    logger.info(f"Checkpoint saved at iteration {iteration}")
        
        logger.info(f"RLHF training completed after {iteration} iterations. Best reward: {best_reward:.4f}")
        return policy_model 