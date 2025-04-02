#!/usr/bin/env python3
"""
Main training script for Distributed Reinforcement Learning for LLMs.
This script orchestrates the entire RLHF training pipeline:
1. Supervised Fine-Tuning (SFT)
2. Reward Model Training
3. Reinforcement Learning (PPO)
"""

import os
import argparse
import yaml
import logging
import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
import ray

from src.distributed.ray_trainer import RayRLHFTrainer
from src.utils.data_utils import TextDataset, PreferenceDataset, PromptDataset
from src.model.transformer import create_model_from_config
from src.model.reward_model import create_reward_model_from_base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distributed RLHF Training")
    
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory for models and logs")
    parser.add_argument("--stage", type=str, choices=["sft", "reward", "rlhf", "all"], default="all",
                        help="Training stage to run (sft, reward, rlhf, or all)")
    parser.add_argument("--sft_data", type=str, help="Path to supervised fine-tuning data")
    parser.add_argument("--preference_data", type=str, help="Path to preference data for reward model training")
    parser.add_argument("--prompt_data", type=str, help="Path to prompt data for RLHF training")
    parser.add_argument("--sft_epochs", type=int, default=3, help="Number of epochs for SFT")
    parser.add_argument("--reward_epochs", type=int, default=3, help="Number of epochs for reward model training")
    parser.add_argument("--rlhf_iterations", type=int, default=10000, help="Number of iterations for RLHF training")
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize distributed trainer
    trainer = RayRLHFTrainer(args.config)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model output paths
    sft_model_path = os.path.join(args.output_dir, "sft_model")
    reward_model_path = os.path.join(args.output_dir, "reward_model")
    rlhf_model_path = os.path.join(args.output_dir, "rlhf_model")
    
    # Initialize tokenizer
    model_type = config.get("model", {}).get("type", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Run the specified training stage(s)
    if args.stage in ["sft", "all"]:
        logger.info("Starting Supervised Fine-Tuning (SFT)")
        
        if args.sft_data is None:
            raise ValueError("Path to SFT data must be provided for SFT stage")
            
        # Create SFT dataset
        sft_dataset = TextDataset(
            data_path=args.sft_data,
            tokenizer=tokenizer,
            max_length=config.get("model", {}).get("max_seq_len", 1024)
        )
        
        # Run SFT training
        trainer.train_supervised(
            dataset=sft_dataset,
            model_save_path=sft_model_path,
            epochs=args.sft_epochs
        )
        
        logger.info(f"SFT completed. Model saved to {sft_model_path}")
    
    if args.stage in ["reward", "all"]:
        logger.info("Starting Reward Model Training")
        
        if args.preference_data is None:
            raise ValueError("Path to preference data must be provided for reward model training stage")
            
        # Create preference dataset
        preference_dataset = PreferenceDataset(
            data_path=args.preference_data,
            tokenizer=tokenizer,
            max_length=config.get("model", {}).get("max_seq_len", 1024)
        )
        
        # Run reward model training
        trainer.train_reward_model(
            preference_dataset=preference_dataset,
            model_save_path=reward_model_path,
            epochs=args.reward_epochs
        )
        
        logger.info(f"Reward model training completed. Model saved to {reward_model_path}")
    
    if args.stage in ["rlhf", "all"]:
        logger.info("Starting RLHF Training (PPO)")
        
        if args.prompt_data is None:
            raise ValueError("Path to prompt data must be provided for RLHF stage")
            
        # Create prompt dataset
        prompt_dataset = PromptDataset(
            data_path=args.prompt_data,
            tokenizer=tokenizer,
            max_length=config.get("model", {}).get("max_seq_len", 1024)
        )
        
        # Run RLHF training
        trainer.train_rlhf(
            dataset=prompt_dataset,
            policy_model_path=sft_model_path,
            reward_model_path=reward_model_path,
            output_path=rlhf_model_path,
            max_iterations=args.rlhf_iterations
        )
        
        logger.info(f"RLHF training completed. Model saved to {rlhf_model_path}")
    
    logger.info("All training stages completed successfully!")

if __name__ == "__main__":
    main() 