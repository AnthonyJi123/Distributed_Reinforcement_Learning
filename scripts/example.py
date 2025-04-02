#!/usr/bin/env python3
"""
Example script to showcase how to use the Distributed Reinforcement Learning system.
This script demonstrates:
1. Loading a configuration
2. Setting up a distributed training environment
3. Running a small-scale example training job
"""

import os
import argparse
import yaml
import logging
import torch
from transformers import AutoTokenizer
import random
import json
import ray

from src.distributed.ray_trainer import RayRLHFTrainer
from src.utils.data_utils import TextDataset, PreferenceDataset, PromptDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run an example training job")
    
    parser.add_argument("--config", type=str, default="configs/model_config.yaml", help="Path to model configuration file")
    parser.add_argument("--dist_config", type=str, default="configs/distributed.yaml", help="Path to distributed configuration file")
    parser.add_argument("--output_dir", type=str, default="./example_outputs", help="Output directory for models and logs")
    parser.add_argument("--create_dummy_data", action="store_true", help="Create dummy data for the example")
    parser.add_argument("--dummy_data_size", type=int, default=100, help="Number of examples to generate for dummy data")
    
    return parser.parse_args()

def create_dummy_data(output_dir: str, num_examples: int = 100):
    """Create dummy data for the example."""
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Example prompts for generation
    example_prompts = [
        "Write a story about a robot who wants to be human.",
        "Explain how to cook a perfect omelet.",
        "What are the key benefits of renewable energy?",
        "Describe the process of photosynthesis.",
        "Write a poem about the changing seasons.",
        "Compare and contrast democracy and authoritarianism.",
        "How would you solve climate change?",
        "Explain the theory of relativity in simple terms.",
        "What are the most important leadership qualities?",
        "Describe the perfect vacation destination."
    ]
    
    # Generate SFT data
    sft_data = []
    for _ in range(num_examples):
        prompt = random.choice(example_prompts)
        completion = f"This is a simulated completion for the prompt: {prompt}. It represents high-quality text generation."
        sft_data.append({
            "prompt": prompt,
            "completion": completion
        })
        
    with open(os.path.join(data_dir, "sft_data.json"), 'w') as f:
        json.dump(sft_data, f, indent=2)
        
    # Generate preference data
    preference_data = []
    for _ in range(num_examples):
        prompt = random.choice(example_prompts)
        chosen = f"This is a well-crafted response that addresses the prompt: {prompt}."
        rejected = f"This is a lower quality response to: {prompt}."
        preference_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        
    with open(os.path.join(data_dir, "preference_data.json"), 'w') as f:
        json.dump(preference_data, f, indent=2)
        
    # Generate prompt data
    prompt_data = []
    for _ in range(num_examples):
        prompt = random.choice(example_prompts)
        prompt_data.append({
            "prompt": prompt
        })
        
    with open(os.path.join(data_dir, "prompt_data.json"), 'w') as f:
        json.dump(prompt_data, f, indent=2)
        
    logger.info(f"Dummy data created in {data_dir}")
    
    return {
        "sft_data": os.path.join(data_dir, "sft_data.json"),
        "preference_data": os.path.join(data_dir, "preference_data.json"),
        "prompt_data": os.path.join(data_dir, "prompt_data.json")
    }

def run_example_training(config_path: str, dist_config_path: str, data_paths: dict, output_dir: str):
    """Run an example training job."""
    # Create merged config file
    with open(config_path, 'r') as f:
        model_config = yaml.safe_load(f)
        
    with open(dist_config_path, 'r') as f:
        dist_config = yaml.safe_load(f)
        
    merged_config = {**model_config, **dist_config}
    
    merged_config_path = os.path.join(output_dir, "merged_config.yaml")
    with open(merged_config_path, 'w') as f:
        yaml.dump(merged_config, f)
        
    # Initialize distributed trainer
    trainer = RayRLHFTrainer(merged_config_path)
    
    # Initialize tokenizer
    model_type = model_config.get("model", {}).get("type", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create model output paths
    sft_model_path = os.path.join(output_dir, "sft_model")
    reward_model_path = os.path.join(output_dir, "reward_model")
    rlhf_model_path = os.path.join(output_dir, "rlhf_model")
    
    # Create datasets
    sft_dataset = TextDataset(
        data_path=data_paths["sft_data"],
        tokenizer=tokenizer,
        max_length=model_config.get("model", {}).get("max_seq_len", 1024)
    )
    
    preference_dataset = PreferenceDataset(
        data_path=data_paths["preference_data"],
        tokenizer=tokenizer,
        max_length=model_config.get("model", {}).get("max_seq_len", 1024)
    )
    
    prompt_dataset = PromptDataset(
        data_path=data_paths["prompt_data"],
        tokenizer=tokenizer,
        max_length=model_config.get("model", {}).get("max_seq_len", 1024)
    )
    
    # Run SFT training
    logger.info("Starting Supervised Fine-Tuning (SFT)")
    trainer.train_supervised(
        dataset=sft_dataset,
        model_save_path=sft_model_path,
        epochs=1  # Just one epoch for the example
    )
    
    # Run reward model training
    logger.info("Starting Reward Model Training")
    trainer.train_reward_model(
        preference_dataset=preference_dataset,
        model_save_path=reward_model_path,
        epochs=1  # Just one epoch for the example
    )
    
    # Run RLHF training
    logger.info("Starting RLHF Training (PPO)")
    trainer.train_rlhf(
        dataset=prompt_dataset,
        policy_model_path=sft_model_path,
        reward_model_path=reward_model_path,
        output_path=rlhf_model_path,
        max_iterations=10  # Just a few iterations for the example
    )
    
    logger.info(f"Example training completed. Models saved in {output_dir}")

def main():
    """Main function for the example script."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dummy data if requested
    if args.create_dummy_data:
        data_paths = create_dummy_data(args.output_dir, args.dummy_data_size)
    else:
        data_dir = os.path.join(args.output_dir, "data")
        data_paths = {
            "sft_data": os.path.join(data_dir, "sft_data.json"),
            "preference_data": os.path.join(data_dir, "preference_data.json"),
            "prompt_data": os.path.join(data_dir, "prompt_data.json")
        }
        
        if not all(os.path.exists(path) for path in data_paths.values()):
            logger.warning("Dummy data not found. Creating it now.")
            data_paths = create_dummy_data(args.output_dir, args.dummy_data_size)
    
    # Run the example training
    run_example_training(args.config, args.dist_config, data_paths, args.output_dir)

if __name__ == "__main__":
    main() 