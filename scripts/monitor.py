#!/usr/bin/env python3
"""
Training monitoring script for visualizing training progress.
This script reads training logs and visualizes metrics for the different training stages:
1. SFT metrics (loss)
2. Reward model metrics (loss)
3. RLHF metrics (policy loss, value loss, rewards, KL divergence)
"""

import os
import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Dict, List, Any, Optional
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor training progress")
    
    parser.add_argument("--log_dir", type=str, required=True, help="Directory containing training logs")
    parser.add_argument("--output_dir", type=str, default="./monitoring", help="Directory to save visualizations")
    parser.add_argument("--plot_type", type=str, choices=["loss", "reward", "kl", "all"], default="all",
                         help="Type of plot to generate")
    
    return parser.parse_args()

def load_logs(log_dir: str) -> Dict[str, Any]:
    """Load training logs from directory."""
    log_files = {
        "sft": glob.glob(os.path.join(log_dir, "*supervised_finetuning*")),
        "reward": glob.glob(os.path.join(log_dir, "*reward_model*")),
        "rlhf": glob.glob(os.path.join(log_dir, "*rlhf_training*"))
    }
    
    logs = {}
    
    # Load SFT logs
    if log_files["sft"]:
        sft_logs = []
        for log_file in log_files["sft"]:
            if os.path.isfile(log_file) and log_file.endswith('.json'):
                with open(log_file, 'r') as f:
                    sft_logs.append(json.load(f))
        logs["sft"] = sft_logs
        
    # Load reward model logs
    if log_files["reward"]:
        reward_logs = []
        for log_file in log_files["reward"]:
            if os.path.isfile(log_file) and log_file.endswith('.json'):
                with open(log_file, 'r') as f:
                    reward_logs.append(json.load(f))
        logs["reward"] = reward_logs
        
    # Load RLHF logs
    if log_files["rlhf"]:
        rlhf_logs = []
        for log_file in log_files["rlhf"]:
            if os.path.isfile(log_file) and log_file.endswith('.json'):
                with open(log_file, 'r') as f:
                    rlhf_logs.append(json.load(f))
        logs["rlhf"] = rlhf_logs
        
    return logs

def plot_sft_loss(logs: List[Dict], output_dir: str):
    """Plot SFT loss over epochs."""
    if not logs:
        logger.warning("No SFT logs found")
        return
        
    # Extract loss values and epochs
    epochs = []
    losses = []
    
    for log in logs:
        if "metrics" in log:
            for entry in log["metrics"]:
                if "epoch" in entry and "loss" in entry:
                    epochs.append(entry["epoch"])
                    losses.append(entry["loss"])
    
    if not epochs:
        logger.warning("No epoch/loss data found in SFT logs")
        return
        
    # Sort by epoch
    sorted_data = sorted(zip(epochs, losses))
    epochs, losses = zip(*sorted_data)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'b-', label='Training Loss')
    
    plt.title('SFT Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Set x-axis to only use integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'sft_loss.png'))
    plt.close()
    
    logger.info(f"SFT loss plot saved to {os.path.join(output_dir, 'sft_loss.png')}")

def plot_reward_model_loss(logs: List[Dict], output_dir: str):
    """Plot reward model loss over epochs."""
    if not logs:
        logger.warning("No reward model logs found")
        return
        
    # Extract loss values and epochs
    epochs = []
    losses = []
    
    for log in logs:
        if "metrics" in log:
            for entry in log["metrics"]:
                if "epoch" in entry and "loss" in entry:
                    epochs.append(entry["epoch"])
                    losses.append(entry["loss"])
    
    if not epochs:
        logger.warning("No epoch/loss data found in reward model logs")
        return
        
    # Sort by epoch
    sorted_data = sorted(zip(epochs, losses))
    epochs, losses = zip(*sorted_data)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, 'r-', label='Training Loss')
    
    plt.title('Reward Model Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Set x-axis to only use integers
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'reward_model_loss.png'))
    plt.close()
    
    logger.info(f"Reward model loss plot saved to {os.path.join(output_dir, 'reward_model_loss.png')}")

def plot_rlhf_metrics(logs: List[Dict], output_dir: str):
    """Plot RLHF metrics over iterations."""
    if not logs:
        logger.warning("No RLHF logs found")
        return
        
    # Extract metrics and iterations
    iterations = []
    policy_losses = []
    value_losses = []
    rewards = []
    kl_divs = []
    entropies = []
    
    for log in logs:
        if "metrics" in log:
            for entry in log["metrics"]:
                if "iteration" in entry:
                    iterations.append(entry["iteration"])
                    policy_losses.append(entry.get("policy_loss", None))
                    value_losses.append(entry.get("value_loss", None))
                    rewards.append(entry.get("reward", None))
                    kl_divs.append(entry.get("kl_div", None))
                    entropies.append(entry.get("entropy", None))
    
    if not iterations:
        logger.warning("No iteration data found in RLHF logs")
        return
        
    # Create a DataFrame for easier handling
    df = pd.DataFrame({
        'iteration': iterations,
        'policy_loss': policy_losses,
        'value_loss': value_losses,
        'reward': rewards,
        'kl_div': kl_divs,
        'entropy': entropies
    })
    
    # Sort by iteration
    df = df.sort_values('iteration')
    
    # Plot policy and value losses
    plt.figure(figsize=(10, 6))
    if not df['policy_loss'].isna().all():
        plt.plot(df['iteration'], df['policy_loss'], 'b-', label='Policy Loss')
    if not df['value_loss'].isna().all():
        plt.plot(df['iteration'], df['value_loss'], 'r-', label='Value Loss')
    
    plt.title('RLHF Training Losses Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'rlhf_losses.png'))
    plt.close()
    
    # Plot rewards
    plt.figure(figsize=(10, 6))
    if not df['reward'].isna().all():
        plt.plot(df['iteration'], df['reward'], 'g-', label='Reward')
    
    plt.title('RLHF Rewards Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'rlhf_rewards.png'))
    plt.close()
    
    # Plot KL divergence and entropy
    plt.figure(figsize=(10, 6))
    if not df['kl_div'].isna().all():
        plt.plot(df['iteration'], df['kl_div'], 'm-', label='KL Divergence')
    if not df['entropy'].isna().all():
        plt.plot(df['iteration'], df['entropy'], 'c-', label='Entropy')
    
    plt.title('RLHF KL Divergence and Entropy Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'rlhf_kl_entropy.png'))
    plt.close()
    
    logger.info(f"RLHF metrics plots saved to {output_dir}")

def main():
    """Main monitoring function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load logs
    logs = load_logs(args.log_dir)
    
    # Generate plots based on plot_type
    if args.plot_type in ["loss", "all"] and "sft" in logs:
        plot_sft_loss(logs["sft"], args.output_dir)
        
    if args.plot_type in ["loss", "all"] and "reward" in logs:
        plot_reward_model_loss(logs["reward"], args.output_dir)
        
    if args.plot_type in ["loss", "reward", "kl", "all"] and "rlhf" in logs:
        plot_rlhf_metrics(logs["rlhf"], args.output_dir)
        
    logger.info("Monitoring completed!")

if __name__ == "__main__":
    main() 