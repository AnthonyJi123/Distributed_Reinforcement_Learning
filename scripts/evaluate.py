#!/usr/bin/env python3
"""
Evaluation script for the trained models.
This script can evaluate:
1. The language model (perplexity, generation quality)
2. The reward model (preference alignment)
3. The RLHF-trained model (compared to the base model)
"""

import os
import argparse
import yaml
import torch
import logging
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from src.model.transformer import GPT2Model, create_model_from_config
from src.model.reward_model import RewardModel, create_reward_model_from_base
from src.utils.data_utils import TextDataset, PreferenceDataset, PromptDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with trained models")
    parser.add_argument("--eval_type", type=str, choices=["perplexity", "generation", "preference", "comparison", "all"],
                         default="all", help="Type of evaluation to run")
    parser.add_argument("--test_data", type=str, help="Path to test data for perplexity evaluation")
    parser.add_argument("--prompt_data", type=str, help="Path to prompt data for generation evaluation")
    parser.add_argument("--preference_data", type=str, help="Path to preference data for preference evaluation")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    
    return parser.parse_args()

def load_model(model_path, config, model_type="policy"):
    """Load a model from checkpoint."""
    if model_type == "policy":
        model = create_model_from_config({"model": config.get("model", {})})
        # In a real implementation, we would load the weights from the checkpoint
        # model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
    elif model_type == "reward":
        base_model = create_model_from_config({"model": config.get("model", {})})
        model = create_reward_model_from_base(base_model, config.get("model", {}))
        # In a real implementation, we would load the weights from the checkpoint
        # model.load_state_dict(torch.load(os.path.join(model_path, "model.pt")))
    
    return model

def evaluate_perplexity(model, dataset, batch_size=8, device="cuda"):
    """Evaluate model perplexity on a dataset."""
    model.to(device)
    model.eval()
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating perplexity"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            
            # Compute loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Mask out padding tokens
            shift_mask = attention_mask[..., 1:].contiguous().view(-1)
            masked_loss = loss * shift_mask
            
            # Sum losses and count tokens
            total_loss += masked_loss.sum().item()
            total_tokens += shift_mask.sum().item()
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return {"perplexity": perplexity, "avg_loss": avg_loss}

def generate_text(model, tokenizer, prompts, max_length=100, temperature=0.8, top_k=50, top_p=0.95, device="cuda"):
    """Generate text from prompts using the model."""
    model.to(device)
    model.eval()
    
    generated_texts = []
    
    for prompt in tqdm(prompts, desc="Generating text"):
        # Tokenize prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Initialize generation
        generated_ids = prompt_ids.clone()
        attention_mask = torch.ones_like(prompt_ids)
        
        # Generate tokens auto-regressively
        for _ in range(max_length):
            with torch.no_grad():
                # Get logits from the model
                logits = model(generated_ids, attention_mask)
                
                # Get the logits for the next token
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    
                    # Shift the indices to the right to keep the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Create a mask based on sorted indices
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Update attention mask and append new token
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((1, 1))], dim=1)
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if we generate an EOS token
                if next_token.item() == tokenizer.eos_token_id:
                    break
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        generated_texts.append({
            "prompt": prompt,
            "generated_text": generated_text,
            "completion": generated_text[len(prompt):]
        })
    
    return generated_texts

def evaluate_preference_alignment(reward_model, preference_dataset, batch_size=8, device="cuda"):
    """Evaluate how well the reward model aligns with human preferences."""
    reward_model.to(device)
    reward_model.eval()
    
    dataloader = DataLoader(preference_dataset, batch_size=batch_size, shuffle=False)
    
    correct_predictions = 0
    total_pairs = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating preference alignment"):
            chosen_ids = batch["chosen_ids"].to(device)
            chosen_mask = batch["chosen_mask"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)
            rejected_mask = batch["rejected_mask"].to(device)
            
            # Get rewards for chosen and rejected sequences
            chosen_rewards = reward_model(chosen_ids, chosen_mask)
            rejected_rewards = reward_model(rejected_ids, rejected_mask)
            
            # Check if the model correctly predicts the preference
            correct_predictions += (chosen_rewards > rejected_rewards).sum().item()
            total_pairs += chosen_rewards.size(0)
    
    # Calculate accuracy
    accuracy = correct_predictions / total_pairs
    
    return {"preference_accuracy": accuracy, "correct_predictions": correct_predictions, "total_pairs": total_pairs}

def compare_models(base_model, rlhf_model, tokenizer, prompts, max_length=100, device="cuda"):
    """Compare the outputs of the base model and the RLHF-trained model."""
    # Generate text with both models
    base_outputs = generate_text(base_model, tokenizer, prompts, max_length, device=device)
    rlhf_outputs = generate_text(rlhf_model, tokenizer, prompts, max_length, device=device)
    
    comparison_results = []
    
    for i in range(len(prompts)):
        comparison_results.append({
            "prompt": prompts[i],
            "base_model_completion": base_outputs[i]["completion"],
            "rlhf_model_completion": rlhf_outputs[i]["completion"]
        })
    
    return comparison_results

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize tokenizer
    model_type = config.get("model", {}).get("type", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define model paths
    sft_model_path = os.path.join(args.model_dir, "sft_model")
    reward_model_path = os.path.join(args.model_dir, "reward_model")
    rlhf_model_path = os.path.join(args.model_dir, "rlhf_model")
    
    # Evaluate perplexity
    if args.eval_type in ["perplexity", "all"]:
        if args.test_data is None:
            logger.warning("Skipping perplexity evaluation: no test data provided")
        else:
            logger.info("Evaluating model perplexity")
            
            # Load model
            sft_model = load_model(sft_model_path, config, model_type="policy")
            rlhf_model = load_model(rlhf_model_path, config, model_type="policy")
            
            # Load test dataset
            test_dataset = TextDataset(
                data_path=args.test_data,
                tokenizer=tokenizer,
                max_length=config.get("model", {}).get("max_seq_len", 1024)
            )
            
            # Evaluate models
            sft_perplexity = evaluate_perplexity(sft_model, test_dataset, args.batch_size, device)
            rlhf_perplexity = evaluate_perplexity(rlhf_model, test_dataset, args.batch_size, device)
            
            # Save results
            perplexity_results = {
                "sft_model": sft_perplexity,
                "rlhf_model": rlhf_perplexity
            }
            
            with open(os.path.join(args.output_dir, "perplexity_results.json"), 'w') as f:
                json.dump(perplexity_results, f, indent=2)
                
            logger.info(f"SFT model perplexity: {sft_perplexity['perplexity']:.4f}")
            logger.info(f"RLHF model perplexity: {rlhf_perplexity['perplexity']:.4f}")
    
    # Evaluate text generation
    if args.eval_type in ["generation", "all"]:
        if args.prompt_data is None:
            logger.warning("Skipping generation evaluation: no prompt data provided")
        else:
            logger.info("Evaluating text generation")
            
            # Load model
            rlhf_model = load_model(rlhf_model_path, config, model_type="policy")
            
            # Load prompt dataset
            prompt_dataset = PromptDataset(
                data_path=args.prompt_data,
                tokenizer=tokenizer,
                max_length=config.get("model", {}).get("max_seq_len", 1024)
            )
            
            # Select a subset of prompts
            prompt_texts = [prompt_dataset[i]["prompt_text"] for i in range(min(args.num_samples, len(prompt_dataset)))]
            
            # Generate text
            generated_texts = generate_text(
                rlhf_model,
                tokenizer,
                prompt_texts,
                max_length=100,
                device=device
            )
            
            # Save results
            with open(os.path.join(args.output_dir, "generation_results.json"), 'w') as f:
                json.dump(generated_texts, f, indent=2)
                
            logger.info(f"Generated texts saved to {os.path.join(args.output_dir, 'generation_results.json')}")
    
    # Evaluate preference alignment
    if args.eval_type in ["preference", "all"]:
        if args.preference_data is None:
            logger.warning("Skipping preference evaluation: no preference data provided")
        else:
            logger.info("Evaluating preference alignment")
            
            # Load reward model
            reward_model = load_model(reward_model_path, config, model_type="reward")
            
            # Load preference dataset
            preference_dataset = PreferenceDataset(
                data_path=args.preference_data,
                tokenizer=tokenizer,
                max_length=config.get("model", {}).get("max_seq_len", 1024)
            )
            
            # Evaluate preference alignment
            preference_results = evaluate_preference_alignment(
                reward_model,
                preference_dataset,
                batch_size=args.batch_size,
                device=device
            )
            
            # Save results
            with open(os.path.join(args.output_dir, "preference_results.json"), 'w') as f:
                json.dump(preference_results, f, indent=2)
                
            logger.info(f"Preference alignment accuracy: {preference_results['preference_accuracy']:.4f}")
    
    # Compare base and RLHF models
    if args.eval_type in ["comparison", "all"]:
        if args.prompt_data is None:
            logger.warning("Skipping model comparison: no prompt data provided")
        else:
            logger.info("Comparing base and RLHF models")
            
            # Load models
            sft_model = load_model(sft_model_path, config, model_type="policy")
            rlhf_model = load_model(rlhf_model_path, config, model_type="policy")
            
            # Load prompt dataset
            prompt_dataset = PromptDataset(
                data_path=args.prompt_data,
                tokenizer=tokenizer,
                max_length=config.get("model", {}).get("max_seq_len", 1024)
            )
            
            # Select a subset of prompts
            prompt_texts = [prompt_dataset[i]["prompt_text"] for i in range(min(args.num_samples, len(prompt_dataset)))]
            
            # Compare models
            comparison_results = compare_models(
                sft_model,
                rlhf_model,
                tokenizer,
                prompt_texts,
                max_length=100,
                device=device
            )
            
            # Save results
            with open(os.path.join(args.output_dir, "comparison_results.json"), 'w') as f:
                json.dump(comparison_results, f, indent=2)
                
            logger.info(f"Model comparison results saved to {os.path.join(args.output_dir, 'comparison_results.json')}")
    
    logger.info("Evaluation completed!")

if __name__ == "__main__":
    main() 