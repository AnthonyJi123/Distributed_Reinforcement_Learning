import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os
import json
import logging
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """
    Dataset for supervised fine-tuning.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file or directory
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.examples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load data from file."""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif data_path.endswith('.jsonl'):
            examples = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line))
            return examples
        else:
            # Assume it's a directory with .txt files
            examples = []
            for filename in os.listdir(data_path):
                if filename.endswith('.txt'):
                    with open(os.path.join(data_path, filename), 'r', encoding='utf-8') as f:
                        text = f.read()
                        examples.append({"text": text})
            return examples
            
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get an example from the dataset."""
        example = self.examples[idx]
        
        # Get text from example
        if "text" in example:
            text = example["text"]
        elif "prompt" in example and "completion" in example:
            text = example["prompt"] + example["completion"]
        else:
            # Try to extract text from the example
            text = str(example)
            
        # Tokenize the text
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract tensors
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids  # For autoregressive language modeling
        }

class PreferenceDataset(Dataset):
    """
    Dataset for preference learning.
    Contains pairs of chosen and rejected completions.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.examples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.examples)} preference pairs from {data_path}")
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load preference data from file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                return json.load(f)
            elif data_path.endswith('.jsonl'):
                examples = []
                for line in f:
                    examples.append(json.loads(line))
                return examples
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
                
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair from the dataset."""
        example = self.examples[idx]
        
        # Extract prompt, chosen and rejected completions
        prompt = example.get("prompt", "")
        
        # Some datasets have different key names
        if "chosen" in example and "rejected" in example:
            chosen = example["chosen"]
            rejected = example["rejected"]
        elif "chosen_completion" in example and "rejected_completion" in example:
            chosen = example["chosen_completion"]
            rejected = example["rejected_completion"]
        else:
            raise ValueError(f"Could not find chosen/rejected texts in example: {example}")
            
        # Tokenize chosen text (prompt + chosen completion)
        chosen_encodings = self.tokenizer(
            prompt + chosen,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize rejected text (prompt + rejected completion)
        rejected_encodings = self.tokenizer(
            prompt + rejected,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract tensors
        chosen_ids = chosen_encodings["input_ids"].squeeze(0)
        chosen_mask = chosen_encodings["attention_mask"].squeeze(0)
        rejected_ids = rejected_encodings["input_ids"].squeeze(0)
        rejected_mask = rejected_encodings["attention_mask"].squeeze(0)
        
        return {
            "chosen_ids": chosen_ids,
            "chosen_mask": chosen_mask,
            "rejected_ids": rejected_ids,
            "rejected_mask": rejected_mask
        }

class PromptDataset(Dataset):
    """
    Dataset for RLHF prompts.
    Used in the RL phase to generate completions.
    """
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the dataset file
            tokenizer: Tokenizer to use for encoding text
            max_length: Maximum sequence length
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        self.examples = self._load_data(data_path)
        logger.info(f"Loaded {len(self.examples)} prompts from {data_path}")
        
    def _load_data(self, data_path: str) -> List[Dict[str, Any]]:
        """Load prompt data from file."""
        with open(data_path, 'r', encoding='utf-8') as f:
            if data_path.endswith('.json'):
                return json.load(f)
            elif data_path.endswith('.jsonl'):
                examples = []
                for line in f:
                    examples.append(json.loads(line))
                return examples
            else:
                # Assume it's a text file with one prompt per line
                return [{"prompt": line.strip()} for line in f.readlines() if line.strip()]
                
    def __len__(self) -> int:
        return len(self.examples)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a prompt from the dataset."""
        example = self.examples[idx]
        
        # Extract prompt
        if "prompt" in example:
            prompt = example["prompt"]
        elif "text" in example:
            prompt = example["text"]
        else:
            # Try to extract prompt from the example
            prompt = str(example)
            
        # Tokenize the prompt
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract tensors
        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_text": prompt
        }

def create_data_loaders(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for a dataset.
    
    Args:
        dataset: Dataset to create DataLoader for
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        num_workers: Number of workers for loading data
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    ) 