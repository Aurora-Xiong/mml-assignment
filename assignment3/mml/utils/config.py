"""
    Project's main config.
"""

import os
from dataclasses import dataclass


@dataclass
class ConfigS:
    """
    Project's main config.
    """

    vision_model: str = "openai/clip-vit-base-patch32"
    text_model: str = "gpt2"
    seed: int = 100
    num_workers: int = 0 # must set to 0 because of cuda error
    train_size: int = 0.84
    val_size: int = 0.13
    epochs: int = 150
    lr: int = 1e-5
    k: float = 0.33
    batch_size_exp: int = 6
    ep_len: int = 4
    num_layers: int = 6
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 40
    dropout: float = 0.1
    weights_dir: str = os.path.join("weights", "small")


@dataclass
class ConfigL:
    """
    Project's main config.
    """

    vision_model: str = "openai/clip-vit-large-patch14"
    text_model: str = "gpt2-medium"
    seed: int = 100
    num_workers: int = 0 # must set to 0 because of cuda error
    train_size: int = 0.84
    val_size: int = 0.13
    epochs: int = 120
    lr: int = 5e-4
    k: float = 0.3
    batch_size_exp: int = 5
    ep_len: int = 4
    num_layers: int = 5
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 40
    dropout: float = 0.08
    weights_dir: str = os.path.join("weights", "large")
    
@dataclass
class QwenConfig:
    """
    Project's main config.
    """
    vision_model: str = "openai/clip-vit-large-patch14"
    blip_model: str = "Salesforce/blip2-opt-2.7b"
    text_model: str = "Qwen/Qwen2.5-0.5B"
    seed: int = 100
    num_workers: int = 0
    train_size: int = 0.8
    val_size: int = 0.1
    epochs: int = 120
    lr: int = 1e-5
    k: float = 0.3
    batch_size_exp: int = 5
    num_query_tokens: int = 32
    mp_hidden_size: int = 1408
    max_len: int = 40
    weights_dir: str = os.path.join("weights", "qwen")