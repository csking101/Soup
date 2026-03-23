"""Pydantic schemas for soup.yaml config — single source of truth."""

from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class LoraConfig(BaseModel):
    r: int = Field(default=64, description="LoRA rank")
    alpha: int = Field(default=16, description="LoRA alpha")
    dropout: float = Field(default=0.05, description="LoRA dropout")
    target_modules: Union[str, List[str]] = Field(
        default="auto",
        description="Target modules for LoRA. 'auto' = let peft decide.",
    )


class DataConfig(BaseModel):
    train: str = Field(..., description="Path to training data or HF dataset name")
    format: Literal["alpaca", "sharegpt", "chatml", "dpo", "auto"] = Field(
        default="auto",
        description="Data format",
    )
    val_split: float = Field(default=0.1, ge=0.0, le=0.5, description="Validation split ratio")
    max_length: int = Field(default=2048, description="Max sequence length in tokens")


class TrainingConfig(BaseModel):
    epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    lr: float = Field(default=2e-5, gt=0, description="Learning rate")
    batch_size: Union[int, Literal["auto"]] = Field(
        default="auto",
        description="Batch size. 'auto' = find max that fits in memory.",
    )
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    warmup_ratio: float = Field(default=0.03, ge=0.0, le=0.5)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0)
    lora: LoraConfig = Field(default_factory=LoraConfig)
    quantization: Literal["4bit", "8bit", "none"] = Field(
        default="4bit",
        description="Quantization: 4bit (QLoRA), 8bit, or none (full precision)",
    )
    optimizer: str = Field(default="adamw_torch", description="Optimizer name")
    scheduler: str = Field(default="cosine", description="LR scheduler type")
    save_steps: int = Field(default=100, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, description="Log metrics every N steps")
    # DPO-specific
    dpo_beta: float = Field(
        default=0.1, gt=0, description="DPO beta — KL penalty coefficient"
    )
    # GRPO-specific
    grpo_beta: float = Field(
        default=0.1, gt=0, description="GRPO beta — KL penalty coefficient"
    )
    num_generations: int = Field(
        default=4, ge=2, description="Number of generations per prompt for GRPO"
    )
    reward_fn: Optional[str] = Field(
        default="accuracy",
        description="Reward function: 'accuracy', 'format', or path to custom .py file",
    )


class SoupConfig(BaseModel):
    """Root config for soup.yaml."""

    base: str = Field(..., description="Base model name or path (HF model ID)")
    task: Literal["sft", "dpo", "grpo"] = Field(default="sft", description="Training task type")
    backend: Literal["transformers", "unsloth"] = Field(
        default="transformers",
        description="Training backend: transformers (default) or unsloth (2-5x faster)",
    )
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: str = Field(default="./output", description="Output directory for trained model")
    experiment_name: Optional[str] = Field(default=None, description="Experiment name for tracking")


# --- Built-in templates ---

TEMPLATES: dict[str, str] = {
    "chat": """# Soup template: Chat Assistant
# Fine-tune a model for conversational chat

base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1
  max_length: 2048

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    "code": """# Soup template: Code Model
# Fine-tune a model for code generation / completion

base: codellama/CodeLlama-7b-Instruct-hf
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/code_train.jsonl
  format: alpaca
  val_split: 0.1
  max_length: 4096

training:
  epochs: 2
  lr: 1e-5
  batch_size: auto
  lora:
    r: 128
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
    "reasoning": """# Soup template: Reasoning / GRPO
# Fine-tune a model for chain-of-thought reasoning with GRPO

base: meta-llama/Llama-3.1-8B-Instruct
task: grpo
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/reasoning_train.jsonl
  format: sharegpt
  val_split: 0.1
  max_length: 4096

training:
  epochs: 3
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: 4bit
  grpo_beta: 0.1
  num_generations: 4
  reward_fn: accuracy

output: ./output
""",
    "medical": """# Soup template: Medical / Domain Expert
# Fine-tune a model with domain-specific knowledge

base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install 'soup-cli[fast]'

data:
  train: ./data/medical_train.jsonl
  format: alpaca
  val_split: 0.15
  max_length: 2048

training:
  epochs: 5
  lr: 1e-5
  batch_size: auto
  gradient_accumulation_steps: 8
  lora:
    r: 128
    alpha: 32
    target_modules: auto
  quantization: 4bit

output: ./output
""",
}
