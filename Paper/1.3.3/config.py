# COTTON Configuration File
# Based on Table 2 from the paper

import os
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class COTTONConfig:
    """
    Configuration class replicating exact hyperparameters from Table 2
    """
    # ===== Model Configuration =====
    base_model_name: str = "codellama/CodeLlama-7b-hf"
    max_input_length: int = 256
    max_output_length: int = 256
    
    # ===== LoRA Configuration (Table 2) =====
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    
    # ===== Training Configuration (Table 2) =====
    learning_rate: float = 1e-4
    training_batch_size: int = 1
    num_epochs: int = 20
    early_stop: int = 5
    optimizer: str = "AdamW"
    random_seed: int = 42
    
    # ===== Data Configuration =====
    data_dir: str = "cotton_data"
    model_save_path: str = "cotton_model"
    dataset_sources: List[str] = field(default_factory=lambda: ["openai/HumanEval"])
    sample_size: int = 100
    
    # ===== Hardware Configuration =====
    device: str = "auto"  # "auto", "cpu", "cuda:0"
    torch_dtype: str = "float16"  # "float16", "float32"
    low_cpu_mem_usage: bool = True
    
    # ===== Evaluation Configuration =====
    eval_batch_size: int = 1
    eval_metrics: List[str] = field(default_factory=lambda: [
        "bleu_1", "bleu_2", "bleu_3", "bleu_4", 
        "meteor", "rouge_l", "consistency"
    ])
    
    # ===== Agent Configuration (Section 3.1) =====
    use_multi_agent: bool = True
    quality_threshold: float = 0.7
    consistency_threshold: float = 0.8
    
    # ===== Logging Configuration =====
    log_level: str = "INFO"
    log_file: Optional[str] = "cotton.log"
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration and create directories"""
        # Create necessary directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_save_path, exist_ok=True)
        
        # Validate hyperparameters
        assert self.lora_r > 0, "LoRA rank must be positive"
        assert self.lora_alpha > 0, "LoRA alpha must be positive"
        assert 0.0 <= self.lora_dropout <= 1.0, "LoRA dropout must be between 0 and 1"
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.num_epochs > 0, "Number of epochs must be positive"
        
    def to_dict(self) -> dict:
        """Convert config to dictionary"""
        return {
            'model': {
                'base_model_name': self.base_model_name,
                'max_input_length': self.max_input_length,
                'max_output_length': self.max_output_length,
            },
            'lora': {
                'r': self.lora_r,
                'alpha': self.lora_alpha,
                'dropout': self.lora_dropout,
                'target_modules': self.target_modules,
            },
            'training': {
                'learning_rate': self.learning_rate,
                'batch_size': self.training_batch_size,
                'epochs': self.num_epochs,
                'optimizer': self.optimizer,
                'seed': self.random_seed,
            },
            'data': {
                'data_dir': self.data_dir,
                'model_save_path': self.model_save_path,
                'dataset_sources': self.dataset_sources,
                'sample_size': self.sample_size,
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'COTTONConfig':
        """Create config from dictionary"""
        # Flatten nested dictionary
        flat_config = {}
        for section, params in config_dict.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    # Map dictionary keys to class attributes
                    if section == 'model' and key == 'base_model_name':
                        flat_config['base_model_name'] = value
                    elif section == 'lora' and key == 'r':
                        flat_config['lora_r'] = value
                    # Add more mappings as needed
            else:
                flat_config[section] = params
        
        return cls(**flat_config)
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'COTTONConfig':
        """Load configuration from JSON file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

# Default configuration instance
default_config = COTTONConfig()

# Configuration variants for different use cases
class COTTONConfigVariants:
    """Pre-defined configuration variants for different scenarios"""
    
    @staticmethod
    def demo_config() -> COTTONConfig:
        """Lightweight configuration for demonstration"""
        config = COTTONConfig()
        config.sample_size = 20
        config.num_epochs = 2
        config.torch_dtype = "float32"
        return config
    
    @staticmethod
    def production_config() -> COTTONConfig:
        """Full configuration for production training"""
        config = COTTONConfig()
        config.sample_size = 9000  # Full CodeCoT-9k dataset
        config.num_epochs = 20
        config.torch_dtype = "float16"
        config.device = "cuda:0"
        return config
    
    @staticmethod
    def research_config() -> COTTONConfig:
        """Configuration for research experiments"""
        config = COTTONConfig()
        config.sample_size = 1000
        config.num_epochs = 10
        config.eval_batch_size = 4
        config.verbose = True
        return config
    
    @staticmethod
    def ablation_config() -> COTTONConfig:
        """Configuration for ablation studies"""
        config = COTTONConfig()
        config.use_multi_agent = False  # For ablation study
        config.consistency_threshold = 0.0  # Disable consistency checker
        config.sample_size = 500
        return config

# Environment-specific configurations
class EnvironmentConfigs:
    """Hardware and environment-specific configurations"""
    
    @staticmethod
    def cpu_only_config() -> COTTONConfig:
        """Configuration for CPU-only environments"""
        config = COTTONConfig()
        config.device = "cpu"
        config.torch_dtype = "float32"
        config.training_batch_size = 1
        config.sample_size = 50
        return config
    
    @staticmethod
    def rtx_3090_config() -> COTTONConfig:
        """Optimized for RTX 3090 (24GB VRAM)"""
        config = COTTONConfig()
        config.device = "cuda:0"
        config.torch_dtype = "float16"
        config.training_batch_size = 1
        config.low_cpu_mem_usage = True
        return config
    
    @staticmethod
    def a100_config() -> COTTONConfig:
        """Optimized for A100 (40GB/80GB VRAM)"""
        config = COTTONConfig()
        config.device = "cuda:0"
        config.torch_dtype = "float16"
        config.training_batch_size = 2
        config.eval_batch_size = 4
        return config
    
    @staticmethod
    def multi_gpu_config() -> COTTONConfig:
        """Configuration for multi-GPU training"""
        config = COTTONConfig()
        config.device = "auto"
        config.torch_dtype = "float16"
        config.training_batch_size = 4
        config.eval_batch_size = 8
        return config

if __name__ == "__main__":
    # Demo configuration usage
    print("COTTON Configuration Examples:")
    print("=" * 50)
    
    # Default config
    config = COTTONConfig()
    print("Default Configuration:")
    print(f"Base Model: {config.base_model_name}")
    print(f"LoRA r: {config.lora_r}, alpha: {config.lora_alpha}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Batch Size: {config.training_batch_size}")
    print(f"Epochs: {config.num_epochs}")
    
    # Save and load example
    config.save("cotton_config.json")
    loaded_config = COTTONConfig.load("cotton_config.json")
    print(f"\nConfiguration saved and loaded successfully!")
    
    # Variants
    print(f"\nAvailable Variants:")
    print(f"- Demo Config: {COTTONConfigVariants.demo_config().sample_size} samples")
    print(f"- Production Config: {COTTONConfigVariants.production_config().sample_size} samples")
    print(f"- Research Config: {COTTONConfigVariants.research_config().sample_size} samples")
    
    # Environment configs
    print(f"\nEnvironment Configs:")
    print(f"- CPU Only: {EnvironmentConfigs.cpu_only_config().device}")
    print(f"- RTX 3090: {EnvironmentConfigs.rtx_3090_config().device}")
    print(f"- A100: {EnvironmentConfigs.a100_config().device}")
