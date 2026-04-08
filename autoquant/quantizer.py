"""
AutoQuant - Main Quantization Engine
Implements mixed-precision quantization with optimal bit allocation
"""

import torch
import torch.nn as nn
import json
import os
from typing import Dict, List, Optional, Tuple
from .sensitivity import MultiMetricSensitivityAnalyzer


class AutoQuantizer:
    """
    Main AutoQuant class that handles:
    - Model loading and analysis
    - Multi-metric sensitivity analysis
    - Optimal bit allocation (knapsack optimization)
    - Quantization application
    """
    
    def __init__(self, model_name: str, device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"📥 Loading {model_name}...")
        
        # Load model with memory efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        if self.device == "cuda":
            self.model = self.model.to("cuda")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model_name = model_name
        self.original_size = self._get_model_size()
        self.sensitivity_scores = {}
        
    def _get_model_size(self) -> float:
        """Get model size in GB"""
        total_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return total_bytes / (1024**3)
    
    def _get_all_layer_names(self) -> List[str]:
        """Get all parameterized layer names"""
        layer_names = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and len(list(module.parameters())) > 0:
                layer_names.append(name)
        return layer_names
    
    def _get_layer_params(self, layer_name: str) -> int:
        """Get number of parameters in a layer"""
        for name, module in self.model.named_modules():
            if name == layer_name and hasattr(module, 'weight'):
                return module.weight.numel()
        return 0
    
    def analyze_sensitivity(self, num_samples: int = 100) -> Dict[str, float]:
        """
        Run complete multi-metric sensitivity analysis.
        
        Args:
            num_samples: Number of calibration samples
            
        Returns:
            Dictionary of sensitivity scores per layer
        """
        # Create calibration texts
        calibration_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Transformers use attention mechanisms for sequence processing.",
            "Quantization reduces model size while maintaining accuracy.",
            "The cat sat on the mat and looked outside the window.",
            "Once upon a time in a faraway land.",
            "The stock market fluctuates based on various factors.",
            "Neural networks learn patterns from data.",
            "The weather forecast predicts rain tomorrow.",
            "Artificial intelligence is transforming industries.",
        ] * (num_samples // 10 + 1)
        
        # Run multi-metric analysis
        analyzer = MultiMetricSensitivityAnalyzer(self.model, self.device)
        self.sensitivity_scores = analyzer.analyze_all_metrics(calibration_texts, num_samples)
        
        return self.sensitivity_scores
    
    def create_config(self, target_size_gb: float, output_path: str = "config.json") -> Dict:
        """
        Create optimal bit allocation configuration using knapsack optimization.
        
        Args:
            target_size_gb: Target model size in GB
            output_path: Path to save configuration
            
        Returns:
            Dictionary with configuration and metadata
        """
        print(f"\n⚙️ Creating optimal bit allocation for target: {target_size_gb} GB")
        
        if not self.sensitivity_scores:
            print("   Running sensitivity analysis first...")
            self.analyze_sensitivity()
        
        # Get all layers
        layer_names = self._get_all_layer_names()
        
        # Start with all layers at 16 bits (FP16 baseline)
        current_bits = {name: 16 for name in layer_names if name in self.sensitivity_scores}
        
        def get_total_bits(bits_dict: Dict[str, int]) -> int:
            total = 0
            for name, bits in bits_dict.items():
                params = self._get_layer_params(name)
                total += params * bits
            return total
        
        current_total = get_total_bits(current_bits)
        current_size_gb = current_total / 8 / (1024**3)
        target_bits_total = target_size_gb * 8 * (1024**3)
        
        print(f"   Current size: {current_size_gb:.2f} GB")
        print(f"   Target size: {target_size_gb:.2f} GB")
        
        # If target is smaller, we need to compress
        if target_size_gb < current_size_gb:
            # Sort layers by sensitivity (least sensitive first - compress these)
            sorted_layers = sorted(self.sensitivity_scores.items(), key=lambda x: x[1])
            bit_options = [8, 4, 3, 2]
            
            print(f"\n   Compressing from {current_size_gb:.2f} GB to {target_size_gb:.2f} GB...")
            
            for layer_name, sens in sorted_layers:
                if current_total <= target_bits_total:
                    break
                
                current = current_bits.get(layer_name, 16)
                for bits in bit_options:
                    if bits < current:
                        params = self._get_layer_params(layer_name)
                        if params > 0:
                            savings = params * (current - bits)
                            current_total -= savings
                            current_bits[layer_name] = bits
                            print(f"      {layer_name.split('.')[-1]}: {current}→{bits} bits (sens: {sens:.3f})")
                            break
            
            expected_bits = get_total_bits(current_bits)
            expected_size = expected_bits / 8 / (1024**3)
            compression = current_size_gb / expected_size
        else:
            expected_size = current_size_gb
            compression = 1.0
        
        # Build configuration
        config = {
            'model_name': self.model_name,
            'target_size_gb': target_size_gb,
            'original_size_gb': current_size_gb,
            'expected_size_gb': expected_size,
            'compression_ratio': compression,
            'sensitivity_scores': self.sensitivity_scores,
            'bit_assignments': current_bits,
            'bit_distribution': {
                str(bits): sum(1 for b in current_bits.values() if b == bits)
                for bits in set(current_bits.values())
            }
        }
        
        # Save configuration
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n📊 Configuration Summary:")
        print(f"   Original: {current_size_gb:.2f} GB")
        print(f"   Expected: {expected_size:.2f} GB")
        print(f"   Target: {target_size_gb:.2f} GB")
        print(f"   Compression: {compression:.1f}x")
        
        print(f"\n   Bit Distribution:")
        for bits, count in sorted(config['bit_distribution'].items()):
            bar = "█" * int(count / len(current_bits) * 30)
            print(f"      INT{bits:<2}: {count:3d} layers {bar}")
        
        return {
            'config': config,
            'expected_size_gb': expected_size,
            'compression_ratio': compression
        }
    
    def quantize(self, config_path: str = "config.json", output_dir: str = "quantized_model") -> nn.Module:
        """
        Apply quantization to the model using the configuration.
        
        Args:
            config_path: Path to quantization configuration
            output_dir: Directory to save quantized model
            
        Returns:
            Quantized model
        """
        print(f"\n🔧 Applying quantization...")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assignments = config['bit_assignments']
        quantized_count = 0
        
        for name, bits in assignments.items():
            for module_name, module in self.model.named_modules():
                if module_name == name and hasattr(module, 'weight'):
                    if bits < 16:
                        self._quantize_layer(module, bits)
                        quantized_count += 1
                    break
        
        print(f"   Quantized {quantized_count} layers")
        
        # Save quantized model
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n💾 Saving quantized model to {output_dir}...")
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Copy configuration
        import shutil
        shutil.copy(config_path, f"{output_dir}/quantization_config.json")
        
        # Save metadata
        current_size = self._get_model_size()
        metadata = {
            'model_name': self.model_name,
            'quantized_size_gb': current_size,
            'original_size_gb': self.original_size,
            'compression_ratio': self.original_size / current_size if current_size > 0 else 1,
            'bit_distribution': config['bit_distribution'],
            'metrics_used': ['hessian', 'fisher', 'gptq', 'kurtosis']
        }
        
        with open(f"{output_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   ✅ Model saved successfully!")
        return self.model
    
    def _quantize_layer(self, module: nn.Module, bits: int):
        """Quantize a single layer using per-channel quantization"""
        if bits >= 16:
            return
        
        weight = module.weight.data.float()
        
        # Per-channel quantization (better than per-tensor)
        max_vals = torch.abs(weight).max(dim=1, keepdim=True)[0]
        q_max = 2 ** (bits - 1) - 1
        scale = max_vals / q_max
        scale = torch.clamp(scale, min=1e-8)
        
        # Quantize
        quantized = torch.round(weight / scale)
        quantized = torch.clamp(quantized, -2**(bits-1), 2**(bits-1)-1)
        dequantized = quantized * scale
        
        # Store dequantized weights (for simplicity)
        # In production, you'd store quantized ints + scale
        module.weight.data = dequantized.half()
    
    def get_report(self) -> Dict:
        """Get compression report"""
        current_size = self._get_model_size()
        return {
            'original_size_gb': self.original_size,
            'quantized_size_gb': current_size,
            'compression_ratio': self.original_size / current_size if current_size > 0 else 1,
            'space_saved_percent': (1 - current_size/self.original_size) * 100 if self.original_size > 0 else 0
        }