"""
Quick evaluation of structurally quantized model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import json

# We must define the layer locally so the torch.load state_dict keys match
class DynamicQuantizedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, bits: int):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bits = bits
        
        # initialize dummy buffers to hold the loaded weights
        # we don't need to recalculate them, load_state_dict will fill these!
        self.register_buffer('weight_q', torch.zeros((self.out_features, self.in_features), dtype=torch.int8))
        self.register_buffer('scale', torch.zeros((self.out_features, 1), dtype=torch.float16))
        
        if original_linear.bias is not None:
             self.register_buffer('bias', torch.zeros(self.out_features, dtype=torch.float16))
        else:
             self.register_parameter('bias', None)
             
    def forward(self, x):
        weight_deq = self.weight_q.to(self.scale.dtype) * self.scale
        return F.linear(x, weight_deq, self.bias)

def _replace_module_in_model(parent_module, target_name, new_module):
    parts = target_name.split('.')
    current = parent_module
    for p in parts[:-1]:
        current = getattr(current, p)
    setattr(current, parts[-1], new_module)

def evaluate(model_path):
    print("="*60)
    print("📊 Evaluating Structurally Quantized Model")
    print("="*60)
    
    config_path = os.path.join(model_path, "quantization_config.json")
    if not os.path.exists(config_path):
        print("❌ Error: quantization_config.json not found. Is this a valid AutoQuant output?")
        return
        
    with open(config_path, 'r') as f:
        quant_config = json.load(f)
        
    base_model_name = quant_config['model_name']
    print(f"📥 Loading base model skeleton: {base_model_name}")
    
    # Load uninitialized/base model skeleton
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cpu", # Keep in CPU until layers are replaced and loaded
        low_cpu_mem_usage=True,
        use_safetensors=True
    )
    
    # 1. Structural Layer Replacement
    print(f"🔧 Applying dynamic quantization structure...")
    assignments = quant_config.get('bit_assignments', {})
    replaced_count = 0
    for name, bits in assignments.items():
        for module_name, module in list(model.named_modules()):
            if module_name == name and isinstance(module, nn.Linear):
                if bits < 16:
                    new_layer = DynamicQuantizedLinear(module, bits)
                    _replace_module_in_model(model, module_name, new_layer)
                    replaced_count += 1
                break
                
    print(f"   Replaced {replaced_count} layers with INT-backed structural layers.")
    
    # 2. State Loading
    checkpoint_file = os.path.join(model_path, "pytorch_model.bin")
    print(f"💾 Loading packed weights from {checkpoint_file} into geometry...")
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    model.load_state_dict(state_dict)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if torch.cuda.is_available():
        print(f"🚀 Moving packed model to GPU memory...")
        model = model.to("cuda")
    
    # Measure true loaded size in RAM/VRAM
    total_bytes = 0
    for name, module in model.named_modules():
        if isinstance(module, DynamicQuantizedLinear):
            total_bytes += module.weight_q.numel() * module.weight_q.element_size()
            total_bytes += module.scale.numel() * module.scale.element_size()
            if module.bias is not None:
                total_bytes += module.bias.numel() * module.bias.element_size()
        elif isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            for p in module.parameters():
                total_bytes += p.numel() * p.element_size()
                
    size_gb = total_bytes / (1024**3)
    
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    if total_params < 0.1: # if parameters iteration skips custom buffers
        total_params = 1.0 # fallback proxy
    
    print(f"\n📦 Loaded Model Info:")
    # print(f"   Parameters: {total_params:.1f}M")
    print(f"   Real Active Physical Size: {size_gb:.2f} GB")
    
    test_prompt = "The future of AI is"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    
    import time
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n📝 Generated: {generated}")
    print(f"⏱️ Time: {elapsed:.2f}s")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "quantized_model"
    evaluate(path)