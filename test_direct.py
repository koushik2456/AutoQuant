"""
Direct test - bypasses web UI
"""

import json

from autoquant import AutoQuantizer

print("=" * 60)
print("Direct Quantization Test")
print("=" * 60)

quantizer = AutoQuantizer("gpt2")

print("\n1. Analyzing sensitivity...")
sensitivity = quantizer.analyze_sensitivity(num_samples=100)
print(f"   Found {len(sensitivity)} layers")

print("\n2. Creating config for 1GB target...")
config_result = quantizer.create_config(1.0)
print(f"   Expected size: {config_result['expected_size_gb']:.2f} GB")
print(f"   Compression: {config_result['compression_ratio']:.1f}x")

with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config_result["config"], f, indent=2)

print("\n3. Applying quantization...")
quantizer.quantize("config.json", "quantized_test")

# Report
report = quantizer.get_report()
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Original size: {report['original_size_gb']:.2f} GB")
print(f"Quantized size: {report['quantized_size_gb']:.2f} GB")
print(f"Compression: {report['compression_ratio']:.1f}x")
print("\n✅ Success! Model saved to 'quantized_test/'")