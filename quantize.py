#!/usr/bin/env python3
"""
AutoQuant CLI - Command line interface
"""

import argparse
import torch
from autoquant import AutoQuantizer

def main():
    parser = argparse.ArgumentParser(description="AutoQuant - LLM Quantization")
    parser.add_argument("--model", "-m", default="gpt2", help="Model name")
    parser.add_argument("--target", "-t", type=float, default=0.5, help="Target size in GB")
    parser.add_argument("--output", "-o", default="quantized_model", help="Output directory")
    parser.add_argument(
        "--samples",
        "-n",
        type=int,
        default=100,
        help="Sensitivity calibration samples (4–200)",
    )
    parser.add_argument(
        "--config-out",
        default="config.json",
        help="Path to write full config JSON before quantize",
    )
    parser.add_argument("--analyze-only", "-a", action="store_true", help="Only analyze")
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔢 AutoQuant - Memory-Efficient LLM Quantization")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    quantizer = AutoQuantizer(args.model)
    n = max(4, min(int(args.samples), 200))
    quantizer.analyze_sensitivity(num_samples=n)
    config_result = quantizer.create_config(args.target)
    
    if args.analyze_only:
        print(f"\n📊 Analysis complete")
        print(f"   Would compress from {quantizer.original_size:.2f} GB to {config_result['expected_size_gb']:.2f} GB")
    else:
        # Save config and quantize
        import json

        cfg_path = args.config_out
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(config_result["config"], f, indent=2)

        quantizer.quantize(cfg_path, args.output)
        report = quantizer.get_report()
        
        print("\n" + "="*60)
        print("📊 QUANTIZATION REPORT")
        print("="*60)
        print(f"   Original size: {report['original_size_gb']:.2f} GB")
        print(f"   Quantized size: {report['quantized_size_gb']:.2f} GB")
        print(f"   Compression: {report['compression_ratio']:.1f}x")
        print(f"   Space saved: {report['space_saved_percent']:.1f}%")

if __name__ == "__main__":
    main()