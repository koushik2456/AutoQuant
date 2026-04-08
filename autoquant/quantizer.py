import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from .sensitivity import MultiMetricSensitivityAnalyzer


# =========================
# INT4 BIT PACKING
# =========================

def pack_int4(x):
    x = x.clamp(-8, 7).to(torch.int8)
    x = x + 8

    high = x[:, ::2]
    low = x[:, 1::2]

    return (high << 4) | low


def unpack_int4(packed, shape):
    high = (packed >> 4) & 0xF
    low = packed & 0xF

    x = torch.stack([high, low], dim=-1).view(packed.size(0), -1)
    x = x[:, :shape[1]]

    return x.float() - 8


# =========================
# QUANT LINEAR
# =========================

class QuantLinear(nn.Module):
    def __init__(self, weight, bits=4, group_size=128):
        super().__init__()

        self.bits = bits
        self.group_size = group_size
        self.original_shape = weight.shape

        out_features, in_features = weight.shape

        num_groups = (in_features + group_size - 1) // group_size
        pad = num_groups * group_size - in_features

        if pad > 0:
            weight = F.pad(weight, (0, pad))

        weight = weight.view(out_features, num_groups, group_size)

        qmax = 2**(bits-1)-1
        max_val = weight.abs().amax(dim=2, keepdim=True)
        scale = max_val / (qmax + 1e-8)

        q = torch.round(weight / scale)
        q = torch.clamp(q, -qmax, qmax).to(torch.int8)

        if bits == 4:
            q = q.view(out_features, -1)
            q = pack_int4(q)

        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale)
        self.num_groups = num_groups

    def forward(self, x):
        B = x.shape[0]

        if x.shape[-1] < self.num_groups * self.group_size:
            pad = self.num_groups * self.group_size - x.shape[-1]
            x = F.pad(x, (0, pad))

        x = x.view(B, self.num_groups, self.group_size)

        if self.bits == 4:
            q = unpack_int4(self.qweight, self.original_shape)
            q = q.view(self.original_shape[0], self.num_groups, self.group_size)
        else:
            q = self.qweight.float()

        w = q * self.scale
        return torch.einsum("bgs,ogs->bo", x, w)


# =========================
# AUTOQUANTIZER
# =========================

class AutoQuantizer:
    def __init__(self, model_name, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def get_calibration_data(self, n=100):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        return [t for t in ds["text"] if len(t) > 20][:n]

    def analyze(self):
        texts = self.get_calibration_data()
        analyzer = MultiMetricSensitivityAnalyzer(self.model, self.device)
        return analyzer.analyze(texts, self.tokenizer)

    def assign_bits(self, scores):
        config = {}
        for k, v in scores.items():
            if v > 0.7:
                config[k] = 8
            elif v > 0.4:
                config[k] = 4
            else:
                config[k] = 2
        return config

    def quantize(self, config):
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and name in config:
                bits = config[name]
                qlayer = QuantLinear(module.weight.data, bits)

                parent = self._get_parent(name)
                setattr(parent, name.split('.')[-1], qlayer)

    def _get_parent(self, name):
        obj = self.model
        for p in name.split('.')[:-1]:
            obj = getattr(obj, p)
        return obj

    def run(self):
        print("🔬 Sensitivity...")
        scores = self.analyze()

        print("⚙️ Bit allocation...")
        config = self.assign_bits(scores)

        print("🔧 Quantizing...")
        self.quantize(config)

        print("✅ Done")
        return self.model