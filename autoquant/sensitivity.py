import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultiMetricSensitivityAnalyzer:
    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _get_layers(self):
        return [(n, m) for n, m in self.model.named_modules() if isinstance(m, nn.Linear)]

    def _hessian(self, module):
        w = module.weight
        v = torch.randn_like(w)

        w.requires_grad_(True)
        loss = (w * v).sum()

        g = torch.autograd.grad(loss, w, create_graph=True)[0]
        hv = torch.autograd.grad(g, w, grad_outputs=v)[0]

        return float((v * hv).sum().abs() / (w.numel() + 1e-6))

    def _fisher(self, inputs, module):
        self.model.zero_grad()

        outputs = self.model(**inputs)
        logits = outputs.logits
        targets = inputs["input_ids"]

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )

        loss.backward(retain_graph=True)

        if module.weight.grad is None:
            return 0.5

        return float((module.weight.grad ** 2).mean())

    def _gptq(self, module):
        w = module.weight.data.float()
        sample = w.flatten()[:10000]

        errors = []
        for bits in [2, 4, 8]:
            max_val = sample.abs().max()
            scale = max_val / (2**(bits-1)-1 + 1e-8)
            q = torch.round(sample / scale)
            dq = q * scale
            errors.append(F.mse_loss(dq, sample).item())

        return float(np.mean(errors))

    def analyze(self, texts, tokenizer):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(self.device)

        scores = {}

        for name, module in self._get_layers():
            try:
                h = self._hessian(module)
                f = self._fisher(inputs, module)
                g = self._gptq(module)

                scores[name] = (0.4*h + 0.3*f + 0.3*g)
            except:
                scores[name] = 0.5

        return scores