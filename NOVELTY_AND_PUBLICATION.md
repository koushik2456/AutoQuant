# AutoQuant (SE-IN-AI): Novelty Mapping and Publication Readiness

**Purpose:** Summarize what this repository *actually* implements, situate it against recent public literature (web-informed as of April 2026), and separate **engineering contributions** from **research claims** that would need experiments to defend in a peer-reviewed venue.

**Scope note:** “Novel” in academia means *new relative to published prior art*, not merely “well implemented.” Mixed-precision quantization and Fisher-style sensitivity are **mature themes**. A strong paper usually combines a **clear algorithmic delta**, **rigorous baselines** (GPTQ, AWQ, uniform PTQ, off-the-shelf mixed precision), and **standard benchmarks** (WikiText-2 / C4 perplexity, downstream tasks, latency on target hardware).

---

## 1. What This Codebase Implements (Ground Truth)

These are the concrete technical elements present in `autoquant/` and the app/CLI:

| Component | Role (as implemented) |
|-----------|------------------------|
| **Sensitivity** (`sensitivity.py`) | Per-layer score = mean over calibration texts of **clipped squared weight gradients** (99th-percentile clip on \|g\|) **times** mean **\|activation\|** from a forward hook; then **min–max** normalized to [0, 1]. Diverse fixed calibration strings (code, SQL, multilingual, dialogue). |
| **Allocation** (`allocator.py`) | **Discrete** mixed precision among **{INT4, INT8, INT16}** under a **target model size (GB)** with **95% effective budget**, **sensitivity “cliff”** locking low-sensitivity layers at INT4, **two-phase greedy** upgrades (4→8 for sensitive layers, then 8→16 only if sensitivity ≥ `INT16_UPGRADE_THRESHOLD`, default 0.75 via env). |
| **Quantized linear** (`quantizer.py`, `DynamicQuantizedLinear`) | **Weight-only** symmetric quantization with **per-group scales** (column groups, default `AUTOQUANT_GROUP_SIZE=128`), **INT8 storage** for quantized levels, **per-output-row FP16 residual** after dequant to absorb rounding error. |
| **Orchestration** | `AutoQuantizer`: load HF causal LM (FP16), sensitivity → `allocate_bits` → replace `nn.Linear` (and GPT-2 `Conv1D` path via utils) → save HF-style folder with `quantization_config.json`, `pytorch_model.bin`, tokenizer copies. |
| **Footprint without full load** | `hf_estimate.py`: meta-device (or CPU fallback) **FP16 footprint estimate** for dashboard/API. |
| **Quality proxy** | `quality_metrics.py` + API: **short-string CE loss** FP16 vs quantized — explicitly **not** full-dataset perplexity. |
| **Product surface** | Flask dashboard, REST APIs, single-flight quantization, optional Ollama listing/chat (orthogonal to quantization math). |

**Important alignment check:** `README.md` describes a focused **PyTorch + Hugging Face** PTQ tool. Older narrative in `AUTOQUANT_BUILD_GUIDE.md` mentions broader backends (e.g. TensorRT / llama.cpp) and strong accuracy claims; **publication text should match what is measured in code** (HF checkpoint + PyTorch inference path).

---

## 2. Themes That *Sound* Novel vs. What Literature Already Covers

### 2.1 Mixed precision under a memory budget

**Idea:** Assign different bit-widths per layer to hit a size target while protecting important layers.

**Prior art (examples from public search, 2024–2026):**

- **IMPQ — Interaction-Aware Layerwise Mixed Precision Quantization for LLMs** — frames layer cooperation; Shapley-style estimation and optimization for very low bit assignments.  
  https://arxiv.org/html/2509.15455v1  

- **MixLLM** (Dec 2024 context in surveys) — mixed precision along different axes than naive per-layer only schemes.

- **SFMP** (search hits reference a “search-free” fractional / block-wise precision direction) — reduces search cost for assignment.

**Implication for AutoQuant:** A paper that only says “we mixed INT4/8/16 by layer under a budget” **overlaps heavily** with existing mixed-precision LLM quantization lines. Novelty must come from **how** scores are formed, **how** the discrete problem is solved, **provable** properties, or **measured** wins vs. strong baselines at matched size.

### 2.2 Fisher / gradient-based sensitivity

**Idea:** Use gradient structure to rank where quantization hurts the loss.

**Prior art (examples):**

- **FGMP: Fine-Grained Mixed-Precision with Fisher Information** (public listing: Apr 2025) — Fisher-weighted **block** / fine-grained decisions, energy and memory claims on Llama-class models.  
  https://arxiv.org/pdf/2504.14152  

- **LQ-LoRA** (ICLR 2024 line) — Fisher-related approximations for **data-aware** low-bit / adapter settings (different problem, same “Fisher as signal” family).  
  https://arxiv.org/pdf/2311.12023  

**AutoQuant’s specific score** (clipped grad² × \|act\|, layerwise, then knapsack-style greedy) is a **reasonable engineering composite** but is **not automatically “new theory.”** A publication would need to **motivate** this combination vs. diagonal Fisher / Hutchinson Hessian trace / blockwise FGMP / interaction-aware IMPQ, with **ablations** (remove clipping, remove activation factor, per-block vs per-layer, etc.).

### 2.3 Weight-only PTQ vs. activation-aware or W+A quantization

**Mainstream PTQ families** (surveys and courses commonly contrast):

- **GPTQ** — approximate second-order, weight-only, widely used.  
- **AWQ** — protects salient weights using activation statistics; weight-only.  
- **SmoothQuant** — migrates difficulty between weights and activations for **W8A8**-style paths.

Example comparison write-up:  
https://www.youngju.dev/blog/llm/2026-03-06-llm-quantization-gptq-awq-gguf-comparison.en  

**Implication:** AutoQuant’s `DynamicQuantizedLinear` is **weight-only PTQ with group scales + a narrow FP16 residual**. That is **orthogonal** to GPTQ’s error-optimal rounding loop and to AWQ’s salience scaling. **Novelty is not “we quantized weights”** — it is whether **this allocator + this kernel representation** beats **GPTQ/AWQ at the same compressed size** on standard metrics.

---

## 3. Candidate “Novelty Angles” That Could *Become* Publication-Worthy

These are **hypotheses tied to this repo’s design**. Each needs **controlled experiments** before claiming in a submission.

1. **Sensitivity cliff + two-phase greedy knapsack**  
   The **data-driven cliff** on sorted sensitivity gaps (`_sensitivity_cliff_threshold`) to lock a tail at INT4, combined with **INT8-first** expansion and **gated INT16** upgrades, is a **clear, describable policy**. A paper could position it as a **simple, interpretable** alternative to heavier discrete optimizers — *if* it matches or beats them under equal compute and size budgets.

2. **Gradient clipping at the 99th percentile × activation magnitude**  
   Stabilizing Fisher-style signals under outlier gradients and coupling to activation scale is **defensible** as a robust sensitivity estimate. Again: only novel with **ablations** and **comparison** to unclipped Fisher, activation-only metrics, and blockwise methods (FGMP family).

3. **Per-row FP16 residual after grouped symmetric quantization**  
   The **narrow residual** (mean error per output channel) is a lightweight **error compensation** mechanism without full FP16 weights. Worth comparing to **GPTQ**’s adaptive rounding and to **zero-shot** baselines at the same bit allocation.

4. **End-to-end “budget-first” UX + reproducible HF artifacts**  
   As a **systems + method** paper: memory-budget-first workflow, meta footprint estimation, single-flight serving, and a **documented checkpoint layout** can be valuable for **MLSys / demo** tracks — still require **benchmark tables** and honest limitation discussion (CPU/GPU kernels, no custom CUDA GEMM in-repo, etc.).

5. **Diversified calibration corpus baked into sensitivity**  
   The breadth of `CALIBRATION_TEXT` (code, SQL, multilingual) supports claims about **domain-robust profiling** — measurable via sensitivity stability across held-out text domains and final perplexity variance.

---

## 4. Gaps to Close Before Claiming “Completely Novel”

| Gap | Why reviewers care |
|-----|-------------------|
| **No large-scale perplexity / LM Harness / Open LLM Leaderboard numbers in-repo** | README already flags smoke-level eval; a paper needs **standard LM metrics** at matched model size. |
| **No head-to-head vs. GPTQ / AWQ / bitsandbytes nf4** | Without this, “better” is unverifiable. |
| **Layerwise {4,8,16} vs. sub-layer / 2-bit / NF4** | Competitive space includes **sub-4-bit** methods; claims must specify the **regime**. |
| **Activation quantization** | Not the focus of current layers; do not claim W8A8-style results without measuring activations. |
| **Hardware story** | Speedup claims need **measured latency/throughput** on stated devices (and kernel details). |
| **Theoretical contribution** | Current code is **heuristic**; theory would need bounds, optimality gaps, or equivalence to a known principled objective. |

---

## 5. Suggested Paper Framing (Honest Options)

**Option A — Algorithms track:** Position the **cliff + gated INT16 + robustified Fisher proxy** as the contribution; compare against **uniform PTQ**, **manual mixed precision**, **IMPQ/FGMP-style** or reimplemented greedy baselines; full WikiText/C4 tables + ablations.

**Option B — Systems / tool track:** Position **budget-constrained AutoQuant with HF export and operational API** as the contribution; include **user study** or **deployment metrics** (time to compress, disk, failure modes) plus **acceptable accuracy** at budget (still need standard eval).

**Option C — Negative result / analysis:** Rigorous study showing when **simple greedy + Fisher proxy** matches expensive search — valuable if executed with unusual thoroughness.

---

## 6. Related Work URLs (Starting Bibliography)

Use these as seeds; extend with Google Scholar / ACL Anthology for your exact venue.

| Topic | URL |
|-------|-----|
| IMPQ (interaction-aware layerwise mixed precision) | https://arxiv.org/html/2509.15455v1 |
| FGMP (Fisher, fine-grained mixed precision, 2025 listing) | https://arxiv.org/pdf/2504.14152 |
| LQ-LoRA (Fisher in data-aware quant / low-bit context) | https://arxiv.org/pdf/2311.12023 |
| LLM PTQ comparison (practitioner narrative) | https://www.youngju.dev/blog/llm/2026-03-06-llm-quantization-gptq-awq-gguf-comparison.en |

---

## 7. Bottom Line

- **Strength of this project as it exists:** A **coherent, end-to-end** pipeline from **sensitivity → discrete bit plan under GB budget → custom quantized modules → saved HF-style artifacts**, with **thoughtful engineering** (cliff lock, gated INT16, gradient clipping, activation weighting, group scales, residuals, dashboard/API).

- **Not automatically novel for a top ML venue** without new **measured** results: **mixed precision**, **Fisher/gradient sensitivity**, and **weight-only PTQ** are all **crowded**; recent papers (IMPQ, FGMP, etc.) already push **interaction-aware** or **fine-grained Fisher** angles.

- **Path to publication:** Pick one **tight claim** the code can support, run **strict baselines** at **matched compression**, report **full benchmarks + ablations**, and align all marketing/guide text with **what is actually evaluated**.

---

*Document generated to support publication planning. Update the “Related Work” table as you finalize venue-specific citations.*
