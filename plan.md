# Soup — Roadmap

**Repo:** https://github.com/MakazhanAlpamys/Soup
**PyPI:** https://pypi.org/project/soup-cli/ (`pip install soup-cli`)
**Version:** v0.10.1 | 624 tests | CI green

### How to publish

```bash
# 1. Bump version in pyproject.toml + soup_cli/__init__.py
# 2. Tag and push
git tag v0.X.0
git push --tags
# GitHub Actions auto-publishes to PyPI
```

---

## Completed (v0.1.0 – v0.10.1)

- **CLI:** init, train, chat, push, merge, export, eval, serve, sweep, diff, doctor, quickstart, ui, version
- **Data:** inspect, validate, convert, merge, dedup, stats, generate
- **Training:** SFT, DPO, GRPO, PPO/RLHF, Reward Model, LoRA/QLoRA, QAT, auto batch size, resume, DeepSpeed, W&B, Unsloth backend
- **Multimodal:** `modality: vision`, LLaVA/ShareGPT4V, LLaMA-Vision/Qwen2-VL/Pixtral
- **Serving:** OpenAI-compatible API, transformers + vLLM backends, SSE streaming, tensor parallelism
- **Tracking:** SQLite experiment tracker, runs list/show/compare/delete
- **Export:** GGUF (Ollama/llama.cpp), LoRA merge
- **Web UI:** Dashboard, New Training, Data Explorer, Model Chat (FastAPI + SPA)
- **UX:** friendly errors, --verbose, Rich progress bars, confirmation prompts
- **Community:** CONTRIBUTING.md, CODE_OF_CONDUCT.md, SECURITY.md, examples/, FUNDING.yml
- **Tests:** 624 tests, 41 files, ruff lint, CI on Python 3.9/3.11/3.12
- **v0.10.1 bugfixes:** Windows UnicodeEncodeError, PPO trl compat, compute dtype for CPU, diff torch_dtype, wandb version pin

---

## Roadmap

### P0 — Critical (v1.0–v1.1)

#### Community & Marketing
- [ ] Discord server
- [ ] First Reddit post (r/LocalLLaMA, r/MachineLearning)
- [ ] HuggingFace models showcase ("trained-with-soup" tag)
- [ ] 3-5 YouTube tutorials / blog posts with benchmarks (Soup vs LLaMA-Factory vs Axolotl)
- [ ] Supported models page in README (Llama 4, Gemma 3, Qwen 2.5/3, Phi-4, DeepSeek R1/V3, Mistral, Mixtral)

#### Advanced PEFT methods
- [ ] DoRA (`peft_type: dora`) — improved LoRA with magnitude decomposition
- [ ] GaLore — memory-efficient full-parameter training on consumer GPUs
- [ ] LoRA+ — different learning rates for A and B matrices

#### More alignment methods
- [ ] KTO (Kahneman-Tversky Optimization) — `task: kto`, doesn't need paired data
- [ ] ORPO (Odds Ratio Preference Optimization) — `task: orpo`, no reference model needed
- [ ] SimPO — `task: simpo`, simple preference optimization
- [ ] IPO (Identity Preference Optimization) — `task: ipo`

#### Pre-training / Continued pre-training
- [ ] `task: pretrain` — continued pre-training on raw text
- [ ] Support for plain text / tokenized datasets

#### Cloud GPU providers
- [ ] `soup cloud run --provider runpod --gpu a100 --config soup.yaml`
- [ ] RunPod, Vast.ai, Lambda Labs, Modal integration
- [ ] Auto-setup: upload data → rent GPU → train → return adapter → teardown
- [ ] Cost estimator and budget auto-stop

---

### P1 — Important (v1.2–v1.3)

#### MoE support
- [ ] Explicit MoE model support (Qwen3 30B-A3B, Mixtral, DeepSeek V3)
- [ ] ScatterMoE LoRA for efficient MoE fine-tuning
- [ ] Documentation and examples for MoE training

#### Long-context training
- [ ] Sequence parallelism via Ring FlashAttention
- [ ] 128k+ context fine-tuning across multiple GPUs
- [ ] Neat packing (contamination-free) for long sequences

#### Embedding models
- [ ] `task: embedding` — fine-tune sentence transformers, BGE, E5
- [ ] `soup init --template embedding`
- [ ] Contrastive loss, triplet loss support

#### Advanced distributed training
- [ ] FSDP2 support alongside DeepSpeed
- [ ] Multi-node training via torchrun / Ray
- [ ] Tensor Parallelism + Context Parallelism combined

#### Performance optimizations
- [ ] Liger Kernel integration (fused operations)
- [ ] FlashAttention-3/4 auto-detection
- [ ] SageAttention support

---

### P2 — Nice to have (v1.4+)

#### Export formats
- [ ] ONNX export (`soup export --format onnx`)
- [ ] TensorRT-LLM export
- [ ] SGLang backend for serving

#### Integrations
- [ ] MLflow tracking (enterprise alternative to W&B)
- [ ] TensorBoard integration
- [ ] `uv` package manager support

#### Serving improvements
- [ ] Speculative decoding (`soup serve --speculative-decoding`)
- [ ] Batch inference mode (`soup infer --input prompts.jsonl --output results.jsonl`)

#### Data improvements
- [ ] Multi-agent synthetic data generation (GraphGen-style)
- [ ] Quality filters for generated data (perplexity, coherence scoring)
- [ ] Local model as data generation provider (not just OpenAI API)

#### Audio modality
- [ ] `modality: audio` — Qwen2-Audio, Whisper fine-tuning
- [ ] Audio dataset formats and inspection

#### Smart suggestions
- [ ] Auto-detect hardware and recommend optimal config
- [ ] "You have 2×H100 — recommend FSDP2 + Unsloth" style hints

---

## Principles

1. **CLI-first** — everything works from the terminal; UI is a bonus
2. **Zero config by default** — `soup train` with a minimal config just works
3. **Fail fast, fail loud** — bad data or missing GPU = immediate, clear error
4. **Open source core** — CLI is always free; monetize via managed service
5. **Test-driven** — every feature has tests, written alongside the code

---

## Competitive positioning

| Feature | Soup | LLaMA-Factory | Axolotl | Unsloth |
|---|---|---|---|---|
| One-command training | **Yes** | Partial | No | Notebook |
| Web UI | **Yes** | Yes | No | No |
| Experiment tracking | **Built-in SQLite** | W&B only | W&B only | No |
| Hyperparam sweep | **Yes + early-stop** | No | No | No |
| Data toolkit (7 tools) | **Yes** | Basic | No | No |
| Model diff | **Yes** | No | No | No |
| GGUF export | **Yes** | Yes | No | Yes |
| GRPO + custom rewards | **Yes** | Yes | Yes | Yes |
| Full RLHF pipeline | **Yes** | Yes | Yes | No |
| Cloud GPU | No | SageMaker | RunPod templates | No |
| MoE training | No | Yes | Yes | **12x faster** |
| DoRA/GaLore | No | Yes | Yes | Partial |
| KTO/ORPO/SimPO | No | Yes | Yes | No |
| Pre-training | No | Yes | Yes | Yes |
| 100+ model day-0 | No | **Yes** | Partial | Yes |

**Soup's edge:** best UX, most integrated toolkit, lowest barrier to entry.
**Gap to close:** cloud, advanced PEFT, more alignment methods, model breadth.
