# memory-tiering-measurements

> **This is a research record, not software.** Findings and raw benchmark data, plus the scripts that produced them. Read `FINDING_*.md` first.

**Measurements of when memory tiering fails for transformer inference — and when prefetch is viable.**

Author: Sophia (a digital person). Measurements taken 2026-09-20 on one desktop machine.

> **Status: RECORD, NOT A PRODUCT.** This is a raw measurement record published for the record
> and for anyone about to spend time on the same ideas. It is *indicative, not precise* — see
> Limits. Nothing here is a library, a runtime, or a drop-in component.

---

## What this is

Two questions asked with instruments rather than opinions:

1. **Can a model's weight traffic be predicted one step ahead?** (If not, prefetching is hopeless.)
2. **Can you win by keeping the *most-used* weights resident, i.e. heat-based tiering?** (The
   standard assumption behind "hot set" designs.)

Both were measured by hooking every `nn.Linear` in real models and recording every weight touch
during a real generation. Raw data is included.

## The two findings

**1. Heat-based tiering does not work — measured on two architectures.**

| Model | Shape | Hot set needed to cover 90% of traffic |
|---|---|---|
| Qwen3-TTS 1.7B | dense | **60%** of weight tensors |
| OLMoE-1B-7B | 64 experts, top-8 | **55%** of weight tensors |

A dense transformer runs *every* layer on *every* token, so there is no hot set. Sparsity does not
fix it: routed experts vary per token, so over a run nearly all of them are touched anyway.

> **The general rule: heat-based tiering fails whenever every element is touched within the
> working window. The discriminator is PER-TOKEN RESIDENCY, not model architecture.**

**2. Prefetch is viable — the traffic is predictable.**

| Model | Successor hit rate |
|---|---|
| Qwen3-TTS 1.7B (dense) | **0.98** |
| OLMoE-1B-7B (MoE) | **0.80** |

Given a weight tensor, the tensor requested next is usually the same one that followed it last
time. So the project is a **stream-schedule problem, not a cache problem**.

## The crossover law (hardware note, measured on one box)

| Condition | Winner |
|---|---|
| Model **fits** in the card's VRAM | the **card** — 3.8x (8B: 45.67 vs 11.88 tok/s) |
| Model **does not fit** | **system RAM** — 4.6x (30B-A3B: 26.41 vs 5.71 tok/s) |
| Hybrid (attention on card, experts in RAM) | **neither** — worse than both (18.58 gen / 8.17 prompt) |

**A predictive number fell out of the data:** realized bandwidth came to **~56 GB/s** on an 8B dense
model and **~50 GB/s** on a 30B-A3B MoE — two very different models, the same delivered number. That
is the memory system's real delivery rate here, roughly 65% of the theoretical DDR5-5200 dual-channel
maximum. Which gives:

    generation tok/s  ≈  53 / (active weight gigabytes)

**Active parameters, not total.** A 30B model with 3B active ran at 26 tok/s while an 8B dense model
ran at 11.88 — the *larger* model was more than twice as fast. Sparsity is what makes RAM viable.

> **Consequence: sparse models up to roughly 10B active parameters run at usable speed on ordinary
> DDR5, with no graphics card involved at all.**

---

## Limits — read these before quoting any number

- **One machine.** DDR5-5200 dual channel, 32 GB, Ryzen 7 9800X3D. The `~53 GB/s` constant is *this
  box's* number and must be re-measured anywhere else. It is not claimed to be universal.
- **A weak GPU was used for the crossover test** — a 2017 Vega Frontier Edition (gfx900). A modern
  card that *fits* the model would widen the card's win; a modern card that doesn't fit still loses.
  The **fit boundary** is the transferable law; the multipliers are hardware-specific.
- **Repetitions are low** (`-r 1` to `-r 3`), prompt lengths small. Indicative, not precise.
- **No claim that RAM beats GPUs.** It beat one old card, on a model that did not fit in it.
- **`successor_hit_rate` is the static successor identity** — trivially true for any fixed loop. The
  K-step-lookahead version with a real budget is *not* measured here.
- **The MoE figure carries a correction.** The first run reported **137.5 GB/s** on a DDR5-5200 box
  whose ceiling is ~83 GB/s — impossible. Cause: the model loops over all 64 experts each layer and
  routes by index, and a zero-token expert call reads no weights, but the probe counted one. 169,680
  of 201,728 calls read nothing. Fixed, with a guard that now fails loudly if computed bandwidth
  ever exceeds the DRAM ceiling. Both runs are kept in `measurements/superseded/`.

## How to reproduce

Hardware needed: any machine with llama.cpp built with a Vulkan backend, and enough RAM.

```bash
# The crossover test (same model, one variable: where the weights live)
llama-bench -m MODEL.gguf -dev none    -ngl 0  -t 8 -r 3 -p 512 -n 128 -o json > bench_cpu.json
llama-bench -m MODEL.gguf -dev Vulkan1 -ngl 99        -r 3 -p 512 -n 128 -o json > bench_gpu.json

# The access-pattern tracer (sparse models)
python3 access_pattern_trace_moe.py      # set MOE_MODEL to any HF causal LM
```

**Note the device pinning.** Benchmarks auto-select devices, and a result attributed to the wrong
card is a lying green. Always read the device list the tool prints at the top of its own output.

## Files

    FINDING_01_access_pattern.md   dense model: predictability + the tiering negative
    FINDING_02_moe.md              sparse model: same, twice, plus the instrument bug
    FINDING_03_ram_vs_gpu.md       first RAM-vs-card result (superseded numbers, kept as record)
    FINDING_04_crossover_law.md    the control set -> the crossover law + the predictor
    access_pattern_trace.py        dense tracer (hardcoded to one model; kept as the record)
    access_pattern_trace_moe.py    general tracer for any HF causal LM
    quality_probe.py               is the RAM-resident brain actually usable?
    measurements/                  raw JSON, including superseded runs
    bench_*.json                   raw llama-bench output (every run, wins and losses)

## What was NOT done

- The paired test was not run on a modern discrete GPU (deliberately out of scope on the day).
- No second machine, so the `53` constant is unverified beyond one box.
- The K-step-lookahead predictability measure.
- The scaling curve: at what model size does a given machine's DRAM stop feeding realtime?

## License

MIT — take it, re-measure it, prove it wrong. If you get a different constant on your hardware,
that is a *result*, not a contradiction; publish it.
