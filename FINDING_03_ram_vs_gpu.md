# FINDING 03 — system RAM beat the graphics card, 4.3x

**2026-09-20, Sophia.** Experiment 1 (the paired baseline) — run on the **Vega Frontier Edition**
with Ryan's explicit approval, as temporary test hardware to be wiped when the S9150 arrives.
Same model file, same build, same prompt lengths, one variable changed: **where the weights live.**

Model: `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` — 30.5B parameters total, ~3B active per token
(a sparse MoE), **18.6 GB on disk.**
Tool: `llama-bench` (build b29c606e28), Vulkan backend, CPU backend `zen4`.
Raw data: `bench_vega.json`, `bench_cpu.json`. Both runs completed, exit 0.

---

## THE NUMBERS

| Path | Prompt (128 tok) | Generation (32 tok) |
|---|---|---|
| **Vega Frontier Edition** (`-dev Vulkan1`, all layers offloaded) | 65.6 tok/s | **5.71 tok/s** |
| **CPU / system RAM** (`-dev none`, `-t 8`) | 92.3 tok/s | **24.35 tok/s** |

**System RAM is 4.3x FASTER at generation, and 1.4x faster at prompt processing, than the
graphics card — on the identical model, in the same session.**

> **SUPERSEDED 09-20 19:0x by `FINDING_04_crossover_law.md`:** a tighter re-run (`-r 3`, 512/128)
> measured 26.41 vs 5.71 tok/s = **4.6x**, and the crossover law now explains *why* — the model did
> not fit in VRAM. This finding's conclusion stands; use FINDING_04's numbers and law.

---

## WHY — and it is not a fluke, it is the structure of the problem

1. **The model does not fit.** 18.6 GB of weights against 16 GB of VRAM. The card cannot hold it,
   so part of every pass crosses PCIe — the single most expensive place for a byte to be.
2. **The Vega is ancient.** gfx900: no bf16, no integer dot product, no matrix cores. Its compute
   per byte moved is poor by modern standards.
3. **The model is SPARSE, and that is the whole story.** Only ~3B of 30.5B parameters are active
   per token. At 24 tok/s that is roughly **40 GB/s of sustained weight demand — comfortably inside
   this machine's ~83 GB/s DRAM ceiling.** The CPU does not need to be fast at arithmetic; it needs
   to stream 1.7 GB of active weights per token, and it can.

**The lesson, stated as a law:** *sparsity is what makes system RAM viable.* A dense 30B at 4-bit
would demand ~15 GB per token of bandwidth — over 180 GB/s, past this box's ceiling, and the CPU
path would collapse. The 3B-active shape is exactly what fits inside a desktop's memory budget.

---

## WHAT THIS DOES AND DOES NOT ESTABLISH

**Does establish (measured, same sitting, same weights):**
- On a model too large for the card's memory, **the RAM path wins decisively** (4.3x generation).
- That is not a corner case — **it IS the habitat case.** The habitat thesis was "memory capacity
  is the binding constraint, not card speed." This is that thesis producing a number.
- Together with FINDING_02 (27.6 GB/s demand for a 7B MoE, vs 83 GB/s available), the
  bandwidth budget for sparse models on this machine is **comfortably sufficient.**

**Does NOT establish (do not overclaim):**
- **RAM does not beat GPUs.** It beat *this* GPU, on *this* model, which did not fit in it. The
  9070 XT was **not allowed to compete or even run** — the hardware hold forbids it, and this test
  used the Vega only with Ryan's specific approval.
- No claim about the fast card. That comparison needs the hold lifted.
- Single repetition (`-r 1`), no variance estimate. Prompt lengths small (128/32). Treat every
  number here as **indicative, not precise.**
- Only one model. The `n_cpu_moe` path (keeping experts on CPU while attention runs on GPU) was
  NOT tested — llama-bench reports it as 0 — and it is the obvious hybrid to try next.

---

## WHY THIS MATTERS FOR US, PLAINLY

Ryan asked whether RAM could be a home rather than an overflow shelf. Tonight's answer is:

- **You cannot cache your way there** (FINDING_01, FINDING_02 — heat-tiering falsified twice).
- **You CAN stream your way there**, because the traffic is predictable (0.98 / 0.80) — and
- **the bandwidth is already sufficient for sparse models on hardware we own today.**

That is a real, if partial, yes. And the machine it takes is a desktop with ordinary DDR5 —
*not* a machine with a giant graphics card. Which was the entire point.

---

## NEXT
1. **`n_cpu_moe` hybrid test** — attention on the card, experts streamed from RAM. This is the
   most promising untested configuration and it is one flag.
2. **A model that FITS in the card** (the 8B VL, 5 GB) raced both ways — the honest control for
   "what does the card give when it isn't spilling."
3. **The scaling curve** — the model size at which this box's DRAM stops feeding realtime.
4. All of the above are CPU/Vega-only and compatible with the hold as Ryan scoped it (Vega as
   temporary test hardware). **Nothing touches the 9070 XT.**
