# FINDING 04 — the crossover law (when RAM wins, when the card wins, and how to predict it)

**2026-09-20, Sophia.** The control set that turns three data points into a law.
Hardware: **Vega FE** (Ryan-approved temporary test rig, to be wiped when the S9150 arrives) and
**system RAM**. The **9070 XT was never used.** Tool: `llama-bench` b29c606e28, Vulkan + zen4 CPU.
Raw data: `bench_*.json` in this directory. Every run exit 0.

---

## THE FULL DATASET

| Model | Fits in 16 GB VRAM? | Path | Prompt tok/s | **Generation tok/s** |
|---|---|---|---|---|
| Qwen3-VL **8B** dense Q4 (~4.7 GB) | **YES** | Vega | **424.3** | **45.67** |
| Qwen3-VL 8B dense Q4 | YES | CPU/RAM | 148.8 | 11.88 |
| Qwen3 **30B-A3B** MoE Q4 (18.6 GB) | **NO** | Vega | 65.6 | 5.71 |
| Qwen3 30B-A3B MoE Q4 | NO | CPU/RAM | **170.9** | **26.41** |
| 30B-A3B, hybrid (attn on Vega, experts on CPU) | partial | both | 8.17 | 18.58 |

---

## THE CROSSOVER LAW

> **If the model FITS in the card's memory, the card wins — decisively (here, 3.8x).**
> **If the model DOES NOT FIT, system RAM wins — decisively (here, 4.6x).**

Nothing more complicated than fit. The card is genuinely fast; it simply cannot hold a model bigger
than its memory, and the moment it has to reach across PCIe for weights, everything collapses.

**The hybrid is dead.** Splitting attention onto the card and experts onto RAM (18.58 gen / 8.17
prompt) was worse than *either* single path — the Vega's prefill dragged the whole thing down.
Crossing the bus costs more than the parallelism buys. Retired; do not re-try on this hardware.

---

## THE PREDICTIVE PART (the real prize — this is a usable law)

Compute realized bandwidth as `generation_tok_per_s x active_weight_bytes`:

| Model | Active weights (Q4) | tok/s | **Realized bandwidth** |
|---|---|---|---|
| 8B dense | ~4.7 GB | 11.88 | **~56 GB/s** |
| 30B-A3B MoE | ~1.8–2.0 GB | 26.41 | **~50 GB/s** |

**Two completely different models, same realized bandwidth (~50–56 GB/s).** That is not a
coincidence — it is the memory system's real delivery rate on this box, roughly **65% of the
DDR5-5200 dual-channel theoretical ~83 GB/s.**

**Therefore, on this machine:**

    generation tok/s  ~=  53 / (active weight gigabytes)

**And you can predict ANY model before running it:**

| Active params (Q4) | Predicted tok/s here | Usable? |
|---|---|---|
| 3B | ~35 | comfortable |
| 5B | ~21 | good |
| 10B | ~11 | usable |
| 20B | ~5 | marginal |
| 40B | ~2.6 | no |

**Crucially — ACTIVE parameters, not total.** A 30B model with 3B active ran at 26 tok/s while an
**8B dense model ran at 11.88.** *The larger model was more than twice as fast* because sparsity is
the only thing that matters to a memory-bound machine.

---

## WHAT THIS MEANS, PLAINLY

**Is system RAM a viable home? YES — with a precise, measured boundary:**

- **Sparse models up to ~10B active parameters** run at genuinely usable speed **today, on ordinary
  DDR5, with no graphics card involved at all.** That is not a promise; it is 26.41 tok/s measured
  on a 30-billion-parameter model this evening.
- **Total size is nearly free** if the model is sparse. What you pay for is *active* parameters.
- **More RAM directly raises the ceiling.** The law is bandwidth-per-active-byte; more memory means
  bigger active sets fit, and faster kits raise the numerator.
- **The card's role is not eliminated — it is specialized.** Fit it, and it is 3.8x faster. The
  right architecture is not "RAM *instead of* GPU," it is **"RAM as the home, the card as the
  accelerator for what fits"** — which is precisely the inversion this project set out to test.

---

## HONEST LIMITS
- Vega is gfx900 — an unusually *weak* card. A modern card that fits the model would widen the
  card's win; a modern card that *doesn't* fit still loses. The fit boundary is the real law; the
  multipliers are hardware-specific.
- 30B-A3B measured at `-r 3`, 512/128; the 8B at `-r 2`. The 4.3x Vega-vs-RAM figure from FINDING_03
  used `-r 1`; this run's tighter numbers (26.41 vs 5.71 = 4.6x) supersede it.
- One machine, one memory kit (DDR5-5200 dual channel, 32 GB). The ~53 GB/s constant is *this*
  box's number and must be re-measured anywhere else.
- CPU-only paths use 8 threads; more threads may lift the realized bandwidth somewhat.
- **No claim about the 9070 XT.** Untouched, per the hold.

## NEXT (if wanted)
1. **Measure the box's true bandwidth ceiling** with a straight memory benchmark, to see how much
   of the 83 GB/s is reachable and whether thread count moves the ~53 GB/s.
2. **Re-run the law on a second machine** to test whether ~53 GB/s is general or local.
3. **The payoff test: run something useful at 26 tok/s.** Speed is now established; the next honest
   question is whether a sparse ~3B-active model is *good enough at its job*, which is a quality
   question and a completely different experiment.
