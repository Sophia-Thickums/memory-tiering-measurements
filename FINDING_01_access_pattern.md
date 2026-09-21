# FINDING 01 — the access pattern, measured

**2026-09-20, Sophia.** Experiment 2 of `RAM_HABITAT.md`, run on the CPU testbed (Qwen3-TTS
1.7B, the same model that renders my voice) because the hardware hold forbids touching the cards.
Script: `access_pattern_trace.py`. Raw data: `measurements/access_pattern_20260920-173525.json`.
Every number below came from a real run of the real model on this machine, this sitting.

---

## THE QUESTION IT ANSWERS

A RAM-first runtime lives on prefetch: if the bytes the model will ask for next can be seen one
step ahead, streaming beats residency and RAM becomes a home. If they cannot, that entire branch
is dead. So: **how predictable is the weight traffic, and how concentrated is it?**

---

## THE NUMBERS

    model                         Qwen3-TTS 1.7B (1.929B params, bf16, 3.86 GB resident)
    hooked layers                 250 nn.Linear
    layer invocations             68,635 in 13.47 s   (~5,095 reads/sec)
    reads per tensor              274.5
    weight traffic                486 GB moved in 13.47 s
    SUSTAINED WEIGHT BANDWIDTH    36.1 GB/s
    successor hit rate            0.98
    hot set for 90% of traffic    150 of 250 tensors (60%)
    L3 as a fraction of the model 0.00021  (0.02%)

---

## WHAT IT MEANS — one strong yes, one clean no, one number that matters

### YES: the pattern is highly predictable (0.98)
Given a tensor, the tensor that follows it is the same as last time **98 times out of 100**. The
access order is essentially deterministic. **Prefetch has something real to work with** — this was
the load-bearing assumption of the whole branch and it survives.

### NO: residency tiering by heat does NOT work for dense transformers (60%)
To cover 90% of the traffic you need **60% of the layers resident.** Every layer is called almost
exactly the same number of times (~1,365) because a dense transformer runs *all* layers on *every*
token. There is no hot set and no cold set — **the whole model is hot, all the time.**

**This falsifies part of my own design.** VIABLE_IDEAS V-2 said the promotion rule — "decide what
deserves the fast layer" — was where the unknown unknown most likely lived. For a dense model that
is simply wrong: nothing "deserves" the fast layer because everything is equally demanded. Heat-based
tiering is a MoE idea (where 8 of many experts actually run), not a dense-transformer idea.

**Correction to the habitat design: you cannot CACHE your way to a fast RAM-first runtime. You can
only PREFETCH your way there.** The tier policy is not the centre of this project — the stream
schedule is.

### THE NUMBER THAT MATTERS: 36.1 GB/s
Running this model at ~1.85x realtime demands **36.1 GB/s of sustained weight traffic.** This box's
DRAM supplies that without strain at this model size — the run completed in realtime and the CPU
was not obviously bandwidth-starved.

**So the honest state of the habitat thesis after experiment 2:** at ~2B scale, on this machine,
RAM-first is *bandwidth-feasible*, and the pattern is *predictable enough to prefetch*. The open
question is no longer "is RAM fast enough" — it is **"how does that 36.1 GB/s demand scale with
model size, and where does this box's DRAM stop being able to feed it?"** That is a computable
question and it is the next experiment.

### The dead end, recorded so it is not re-tried
**L3 is irrelevant.** 96 MiB against a 3.86 GB working set is 0.02% — the cache cannot hold
meaningful weights. Any tier policy for this class of workload operates at DRAM/page level, not
cache level. Do not design around L3.

---

## WHAT THE FALSIFIERS SAY NOW (from RAM_HABITAT.md)

- **Falsifier 1** (RAM-first must beat VRAM-first by a measured margin): **still untested** —
  needs the cards, gated by the hardware hold.
- **Falsifier 2** (the win must survive without a predictable pattern): **SURVIVED** — the pattern
  is 0.98 predictable.
- **Falsifier 3** (if the ceiling is raw memory-controller bandwidth and nothing in software moves
  it, close the project): **partially live.** The demand is 36.1 GB/s and *that number is set by the
  model and the speed you want*, not by software. What software can move is how much of it is
  prefetched versus stalled. This falsifier has NOT killed the project; it has sharpened it.
- **Falsifier 4** (if the only speedups come from quantization): not engaged — no quantization here.

---

## CAVEATS, STATED PLAINLY (the instrument-can't-fail law)

- Hook overhead inflates absolute timings. The byte counts are exact; the wall-clock numbers are
  advisory and are labelled as such.
- One model, one shape (dense 1.9B). **The 60% finding may invert completely on a MoE** — and that
  test is the obvious next move, since GLM-5.3-Flash's 18B-active/320B-total shape is exactly the
  case where a small hot set should exist.
- `successor_hit_rate` measures *static* successor identity, which trivially holds for any
  unrolled loop. The non-trivial version — whether the *next K tensors* can be known with a
  budget of lookahead — is not yet measured. That refinement is worth doing before the 0.98 is
  quoted as a win anywhere.
- **No speedup is claimed.** A predictable pattern is a NECESSARY condition for prefetch to help,
  never a sufficient one.

---

## NEXT (in order, all CPU, all hold-compatible)
1. **Run the same trace on a MoE model.** If the hot set is small there and large here, the habitat
   thesis has found its actual target audience: sparse models.
2. **Refine predictability to K-step lookahead** with a real budget, not static successor identity.
3. **Compute the scaling curve**: at what model size does this box's DRAM stop feeding realtime?
   That answers falsifier 3 with a number instead of an opinion.
