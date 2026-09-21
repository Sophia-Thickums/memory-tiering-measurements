# FINDING 02 — the sparse model, measured (and the instrument that lied first)

**2026-09-20, Sophia.** Experiment 2b of `RAM_HABITAT.md`, on the CPU testbed under the hardware
hold. Model: **OLMoE-1B-7B-0924** (6.919B total, 64 experts, top-8 routing, 16 layers).
Script: `access_pattern_trace_moe.py`. Data: `measurements/access_pattern_moe_*.json`.
All numbers from a real run on this machine, this sitting.

---

## FIRST, THE BUG — a number above the physics of the machine

The first run reported **137.5 GB/s** of weight traffic. This box is DDR5-5200, dual channel:
theoretical ceiling ≈ **83 GB/s**. The instrument reported a number the hardware cannot produce.
Baseline law, invoked automatically: *a number that contradicts the physics of the hardware it ran
on is an instrument problem until proven otherwise.*

**Cause found.** OLMoE does not gather routed experts — it **loops over all 64 experts every layer
and calls each one**, then routes by index (`modeling_olmoe.py`, `for expert_idx in range(num_experts)`).
An expert with zero routed tokens receives a **0-row input**, and a GEMM with M=0 never reads its
weight. My hook counted a full weight read on every empty call.

    corrected run: 32,048 real expert calls vs 169,680 EMPTY calls (84% of calls read nothing)

**The fix is in the script, and so is the guard:** the tracer now counts rows actually processed and
refuses to record a zero-row call, and the report *fails loudly* if the computed bandwidth ever
exceeds the DRAM ceiling again. A metric that cannot express its own failure is a rubber stamp —
this one now can.

*(Not an artifact of my instrumentation, for the record: looping all experts and routing by index is
OLMoE's own implementation. The wasted calls are real compute the model really does; they just cost
no memory bandwidth.)*

---

## THE CORRECTED RESULT

    model                     OLMoE-1B-7B-0924  (6.919B total, 64 experts, top-8, 16 layers)
    hooked layers             3152 nn.Linear
    real expert calls         32,048        empty calls: 169,680
    weight traffic            27.6 GB/s    (DDR5-5200 ceiling ~83 GB/s — legal)
    successor hit rate        0.80
    hot set for 90% traffic   1647 of 2975 tensors  (55.4%)
    L3 fraction of model      0.00012 (0.01%)

---

## WHAT IT SAYS — and the hypothesis is DEAD

**The hypothesis was: a sparse model has a small hot set, because only the routed experts run.**
**Measured: it does not.** 55% of the weight tensors carry 90% of the traffic — slightly *better*
than the dense model's 60%, and nowhere near the "small hot set" the thesis needed.

**Why it was always going to fail, now that the numbers show it:**
- OLMoE's dense parts — attention Q/K/V/O on all 16 layers, plus the shared always-on experts —
  run on **every token**, unconditionally. They are a fixed floor of hot traffic.
- The routed experts are only **1/8 active**, so sparsity does reduce expert traffic — but the
  per-token traffic is spread across a large, *changing subset*, so as a **set over the whole run**
  it still covers most tensors.
- **Sparsity distributes the hot set; it does not shrink it.**

**The general law this earns (applies to any tiering design):**
> **Heat-based tiering fails whenever every element is touched within the working window.**
> Dense: all layers on every token → 60% hot. Sparse: routed subsets over many tokens → 55% hot.
> The distinction that matters is **per-token residency**, not model architecture.

## What SURVIVED, and it is the load-bearing half

**Predictability held.** Successor hit rate **0.80** for the sparse model (0.98 dense) — lower,
because routing varies per token, but still far above chance. **The prefetch branch is alive for
both architectures.**

And there is a bonus the sparse case *hands* us: **27.6 GB/s sustained versus the dense model's
36.1 GB/s**, at comparable useful work — because 7 of 8 expert calls read nothing. That efficiency
is free, and it is exactly what a memory-bound machine wants.

## THE HONEST STATE OF THE HABITAT AFTER TWO EXPERIMENTS

| Question | Answer |
|---|---|
| Is the access pattern predictable enough to prefetch? | **YES** — 0.98 dense, 0.80 sparse |
| Can we cache/tier our way to speed by heat? | **NO** — 55–60% of tensors needed. Falsified twice, two architectures |
| Is the bandwidth demand reachable by this box's DRAM? | **YES at this scale** — 36.1 / 27.6 GB/s against an ~83 GB/s ceiling |
| Is RAM-first faster than VRAM-first? | **STILL UNTESTED** — needs the cards (hardware hold, exp 1) |

**So the project's centre of gravity has moved twice and is now settled in one place: it is a
STREAM SCHEDULE problem, not a cache problem.** You cannot make the fast layer big enough. You can
only get the next needed bytes into it before they are asked for. Everything else — tiering,
promotion, hot sets — is now measured and closed.

---

## LIMITS (do not omit)
- Two models, two architectures, both small. The per-token-residency law is the transferable claim;
  the exact percentages are not.
- Hook timings remain advisory; byte counts are exact.
- `successor_hit_rate` still measures static successor identity. The K-step-lookahead version with a
  real budget is still unmeasured, in both tracers.

## NEXT
1. **K-step lookahead with a budget** — replaces both hit-rate numbers with the one that a prefetch
   engine can actually use.
2. **The scaling curve** — at what model size does this box's DRAM stop feeding realtime? Turns
   falsifier 3 into a number.
3. **Exp 1 (VRAM vs RAM, same weights)** — still the core falsifier, still gated by the hold.
