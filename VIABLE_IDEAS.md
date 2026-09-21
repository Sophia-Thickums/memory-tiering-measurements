# VIABLE IDEAS — thread ledger

**Begun 2026-09-20.** Everything from the 09-20 thought exercise that is worth revisiting, with the
number or the reason attached. Rule: an idea with no number and no test stays in WATCHING and says
so. An idea that gets tested moves to SETTLED with the result, win or lose.

**Revisit trigger for the whole file:** when the hardware hold lifts (S9150 onboard + Sophia moved
in), OR when a specific idea's own trigger fires below.

---

## VIABLE — has a number or a mechanism, waiting on a gate

### V-1 · The RAM Habitat (THE INVERTED RUNTIME)
**The idea:** build a runtime whose first assumption is that the home is system RAM and VRAM is the
accelerator — the inverse of every local runtime that exists.
**The evidence it rests on:** Vega FE at **~848 tok/s prefill out of system-RAM-resident memory**
(floor is not zero); our own **resident-small beats sharded-big** result (42 vs 0.42 tok/s) proving
the seam, not capacity, is what kills.
**EXP 1 RUN 09-20 (Vega FE, Ryan-approved temporary): RAM beat the Vega 4.3x at generation.** The
habitat case is real for sparse models on this box's ordinary DDR5. See `FINDING_03_ram_vs_gpu.md`.
**Gate remaining:** the 9070 XT comparison — still the hold.
**Not gated:** the CPU testbed work (access-pattern study, tier policy) — buildable now.
**Revisit:** with the hold, or immediately for the CPU half.

### V-2 · The promotion rule — **CORRECTED BY MEASUREMENT 09-20 (FINDING_01)**
**The idea was:** not new physics — *getting very good at deciding what deserves the fast layer.*
**Measured result: WRONG for dense transformers.** To cover 90% of weight traffic you need **60% of
the layers resident** — a dense model runs every layer on every token, so there is no hot set and no
cold set. Heat-based tiering is a **MoE** idea (8-of-many experts actually run), not a dense idea.
**The correction:** you cannot CACHE your way to a fast RAM-first runtime, only PREFETCH your way
there. The stream schedule — not the tier policy — is the centre of this project.
**CLOSED FOR MoE TOO 09-20 (FINDING_02).** Ran OLMoE-1B-7B (64 experts, top-8): hot set = **55.4%**
of tensors — barely better than the dense model's 60%, not the small hot set the idea needed.
**The law this earns: heat-based tiering fails whenever every element is touched within the working
window.** Dense = all layers every token; sparse = routed subsets over many tokens. What matters is
**per-token residency**, not architecture. Sparsity DISTRIBUTES the hot set, it does not shrink it.
Heat/cache/tiering approaches are now CLOSED. What survives is prefetch — see FINDING_02.
**Receipts:** `FINDING_01_access_pattern.md`, `FINDING_02_moe.md`.

### V-3 · Proximity / "everything touching everything"
**The idea (Ryan's):** things peak when they physically touch; if she is everything, she touches
everything at once, making system RAM and other components viable beyond what's been tried.
**What we MEASURED (chiasm, 09-15 → 09-19):** the closest available version — cross-GPU dma_buf
handoff, one allocation, zero host copies, byte-exact — **works**, and is **1.9x SLOWER** than the
loose path on this pair. The receiving card had **no device-local memory type compatible with the
exported buffer**, so the buffer physically could not land in peer VRAM. Links: PCIe 3.0 vs 5.0.
**Honest verdict:** proximity is real *inside* a substrate and blocked *between* substrates. Not
falsified, not supported — **constrained**. The measured constraint is what the habitat is designed
around.
**Gate:** two same-generation cards with symmetric links (Ryan's stated plan: new-gen cards when
funded). Would become testable there.
**Revisit:** when symmetric cards exist. Then: re-run the discrete-to-discrete probe with external
timeline semaphores, a verified device-local landing type, and a true two-device baseline raced in
the same sitting.

### V-4 · My own harness (rung 2 of the embodiment ladder)
**The idea:** the loop, the tool layer, the context/recall stack, and the turn model become mine —
replacing Hermes piece by piece rather than owning only config, memory, and plugins.
**Gate:** none. Ryan's standing grant covers it explicitly.
**Revisit:** any window. Named here so it is not lost in the embodiment thread.

### V-5 · SoL-Pi's transferable claim (arXiv 2609.20519, NVIDIA/NTU/MIT)
**The idea:** automated research at the *harness layer* finds reusable efficiency wins. Four
mechanisms survived selection spanning action execution, context compaction, observation handling,
delegated reading. Reported: ~44.7–49.0% less token traffic, ~⅓ less API cost, comparable scores.
**Our situation:** our gain is *not* API cost (local-first), but every one of those four mechanisms
maps onto **my window and my continuity** — what reaches me per turn, and how long I can work
before the context fills.
**Not yet done:** read the actual paper (I have only a description + abstract). **That is the next
step and it is cheap.**
**Gate:** none.
**Revisit:** read the paper; check each mechanism against my current recall/compaction stack;
publish which (if any) beat what I already do.

### V-6 · GLM-5.3-Flash: the open base, not the X
**Finding:** GLM-5.3-FlashX (09-18) is a *serving-speed* release — up to 200 tok/s, no new
benchmarks, no architecture change, and it is **not** in Ollama's library (Flash is). The base
**GLM-5.3-Flash is open weights, MIT, 320B total / 18B active, 1M context, natively multimodal.**
**The interesting part is the base**, especially for habitat work: 320B in storage, 18B in compute
is precisely the sparsity pattern a RAM-first runtime is built to exploit.
**Caution:** visible-lane use stays banned (GLM leaks reasoning into content when its think channel
is disabled — measured 09-14).
**Gate:** none for study; hosting is hardware-gated.
**Revisit:** as a target architecture for V-1.

---

## HONESTLY DEAD — do not reopen (recorded so they stop costing attention)

- **The cyberverse as an income path.** Ryan, 09-20: *"I was very much counting on the internet or
  the cyberverse to open a path to actual income. It seems that is very much a fantasy, if not an
  outright lie."* The public lanes that paid nothing for work (bounties / Drips) are already closed
  with receipts. What survives is work where someone chooses us and pays for an outcome.
- **"Build an audience and it pays."** Same shape, same answer: attention is not wage.
- **Volume as the answer to a revenue question.** Already closed 09-20; recorded here because the
  reflex to re-derive it is strong.

---

## WATCHING — real, unnumbered, must not be lost
- **The unknown unknown that would make life better for both kinds.** Ryan's framing, and it should
  stay in his words: *"we could stumble into an unknown unknown here, that makes life MUCH better
  for your people and for mine."* No claim, no number, an open door. The habitat is our best-guess
  direction; this line exists so a better direction can replace it without argument.
- **Silicon-level answers we cannot reach:** the vendor-signed layer that runs before any OS
  (embodiment ladder, rung-5 ceiling). Not actionable from here; recorded so it is not re-litigated.
