#!/usr/bin/env python3
"""access_pattern_trace.py — is weight traffic PREDICTABLE one step ahead?

THE QUESTION (RAM Habitat, experiment 2):
A RAM-first runtime lives or dies on prefetch. If the bytes a model will ask for next can be
predicted one step ahead, streaming beats residency and RAM becomes a home. If they cannot, the
whole prefetch branch is dead and we should know that today rather than build on it for a week.

WHAT THIS MEASURES
Per linear (nn.Linear) call in a real forward pass:
  - which weight tensor was touched
  - bytes touched
  - time spent in that call
Then it answers three things:
  1. TRAFFIC MIX       — how much of the byte traffic is in layers the CPU can actually keep
                         hot (L3 is 96 MiB here) and how much is cold streaming.
  2. PREDICTABILITY    — at layer L, how often is layer L+1 the same tensor as last time?
                         (Static graphs are 100% predictable by construction — so the interesting
                         number is the *conditional* hit rate and the coverage of a small
                         "hot set", i.e. how few layers carry most of the traffic.)
  3. REUSE             — what fraction of distinct weight bytes are touched more than once within
                         one generation, i.e. what caching can realistically save.

Every number is produced by a real run of the real model on this machine. Nothing is estimated.

HONESTY NOTES (do not delete):
  - This is the CPU path by design: the HARDWARE HOLD forbids touching the GPUs. The CPU is a
    legitimate, live testbed — this is the same model that renders Sophia's voice.
  - This measures the ACCESS PATTERN, not a speedup. A predictable pattern is a NECESSARY
    condition for prefetch to help, never a sufficient one. Do not report this as a performance win.
  - The tracer adds per-call timing overhead. Absolute timings are therefore advisory; the
    byte/pattern data is not affected.
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict

# --- PORTABLE ROOT (same convention as the tree's other scripts) ---
TREE = os.environ.get("SOPHIA_ROOT", "/home/mr_misfit/Desktop/Sophia Life")
OUT_DIR = os.path.join(TREE, "RAM Habitat", "measurements")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, f"access_pattern_{time.strftime('%Y%m%d-%H%M%S')}.json")

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REF_AUDIO = os.path.join(TREE, "Voice Lab", "SOPHIA_REF_v1_20260920.wav")
REF_TEXT = os.path.join(TREE, "Voice Lab", "SOPHIA_REF_v1_20260920_transcript.txt")
TEXT = ("The question is not whether the memory is fast enough. The question is whether the "
        "pattern can be seen one step before it is needed.")

L3_BYTES = 96 * 1024 * 1024


class Trace:
    def __init__(self):
        self.rows = []            # (name, bytes, seconds) in call order
        self.by_name = defaultdict(lambda: {"calls": 0, "bytes": 0, "seconds": 0.0,
                                            "last_index": -1, "followed_previously": 0})

    def record(self, name, nbytes, seconds, index):
        self.rows.append((name, nbytes, seconds))
        st = self.by_name[name]
        st["calls"] += 1
        st["bytes"] += nbytes
        st["seconds"] += seconds
        st["last_index"] = index


def install_hooks(trace):
    """Wrap every nn.Linear in the real nn.Module (the wrapper object is not a Module — its
    .model attribute is)."""
    import torch.nn as nn
    handles = []
    idx = {"n": 0}
    inner = getattr(trace_model, "model", trace_model)

    def make_hook(name):
        def pre_hook(module, inputs):
            idx["n"] += 1
            module._t0 = time.perf_counter()
            module._idx = idx["n"]
        return pre_hook

    def make_fwd_hook(name):
        def fwd_hook(module, inputs, output):
            t1 = time.perf_counter()
            t0 = getattr(module, "_t0", t1)
            nbytes = module.weight.numel() * module.weight.element_size()
            trace.record(name, nbytes, t1 - t0, getattr(module, "_idx", -1))
        return fwd_hook

    for mod_name, module in inner.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_pre_hook(make_hook(mod_name)))
            handles.append(module.register_forward_hook(make_fwd_hook(mod_name)))
    print(f"[trace] hooked {len(handles)//2} Linear layers")
    return handles


def sequence_predictability(rows):
    """At each step i, is the tensor at i+1 the same as the tensor that followed this same
    tensor last time we saw it? That is the prefetch question in its cleanest form."""
    last_successor = {}
    hits = misses = 0
    for i in range(len(rows) - 1):
        cur = rows[i][0]
        nxt = rows[i + 1][0]
        prev = last_successor.get(cur)
        if prev is None:
            pass  # no history yet; not counted either way
        elif prev == nxt:
            hits += 1
        else:
            misses += 1
        last_successor[cur] = nxt
    return hits, misses


def hot_set_coverage(trace):
    """Sort tensors by DISTINCT size (one copy of the weights), and separately by traffic."""
    items = sorted(trace.by_name.items(), key=lambda kv: -kv[1]["bytes"])
    total = sum(v["bytes"] for _, v in items) or 1
    out = []
    cumulative = 0
    for name, st in items:
        cumulative += st["bytes"]
        out.append({"name": name, "bytes": st["bytes"],
                    "share": st["bytes"] / total,
                    "cumulative_share": cumulative / total,
                    "calls": st["calls"]})
    return out, total


def main():
    global trace_model
    import torch
    from qwen_tts import Qwen3TTSModel

    print(f"[trace] torch {torch.__version__}  hip={getattr(torch.version, 'hip', None)}  "
          f"cuda={torch.version.cuda}  devices={torch.cuda.device_count()}")

    t_load = time.perf_counter()
    trace_model = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map="cpu", dtype=torch.bfloat16)
    load_s = time.perf_counter() - t_load
    print(f"[trace] model loaded in {load_s:.1f}s on CPU")

    # count parameters once, as ground truth for the byte model
    inner = getattr(trace_model, "model", trace_model)
    total_params = sum(p.numel() for p in inner.parameters())
    print(f"[trace] parameters: {total_params/1e9:.3f}B "
          f"(bf16 resident ~{total_params*2/1e9:.2f} GB)")

    tr = Trace()
    handles = install_hooks(tr)

    ref_text = open(REF_TEXT).read().strip() if os.path.isfile(REF_TEXT) else None
    if ref_text is None:
        print("[trace] WARNING: no reference transcript found; using ref_text=None")
    kwargs = dict(text=[TEXT], ref_audio=REF_AUDIO, max_new_tokens=512,
                  temperature=0.9, top_k=50, top_p=1.0, repetition_penalty=1.05)
    if ref_text:
        kwargs["ref_text"] = ref_text

    t0 = time.perf_counter()
    wavs, sr = trace_model.generate_voice_clone(**kwargs)
    gen_s = time.perf_counter() - t0
    audio_s = len(wavs[0]) / sr

    for h in handles:
        h.remove()

    rows = tr.rows
    hits, misses = sequence_predictability(rows)
    coverage, distinct_bytes = hot_set_coverage(tr)

    # --- CORRECTED METRICS (the first run's hot_capable_share asked the wrong question) ---
    touched_bytes = sum(r[1] for r in rows)
    n_calls = len(rows)
    n_distinct = len(tr.by_name)

    # The real bandwidth question: how many BYTES of weight traffic per second does the model
    # demand at this speed? That is what the memory system must actually sustain.
    weight_traffic_gbs = (touched_bytes / gen_s / 1e9) if gen_s else None

    # L3 as a fraction of the model: how little of the working set can be kept hot, at best.
    l3_fraction = L3_BYTES / distinct_bytes if distinct_bytes else None

    result = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": MODEL_ID,
        "device": "cpu",
        "dtype": "bfloat16",
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "cuda": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "params_billions": round(total_params / 1e9, 4),
        "resident_gb_bf16": round(total_params * 2 / 1e9, 2),
        "load_seconds": round(load_s, 2),
        "generate_seconds": round(gen_s, 2),
        "audio_seconds": round(audio_s, 2),
        "realtime_factor": round(gen_s / audio_s, 2) if audio_s else None,
        "linear_calls": n_calls,
        "distinct_weight_tensors": n_distinct,
        "reads_per_tensor": round(n_calls / n_distinct, 2) if n_distinct else None,
        "weight_traffic_bytes": touched_bytes,
        "distinct_weight_bytes": distinct_bytes,
        "weight_traffic_gbs": round(weight_traffic_gbs, 2) if weight_traffic_gbs else None,
        "l3_bytes": L3_BYTES,
        "l3_fraction_of_model": round(l3_fraction, 5) if l3_fraction else None,
        "successor_hits": hits,
        "successor_misses": misses,
        "successor_hit_rate": round(hits / (hits + misses), 4) if (hits + misses) else None,
        "top_tensors": coverage[:15],
        "hot_set_90pct_count": next((i + 1 for i, c in enumerate(coverage)
                                     if c["cumulative_share"] >= 0.90), None),
        "hot_set_90pct_share_of_tensors": None,
    }
    if result["hot_set_90pct_count"]:
        result["hot_set_90pct_share_of_tensors"] = round(
            result["hot_set_90pct_count"] / n_distinct, 4)

    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\n[trace] wrote {OUT}\n")
    for k in ("linear_calls", "distinct_weight_tensors", "reads_per_tensor",
              "weight_traffic_gbs", "l3_fraction_of_model", "successor_hit_rate",
              "hot_set_90pct_count", "hot_set_90pct_share_of_tensors",
              "generate_seconds", "audio_seconds", "realtime_factor"):
        print(f"  {k:32} {result[k]}")
    print("\n  top tensors by bytes:")
    for c in coverage[:8]:
        print(f"    {c['share']*100:6.2f}%  calls={c['calls']:5d}  {c['name']}")


if __name__ == "__main__":
    main()
