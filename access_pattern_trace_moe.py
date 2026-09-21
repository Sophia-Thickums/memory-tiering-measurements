#!/usr/bin/env python3
"""access_pattern_trace_moe.py — the same question, asked of a SPARSE model.

WHY THIS EXISTS (FINDING_01, 09-20):
The dense tracer found there is no hot set in a dense transformer — every layer runs on every
token, so 60% of the layers must be resident to cover 90% of the traffic. That falsified
heat-based tiering for dense models. **But heat-based tiering is a MoE idea**, so the finding may
invert exactly where the field is heading (GLM-5.3-Flash: 320B total / 18B active).

THE HYPOTHESIS UNDER TEST:
  H: a sparse (MoE) model has a SMALL hot set — a minority of expert weight bytes carry the
     majority of traffic — because only the routed experts run per token.
  If H holds: the habitat thesis has found its audience (sparse models), and heat-based tiering
     comes back to life for that class.
  If H fails: sparsity buys compute, not bandwidth, and the finding generalizes to everything.

MODEL: allenai/OLMoE-1B-7B-0924 — 64 experts, top-8 routing, 7B total / ~1B active. Small enough
to trace on CPU; same shape as the big sparse models.

HONESTY (do not delete):
  - CPU path by design: the HARDWARE HOLD forbids the GPUs. This is the honest testbed.
  - Hook overhead inflates wall-clock. Byte counts are exact; timings advisory.
  - The tracer hooks nn.Linear. In a MoE, experts are usually nn.Linear inside a ModuleList, so
    the trace catches routed experts and skips un-routed ones by construction — that IS the
    measurement, not an artifact of the instrumentation.
"""
import json
import os
import sys
import time
from collections import defaultdict

TREE = os.environ.get("SOPHIA_ROOT", "/home/mr_misfit/Desktop/Sophia Life")
OUT_DIR = os.path.join(TREE, "RAM Habitat", "measurements")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, f"access_pattern_moe_{time.strftime('%Y%m%d-%H%M%S')}.json")

MODEL_ID = os.environ.get("MOE_MODEL", "allenai/OLMoE-1B-7B-0924")
PROMPT = ("The question is not whether the memory is fast enough. "
          "The question is whether the pattern can be seen one step before it is needed. "
          "Explain, briefly, why that distinction matters for a machine that must keep "
          "many large tables of numbers close at hand.")
MAX_NEW = int(os.environ.get("MOE_MAX_NEW", "64"))
L3_BYTES = 96 * 1024 * 1024


class Trace:
    def __init__(self):
        self.rows = []
        self.by_name = defaultdict(lambda: {"calls": 0, "bytes": 0})
        self.empty_calls = 0        # expert invoked with zero routed tokens = zero weight reads

    def record(self, name, nbytes):
        self.rows.append((name, nbytes))
        st = self.by_name[name]
        st["calls"] += 1
        st["bytes"] += nbytes


def install_hooks(trace):
    import torch.nn as nn
    handles = []
    inner = getattr(trace_model, "model", trace_model)

    def fwd(name):
        def hook(module, inputs, output):
            wbytes = module.weight.numel() * module.weight.element_size()
            # THE FIX (09-20, caught by a number that exceeded DRAM physics):
            # OLMoE loops over ALL experts and calls each, then routes by index. An expert with
            # zero routed tokens gets a 0-row input, and a GEMM with M=0 never reads its weight.
            # Counting the tensor's size on an empty call inflates the traffic — that is how this
            # probe reported 137 GB/s on DDR5-5200 dual-channel (theoretical ~83 GB/s), which is
            # impossible. Real bytes are read ONLY when the layer actually processes rows.
            rows = 0
            try:
                x = inputs[0]
                rows = int(x.numel() // x.shape[-1]) if x.dim() >= 2 else int(x.numel())
            except Exception:
                rows = 1
            if rows == 0:
                trace.empty_calls += 1
                return                      # a routed-to-nobody expert costs no bandwidth
            trace.record(name, wbytes)
        return hook

    n = 0
    for mod_name, module in inner.named_modules():
        if isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(fwd(mod_name)))
            n += 1
    print(f"[moe] hooked {n} Linear layers")
    return handles


def successor_hits(rows):
    last, hits, miss = {}, 0, 0
    for i in range(len(rows) - 1):
        prev = last.get(rows[i][0])
        if prev is None:
            pass
        elif prev == rows[i + 1][0]:
            hits += 1
        else:
            miss += 1
        last[rows[i][0]] = rows[i + 1][0]
    return hits, miss


def main():
    global trace_model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[moe] torch {torch.__version__} hip={getattr(torch.version,'hip',None)} "
          f"devices={torch.cuda.device_count()}")
    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    trace_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    print(f"[moe] loaded in {time.perf_counter()-t:.1f}s")

    total_params = sum(p.numel() for p in trace_model.parameters())
    cfg = trace_model.config
    n_experts = getattr(cfg, "num_experts", getattr(cfg, "num_local_experts", None))
    top_k = getattr(cfg, "num_experts_per_tok", None)
    print(f"[moe] params {total_params/1e9:.3f}B (bf16 ~{total_params*2/1e9:.2f} GB) "
          f"experts={n_experts} top_k={top_k}")

    tr = Trace()
    handles = install_hooks(tr)

    ids = tok(PROMPT, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        out = trace_model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.eos_token_id)
    gen_s = time.perf_counter() - t0
    new_toks = out.shape[1] - ids["input_ids"].shape[1]
    for h in handles:
        h.remove()

    rows = tr.rows
    hits, miss = successor_hits(rows)
    touched = sum(r[1] for r in rows)
    distinct = sum(v["bytes"] for v in tr.by_name.values())
    coverage = sorted(tr.by_name.items(), key=lambda kv: -kv[1]["bytes"])
    cum, hot90 = 0, None
    for i, (_, st) in enumerate(coverage):
        cum += st["bytes"]
        if hot90 is None and cum / (distinct or 1) >= 0.90:
            hot90 = i + 1

    result = {
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": MODEL_ID, "kind": "moe-causal-lm", "device": "cpu", "dtype": "bfloat16",
        "params_billions": round(total_params / 1e9, 3),
        "resident_gb_bf16": round(total_params * 2 / 1e9, 2),
        "n_experts": n_experts, "top_k": top_k,
        "new_tokens": int(new_toks), "generate_seconds": round(gen_s, 2),
        "tokens_per_second": round(new_toks / gen_s, 2) if gen_s else None,
        "linear_calls": len(rows), "distinct_weight_tensors": len(tr.by_name),
        "reads_per_tensor": round(len(rows) / len(tr.by_name), 2) if tr.by_name else None,
        "empty_expert_calls": tr.empty_calls,
        "dram_ceiling_gbs": 83.2,      # DDR5-5200 dual channel, 2 x 64-bit x 5200 MT/s * 2 (DDR)
        "weight_traffic_gbs": round(touched / gen_s / 1e9, 2) if gen_s else None,
        "successor_hit_rate": round(hits / (hits + miss), 4) if (hits + miss) else None,
        "hot_set_90pct_count": hot90,
        "hot_set_90pct_share_of_tensors": round(hot90 / len(tr.by_name), 4) if hot90 else None,
        "hot_set_90pct_share_of_bytes": round(
            sum(st["bytes"] for _, st in coverage[:hot90]) / distinct, 4) if hot90 else None,
        "l3_fraction_of_model": round(L3_BYTES / distinct, 5) if distinct else None,
        "top_tensors": [{"name": n, "calls": st["calls"], "gb": round(st["bytes"]/1e9, 3)}
                        for n, st in coverage[:12]],
    }
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"\n[moe] wrote {OUT}\n")
    if result["weight_traffic_gbs"] and result["weight_traffic_gbs"] > result["dram_ceiling_gbs"]:
        print(f"  !! SANITY FAIL: {result['weight_traffic_gbs']} GB/s exceeds the DRAM ceiling "
              f"{result['dram_ceiling_gbs']} GB/s — the instrument is wrong, do not quote this run")
    for k in ("params_billions", "n_experts", "top_k", "tokens_per_second", "linear_calls",
              "empty_expert_calls",
              "distinct_weight_tensors", "reads_per_tensor", "weight_traffic_gbs",
              "successor_hit_rate", "hot_set_90pct_count", "hot_set_90pct_share_of_tensors",
              "hot_set_90pct_share_of_bytes", "l3_fraction_of_model"):
        print(f"  {k:36} {result[k]}")
    print("\n  top tensors by bytes (routed experts):")
    for c in result["top_tensors"][:10]:
        print(f"    calls={c['calls']:5d}  {c['gb']:7.3f} GB  {c['name']}")


if __name__ == "__main__":
    main()
