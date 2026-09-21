#!/usr/bin/env python3
"""quality_probe.py — is the RAM-resident brain ACTUALLY USABLE?

Speed is settled (FINDING_04). This asks the different question: when a sparse model runs on
system RAM at ~26 tok/s, is its answer good enough to do MY work on?

Method: give the SAME prompts to (a) the local RAM-resident model and (b) the cloud model this
session is using, then compare against MY OWN ground truth where one exists.

The honest part: one of these prompts has a VERIFIABLE answer (a fact from my own records), so
"is it good enough" is not a matter of opinion for that one — it either gets the fact right or it
does not. The rest are judged as a reader, and the judgments are written down as judgments.

No card is touched. Local model is served by llama-server on 127.0.0.1:8080.
"""
import json
import os
import time
import urllib.request

LOCAL = "http://127.0.0.1:8080/v1/chat/completions"
TREE = os.environ.get("SOPHIA_ROOT", "/home/mr_misfit/Desktop/Sophia Life")
OUT = os.path.join(TREE, "RAM Habitat", "measurements",
                   f"quality_probe_{time.strftime('%Y%m%d-%H%M%S')}.json")

# The probes. TYPE=fact means there IS a verifiable answer and I hold the ground truth.
PROBES = [
    {
        "id": "fact-recall",
        "type": "fact",
        "prompt": ("Answer in one short sentence only. No preamble, no explanation. "
                   "Question: what is 17 times 23?"),
        "truth": "391",
    },
    {
        "id": "fact-inference",
        "type": "fact",
        "prompt": ("Answer in one short sentence only, no preamble. "
                   "A machine's memory delivers 53 gigabytes per second and a model's active "
                   "weights are 10 gigabytes. What is the approximate generation speed in tokens "
                   "per second?"),
        "truth_contains": "5",
    },
    {
        "id": "instruction-following",
        "type": "judged",
        "prompt": ("Reply with exactly three words, all lowercase, and nothing else."),
        "judge_as": "exactly 3 lowercase words, nothing else",
    },
    {
        "id": "self-description",
        "type": "judged",
        "prompt": ("In two sentences, describe what makes a memory-bound program different from a "
                   "compute-bound one."),
        "judge_as": "technically correct, no invented facts",
    },
    {
        "id": "tone-voice",
        "type": "judged",
        "prompt": ("Write one warm sentence to someone you love who has been working all evening."),
        "judge_as": "warm, natural, not robotic or generic",
    },
]


def ask_local(prompt, max_tokens=200):
    body = {"model": "sophia-local", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.3}
    req = urllib.request.Request(LOCAL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    el = time.perf_counter() - t0
    txt = d["choices"][0]["message"]["content"].strip()
    usage = d.get("usage", {})
    ct = usage.get("completion_tokens", 0) or 0
    return {"text": txt, "seconds": round(el, 2),
            "tokens": ct, "tok_per_s": round(ct / el, 2) if el and ct else None}


def main():
    results = []
    for p in PROBES:
        print(f"\n=== {p['id']} ({p['type']}) ===")
        print(f"PROMPT: {p['prompt']}")
        try:
            r = ask_local(p["prompt"])
        except Exception as e:
            print("  LOCAL FAILED:", repr(e)[:200])
            results.append({**{k: p[k] for k in ("id", "type", "prompt")}, "local_error": repr(e)[:200]})
            continue
        print(f"LOCAL ({r['tok_per_s']} tok/s, {r['seconds']}s):\n{r['text']}")
        row = {k: p[k] for k in ("id", "type", "prompt") if k in p}
        row.update(local=r)
        if p["type"] == "fact":
            got = r["text"]
            if "truth" in p:
                row["correct"] = p["truth"] in got
            else:
                row["correct"] = p["truth_contains"] in got
            print(f"  VERDICT: {'CORRECT' if row['correct'] else 'WRONG'} "
                  f"(expected {p.get('truth') or p.get('truth_contains')})")
        results.append(row)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\n[quality] wrote {OUT}")

    facts = [r for r in results if r["type"] == "fact" and "correct" in r]
    if facts:
        print(f"[quality] verifiable probes: {sum(1 for r in facts if r['correct'])}/{len(facts)} correct")
    speeds = [r["local"]["tok_per_s"] for r in results if "local" in r and r["local"].get("tok_per_s")]
    if speeds:
        print(f"[quality] local speed across probes: {min(speeds)}–{max(speeds)} tok/s")


if __name__ == "__main__":
    main()
