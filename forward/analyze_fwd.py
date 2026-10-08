import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, "..\\..")

issues = []
with open("issues.jsonl") as f:
    for line in f:
        try:
            issues.append(json.loads(line.strip()))
        except Exception:
            continue

par_modele = defaultdict(list)
for d in issues:
    if d.get("decision") == "HOLD":
        continue
    par_modele[d["model"]].append(d)

baseline_rows = par_modele.get("BaselineUP", [])
n_b = len(baseline_rows)
up_b = sum(1 for r in baseline_rows if r["win"])
base_pct = (up_b / n_b * 100) if n_b else 0

print(f"Total issues: {len(issues)}")
print(f"BaselineUP: {n_b} decisions, {up_b}/{n_b} = {base_pct:.1f}%")
print()
header = f"{'Modele':<20} {'n':>5} {'WR':>8} {'Edge':>8} {'p-val':>8} {'PnL net':>10} {'Verdict':>12}"
print(header)
print("-" * 85)

from scipy.stats import binomtest

for nom in sorted(par_modele.keys()):
    if nom == "BaselineUP":
        continue
    rows = par_modele[nom]
    n = len(rows)
    wins = sum(1 for r in rows if r["win"])
    wr = wins / n if n else 0
    edge = (wr - base_pct / 100) * 100
    p_val = binomtest(wins, n, 0.5).pvalue if n > 0 else 1.0
    pnl = sum(r["pnl_net"] for r in rows)
    sig = (p_val < 0.05 and abs(edge) > 0.5)
    verdict = "VERT_EDGE" if sig else ("drift" if abs(edge) < 0.5 else "ROUGE_drift")
    print(f"{nom:<20} {n:>5} {wr*100:>7.1f}% {edge:>+7.1f}pp {p_val:>8.4f} {pnl:>+10.6f} {verdict:>12}")

t0 = json.load(open("checkpoint.json")).get("t0_forward", 0)
elapsed = (time.time() - t0) / 60 if t0 else 0
print(f"\nElapsed: {elapsed:.1f} min / 240 min")
print(f"Remaining: {240 - elapsed:.1f} min")

n_dec = json.load(open("checkpoint.json")).get("n_decisions", 0)
n_iss = json.load(open("checkpoint.json")).get("n_issues", 0)
pending = json.load(open("checkpoint.json")).get("pending", 0)
print(f"n_decisions={n_dec}, n_issues={n_iss}, pending={pending}")
