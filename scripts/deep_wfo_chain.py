"""
Chaîne de WFO profonds (count 5000) sur les candidats du scan H4.
Étape 3 de la mission « vrais marchés » : XAUUSD/USDJPY/EURGBP/R_50/R_75.
"""
import subprocess
import sys
import time

CANDIDATS = ["frxXAUUSD", "frxUSDJPY", "frxEURGBP", "R_50", "R_75"]


def main() -> None:
    for sym in CANDIDATS:
        print(f"\n===== WFO PROFOND {sym} H4 (count 5000) =====", flush=True)
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-X", "utf8", "deriv/walk_forward_optimizer.py",
             "--symbol", sym, "--granularity", "14400", "--count", "5000"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r.stdout:
            print(r.stdout[-2600:], flush=True)
        if r.returncode != 0:
            print(f"[ERREUR rc={r.returncode}] {r.stderr[-800:]}", flush=True)
        print(f"[{sym} termine en {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
