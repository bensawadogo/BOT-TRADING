"""Télécharge les clôtures journalières FRED utilisées par `lab.run_systeme daily`.

Usage : python scripts/fetch_fred.py data/fred
"""
import sys
import urllib.request
from pathlib import Path

IDS = ["DEXUSEU", "DEXUSUK", "DEXJPUS", "DEXUSAL", "DEXCAUS", "DEXSZUS", "DEXUSNZ",
       "DEXMXUS", "DEXNOUS", "DEXSDUS", "DCOILWTICO", "DCOILBRENTEU", "NASDAQCOM",
       "DHHNGSP", "DGS10"]  # mêmes séries que lab.run_systeme.FRED (+ taux 10 ans)

dossier = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fred")
dossier.mkdir(parents=True, exist_ok=True)
for sid in IDS:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
    (dossier / f"{sid}.csv").write_bytes(urllib.request.urlopen(url, timeout=60).read())
    print(sid, "ok")
