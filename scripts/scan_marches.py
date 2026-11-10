"""Balayage des marchés tradables chez Deriv MT5 : où y a-t-il un avantage de tendance ?

    python scripts/scan_marches.py fetch   → clôtures journalières Yahoo (proxy) → data/yahoo/
    python scripts/scan_marches.py couts   → spread, swap, lot minimum relevés dans MT5 → data/couts_scan.json
    python scripts/scan_marches.py run     → le bot (signaux 1/3/12 mois, vol cible) sur chaque marché,
                                             aux coûts réels, 7 processus → data/scan_marches.csv
    python scripts/scan_marches.py classes → chaque classe en panier, comparée au forex de production
                                             → data/scan_classes.pkl. Conclusions : docs/reports/SCAN_MARCHES_*.md

Rien n'est réglé : on applique tel quel le bot d'origine (E0) à des marchés qu'il n'a jamais vus.
Les clôtures Yahoo servent de proxy historique (les CFD Deriv n'ont que 2 à 8 ans d'historique).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

YAHOO = ROOT / "data" / "yahoo"
COUTS = ROOT / "data" / "couts_scan.json"

# (nom, classe, ticker Yahoo, symbole MT5, heure UTC où mesurer le spread)
CRYPTO = [("BTC", "BTC-USD", "BTCUSD"), ("ETH", "ETH-USD", "ETHUSD"), ("LTC", "LTC-USD", "LTCUSD"),
          ("XRP", "XRP-USD", "XRPUSD"), ("BCH", "BCH-USD", "BCHUSD"), ("ADA", "ADA-USD", "ADAUSD"),
          ("DOGE", "DOGE-USD", "DOGUSD"), ("DOT", "DOT-USD", "DOTUSD"), ("SOL", "SOL-USD", "SOLUSD"),
          ("LINK", "LINK-USD", "LNKUSD"), ("XLM", "XLM-USD", "XLMUSD"), ("XMR", "XMR-USD", "XMRUSD"),
          ("BNB", "BNB-USD", "BNBUSD"), ("TRX", "TRX-USD", "TRXUSD"), ("ETC", "ETC-USD", "ETCUSD"),
          ("UNI", "UNI7083-USD", "UNIUSD"), ("AAVE", "AAVE-USD", "AAVUSD"), ("AVAX", "AVAX-USD", "AVAUSD"),
          ("ZEC", "ZEC-USD", "ZECUSD"), ("DASH", "DASH-USD", "DSHUSD"), ("XTZ", "XTZ-USD", "XTZUSD"),
          ("FIL", "FIL-USD", "FILUSD"), ("NEO", "NEO-USD", "NEOUSD"), ("ALGO", "ALGO-USD", "ALGUSD"),
          ("SHIB", "SHIB-USD", "SHBUSD")]
INDICES = [("SPX", "^GSPC", "US SP 500", 15), ("NDX", "^NDX", "US Tech 100", 15),
           ("DJI", "^DJI", "Wall Street 30", 15), ("RUT", "^RUT", "US Small Cap 2000", 15),
           ("MID", "^MID", "US Mid Cap 400", 15), ("DAX", "^GDAXI", "Germany 40", 9),
           ("FTSE", "^FTSE", "UK 100", 9), ("CAC", "^FCHI", "France 40", 9),
           ("STOXX", "^STOXX50E", "Europe 50", 9), ("IBEX", "^IBEX", "Spain 35", 9),
           ("AEX", "^AEX", "Netherlands 25", 9), ("SMI", "^SSMI", "Swiss 20", 9),
           ("N225", "^N225", "Japan 225", 2), ("HSI", "^HSI", "Hong Kong 50", 2),
           ("ASX", "^AXJO", "Australia 200", 2)]
METAUX = [("XAU", "GC=F", "XAUUSD"), ("XAG", "SI=F", "XAGUSD"), ("XPT", "PL=F", "XPTUSD"),
          ("XPD", "PA=F", "XPDUSD"), ("CUIVRE", "HG=F", "XCUUSD")]
SOFTS = [("CAFE", "KC=F", "CoffeeArab"), ("CACAO", "CC=F", "Cocoa"), ("SUCRE", "SB=F", "Sugar"),
         ("COTON", "CT=F", "Cotton")]
ETFS = [(t, t, f"{t}.US") for t in ("SPY", "IWM", "EEM", "EFA", "VNQ", "GLD", "SLV", "HYG", "LQD", "AGG",
                                    "XLE", "XLF", "XLK", "GDX", "DIA", "VWO", "VEA", "ARKK", "UNG")]
FX = [(p, f"{p}=X", p) for p in ("EURJPY", "GBPJPY", "AUDJPY", "EURGBP", "EURCHF", "EURAUD", "GBPAUD",
                                 "CADJPY", "CHFJPY", "NZDJPY", "AUDNZD", "EURCAD", "USDZAR", "USDSGD",
                                 "USDPLN", "USDTHB")]
ACTIONS_OQ = ("AAPL MSFT AMZN GOOGL NVDA TSLA INTC CSCO PEP COST QCOM AMGN GILD ADBE NFLX").split()
ACTIONS_N = ("JPM BAC C GS KO PG JNJ PFE MRK CVX HD DIS BA CAT MCD IBM GE F MMM ORCL NKE LLY MO COP DE"
             ).split()

UNIVERS: list[tuple[str, str, str, str, int]] = (
    [(n, "crypto", y, m, 8) for n, y, m in CRYPTO]
    + [(n, "indices", y, m, h) for n, y, m, h in INDICES]
    + [(n, "métaux", y, m, 8) for n, y, m in METAUX]
    + [(n, "softs", y, m, 10) for n, y, m in SOFTS]
    + [(n, "ETF", y, m, 15) for n, y, m in ETFS]
    + [(n, "forex croisés", y, m, 8) for n, y, m in FX]
    + [(t, "actions", t, f"{t}.OQ", 15) for t in ACTIONS_OQ]
    + [(t, "actions", t, f"{t}.N", 15) for t in ACTIONS_N]
)


def fetch() -> None:
    import requests
    YAHOO.mkdir(parents=True, exist_ok=True)
    h = {"User-Agent": "Mozilla/5.0"}
    for nom, _, y, _, _ in UNIVERS:
        dest = YAHOO / f"{nom}.csv"
        if dest.exists():
            continue
        try:
            r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{y}",
                             params={"period1": 0, "period2": int(time.time()), "interval": "1d"},
                             headers=h, timeout=30)
            j = r.json()["chart"]["result"][0]
            idx = pd.to_datetime(j["timestamp"], unit="s").normalize()
            s = pd.Series(j["indicators"]["quote"][0]["close"], index=idx, dtype=float).dropna()
            s = s[~s.index.duplicated(keep="last")]
            s.rename("close").to_csv(dest)
            print(f"{nom:8} {y:14} {len(s):6} {s.index[0].date()} -> {s.index[-1].date()}")
        except Exception as e:                                        # noqa: BLE001
            print(f"{nom:8} {y:14} ECHEC : {type(e).__name__} {e}")
        time.sleep(0.3)


def couts() -> None:
    from scripts.couts_mt5 import _jours_recents, _mt5, spread_relatif, swap_annuel
    mt5 = _mt5()
    noms = {s.name for s in mt5.symbols_get()}
    jours = _jours_recents(5, semaine=True)
    out = {}
    print(f"{'marché':8} {'symbole':16} {'½ spread':>9} {'swap L':>8} {'swap C':>8} {'lot min $':>10}  mode")
    for nom, classe, _, sym, heure in UNIVERS:
        if sym not in noms:
            print(f"{nom:8} {sym:16} absent")
            continue
        mt5.symbol_select(sym, True)
        info, tick = mt5.symbol_info(sym), mt5.symbol_info_tick(sym)
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_D1, 0, 1)
        prix = (tick.bid + tick.ask) / 2 if tick and tick.bid > 0 else \
            (float(rates["close"][-1]) if rates is not None and len(rates) else float("nan"))
        lot_value = prix * info.trade_tick_value / info.trade_tick_size
        demi = spread_relatif(mt5, sym, (heure,), jours) / 2
        sl, ss = swap_annuel(mt5, info, lot_value)
        out[nom] = {"classe": classe, "symbole": sym, "cout": demi, "swap_long": sl, "swap_short": ss,
                    "notionnel_min": info.volume_min * lot_value, "swap_mode": int(info.swap_mode)}
        print(f"{nom:8} {sym:16} {demi*1e4:7.2f}bp {sl:8.1%} {ss:8.1%} {info.volume_min*lot_value:10,.0f}  "
              f"{info.swap_mode}")
    COUTS.write_text(json.dumps(out, indent=1))
    print(f"-> {COUTS}")


def charger(depuis: str = "1990") -> pd.DataFrame:
    cols = {}
    for nom, *_ in UNIVERS:
        f = YAHOO / f"{nom}.csv"
        if f.exists():
            x = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0].dropna()
            x = x[x > 0][depuis:]
            if len(x) > 600:
                cols[nom] = x
    return pd.DataFrame(cols).sort_index()


def _alpha(r: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """Rendement annuel et t-stat de la part de `r` non expliquée par le marché (régression)."""
    r, b = r.align(bench, join="inner")
    b = b.fillna(0.0)
    X = np.column_stack([np.ones(len(b)), b.to_numpy()])
    coef, *_ = np.linalg.lstsq(X, r.to_numpy(), rcond=None)
    res = r.to_numpy() - X @ coef
    se = np.sqrt(res.var(ddof=2) * np.linalg.inv(X.T @ X)[0, 0])
    return float(coef[0] * 252), float(coef[0] / se)


def _un_marche(args):
    """Un marché seul, avec le bot d'origine : (ligne de résultats, rendements) ou None."""
    from trend_bot.backtest import Options, backtest, stats
    from trend_bot.signals import daily_returns, weekdays_only
    nom, classe, s, k = args
    base = dict(closes=s, opt=Options(regime=False))
    rA = backtest(costs={nom: k["cout"]}, swaps={nom: -(k["swap_long"] + k["swap_short"]) / 2},
                  **base).rets                                  # convention du projet : spot - surcoût
    rB = backtest(costs={nom: k["cout"]}, admin={nom: (-k["swap_long"], -k["swap_short"])},
                  grace=0, **base).rets                         # swaps réels, côté par côté
    sA, sB = stats(rA), stats(rB)
    if not sA:
        return None
    mk = daily_returns(weekdays_only(s))[nom].reindex(rA.index)   # le marché lui-même
    al, tal = _alpha(rA, mk)
    row = {"marché": nom, "classe": classe, "début": rA.index[0].date(), "ans": len(rA) / 252,
           "sharpe_A": sA["sharpe"], "t_A": sA["t"], "cagr_A": sA["cagr"], "dd_A": sA["dd_max"],
           "sharpe_B": sB["sharpe"], "alpha_A": al, "t_alpha": tal,
           "swap_L": k["swap_long"], "swap_C": k["swap_short"], "cout_bp": k["cout"] * 1e4,
           "lot_min": k["notionnel_min"]}
    return row, rA


def run() -> None:
    from concurrent.futures import ProcessPoolExecutor
    c = json.loads(COUTS.read_text())
    px = charger("1995")
    taches = [(nom, classe, px[[nom]].dropna(), c[nom]) for nom, classe, *_ in UNIVERS
              if nom in px and nom in c and np.isfinite(c[nom]["swap_long"])]
    rows, perf = [], {}
    with ProcessPoolExecutor(7) as ex:
        for res in ex.map(_un_marche, taches):
            if res is None:
                continue
            row, r = res
            rows.append(row)
            perf[row["marché"]] = r
            print(f"{row['marché']:8} {row['classe']:14} {row['ans']:5.1f} ans  A {row['sharpe_A']:+.2f} "
                  f"(t {row['t_A']:+.1f})  B {row['sharpe_B']:+.2f}  alpha {row['alpha_A']:+.1%} "
                  f"(t {row['t_alpha']:+.1f})", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "data" / "scan_marches.csv", index=False)
    pd.to_pickle(perf, ROOT / "data" / "scan_rets.pkl")
    print(f"-> data/scan_marches.csv ({len(rows)} marches)")


def _portefeuille(args):
    """Le bot d'origine (sans régime, pour comparer au balayage) sur un panier entier."""
    from trend_bot.backtest import Options, backtest, stats
    nom, closes, couts, swaps = args
    r = backtest(closes, couts, Options(regime=False), swaps=swaps).rets
    return nom, r, stats(r)


def classes() -> None:
    """Portefeuilles par classe d'actifs, convention du projet (spread + surcoût de swap),
    comparés au panier forex de production (11 marchés FRED) sur la même période."""
    from concurrent.futures import ProcessPoolExecutor
    from trend_bot.data import FRED, MARCHE, charger_fred
    from trend_bot.backtest import proba_meilleur
    c = json.loads(COUTS.read_text())
    px = charger("1995")
    # Référence : le panier de production, mêmes conventions
    cm = json.loads((ROOT / "data" / "couts_mt5.json").read_text())["marches"]
    fred = charger_fred(ROOT / "data" / "fred", [k for k in FRED if MARCHE[k] in cm and cm[MARCHE[k]].get("panier", cm[MARCHE[k]].get("valide"))
                                                  and MARCHE[k] not in ("USOIL", "UKOIL", "NGAS")], "1995")
    fx = pd.DataFrame(fred).rename(columns=MARCHE)
    fx = fx[[m for m in fx.columns if m in ("EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF","NZDUSD","USDMXN","USDNOK","USDSEK","NAS100")]]
    fx_c = {m: cm[m]["cout"] for m in fx.columns}
    fx_s = {m: cm[m]["surcout_swap"] for m in fx.columns}
    taches = [("forex 11 (production)", fx, fx_c, fx_s)]
    for cl in ("indices", "crypto", "forex croisés", "actions", "ETF", "métaux"):
        noms = [n for n, k, *_ in UNIVERS if k == cl and n in px and n in c and np.isfinite(c[n]["swap_long"])]
        taches.append((f"{cl} ({len(noms)})", px[noms],
                       {n: c[n]["cout"] for n in noms},
                       {n: -(c[n]["swap_long"] + c[n]["swap_short"]) / 2 for n in noms}))
    # Combinaisons avec la production
    idx = [n for n, k, *_ in UNIVERS if k == "indices" and n in px and n in c]
    comb = pd.concat([fx, px[idx]], axis=1)
    taches.append(("forex 11 + indices", comb, {**fx_c, **{n: c[n]["cout"] for n in idx}},
                   {**fx_s, **{n: -(c[n]["swap_long"] + c[n]["swap_short"]) / 2 for n in idx}}))
    res = {}
    with ProcessPoolExecutor(min(7, len(taches))) as ex:
        for nom, r, st in ex.map(_portefeuille, taches):
            res[nom] = r
            print(f"{nom:28} {r.index[0].year}-{r.index[-1].year}  Sharpe {st['sharpe']:+.2f} (t {st['t']:+.1f})  "
                  f"CAGR {st['cagr']:+.1%}  DD {st['dd_max']:.0%}", flush=True)
    ref = res["forex 11 (production)"]
    print()
    print("Même période que chaque classe, Sharpe forex 11 vs classe, P(classe > forex) par bootstrap par blocs :")
    from trend_bot.backtest import stats
    for nom, r in res.items():
        if nom == "forex 11 (production)":
            continue
        a, b = r.align(ref, join="inner")
        if len(a) < 500:
            continue
        print(f"{nom:28} {a.index[0].year}-{a.index[-1].year}  forex {stats(b)['sharpe']:+.2f}  "
              f"classe {stats(a)['sharpe']:+.2f}  P(classe>forex) {proba_meilleur(a, b):.2f}", flush=True)
    pd.to_pickle(res, ROOT / "data" / "scan_classes.pkl")


if __name__ == "__main__":
    {"fetch": fetch, "couts": couts, "run": run, "classes": classes}[sys.argv[1]]()
