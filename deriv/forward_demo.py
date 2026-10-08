"""Forward-test DÉMO — mesure du vrai edge en conditions réelles.

Principe anti look-ahead (aucun raccourci possible) :
  - t   : fermeture de la bougie t → features(df[:t]) → votes modèles
          → décisions horodatées (UP / DOWN / HOLD) ;
  - t+1 : fermeture de la bougie t+1 → issue RÉELLE révélée
          (win = direction correcte, ret = close[t+1]/close[t] - 1).

Les modèles entraînables sont entraînés UNIQUEMENT sur l'historique
antérieur au début du test (train figé). Les modèles sans état (HMM,
Kalman, RSI, TrendStrength, Momentum) sont ceux du WFO. OnlineLearner
(River) apprend après chaque révélation (boucle fermée, déclaré).

Métriques par modèle : n décisions, WR, baseline UP (mêmes bougies),
edge, p-value binomiale exacte vs 0.5, PnL net (spread par trade),
max drawdown. L'Ensemble applique la règle 4/5 sur les votes.

Sorties (forward/<symbole>_<gran>/) :
  decisions.jsonl   une ligne par décision horodatée
  checkpoint.json   stats courantes + position de reprise
  RAPPORT_CLAUDE.md rapport autonome à donner à un agent d'analyse

Usage :
  .venv/Scripts/python.exe deriv/forward_demo.py \
      --symbol CRASH500 --granularity 60 --train-count 3000 \
      --duration-min 240
"""
from __future__ import annotations

import os
import sys

# Bootstrap du chemin repo (sinon `python deriv/forward_demo.py` met `deriv/`
# sur sys.path et `import deriv` échoue — cf. conftest.py).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import asyncio
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from deriv.data_collector import DerivDataCollector
from deriv.ensemble_predictor import build_features
from deriv.online_learner import OnlineLearner
from deriv.walk_forward_optimizer import WalkForwardOptimizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("FORWARD_DEMO")

# Spread aller par défaut (coût par trade) — overridable en CLI.
SPREAD_DEFAUT = {"CRASH": 0.2, "BOOM": 0.2, "DEFAULT": 0.2}

ONLINE_FEATURES_MAP = {          # feat_col -> clé FEATURES_ONLINE
    "rsi": "rsi_14", "adx": "adx_14", "atr": "atr_14",
}


def spread_defaut(symbol: str) -> float:
    s = symbol.upper()
    for cle, val in SPREAD_DEFAUT.items():
        if cle != "DEFAULT" and cle in s:
            return val
    return SPREAD_DEFAUT["DEFAULT"]


def proba_from(res: dict) -> float:
    """Priorité P0 : p_useful → proba_up → vote (0/0.5/1)."""
    for cle in ("p_useful", "proba_up"):
        v = res.get(cle)
        if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
            return float(v)
    v = res.get("vote")
    if isinstance(v, (int, float)) and 0.0 <= float(v) <= 1.0:
        return float(v)
    return 0.5


def binom_pvalue(wins: int, n: int) -> float:
    """p-value bilatérale exacte du WR vs 0.5 (test binomial)."""
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(int(wins), int(n), 0.5).pvalue)


def max_drawdown(pnl: list[float]) -> float:
    if not pnl:
        return 0.0
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    return float(np.max(peak - cum))


class ForwardTester:
    """Harnais de mesure d'edge live — décisions AVANT, issues APRÈS."""

    def __init__(self, symbol: str, granularity: int, train_count: int,
                 spread: float, out_root: str = "forward"):
        self.symbol = symbol
        self.granularity = granularity
        self.train_count = train_count
        self.spread = spread
        self.out_dir = Path(out_root) / f"{symbol}_{granularity}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.f_dec = self.out_dir / "decisions.jsonl"
        self.f_issues = self.out_dir / "issues.jsonl"
        self.f_ck = self.out_dir / "checkpoint.json"
        self.f_rap = self.out_dir / "RAPPORT_CLAUDE.md"
        self.wfo = WalkForwardOptimizer()
        self.trained: dict = {}          # nom -> predict(df_ohlcv)->dict
        # Isolation : état online du FORWARD (jamais le pickle de prod)
        self.online = OnlineLearner(
            state_file=str(self.out_dir / "online_learner_fwd.pkl"),
        )
        self.pending: list[dict] = []    # décisions en attente d'issue
        self.t0_forward: float | None = None
        self.n_decisions = 0
        self.n_issues = 0
        self.dernier_close: float | None = None
        self.n_up_reel = 0               # baseline UP (même échantillon)
        self.n_down_reel = 0

    # ── Entraînement initial (historique STRICTEMENT antérieur) ────
    def train_models(self, df: pd.DataFrame) -> dict:
        rap = {}
        for name, ModelClass in self.wfo.models.items():
            try:
                m = ModelClass()
                self.wfo._disarm_save(m)
                m.train(df)
                self.trained[name] = m.predict
                rap[name] = "OK"
            except Exception as exc:
                rap[name] = f"indisponible : {exc}"
        for name, ModelClass in self.wfo.stateless_models.items():
            try:
                wrap = self.wfo._StatelessWrapper(ModelClass())
                self.trained[name] = wrap.predict
                rap[name] = "OK (stateless)"
            except Exception as exc:
                rap[name] = f"indisponible : {exc}"
        self.t0_forward = time.time()
        return rap

    # ── Décision à la fermeture de t ────────────────────────────────
    def decide(self, df: pd.DataFrame) -> None:
        feat = build_features(df)
        if feat.empty:
            return
        row_close = float(df["close"].iloc[-1])
        self.dernier_close = row_close     # référence baseline UP
        ts = str(df.index[-1])
        votes_dir: dict[str, float] = {}
        for name, predict in self.trained.items():
            try:
                # Contrat WFO : fenêtre OHLCV BRUTE (chaque modèle fait son
                # build_features interne — cf. _StatelessWrapper.predict).
                res = predict(df)
                p = proba_from(res if isinstance(res, dict) else {})
            except Exception:
                continue
            dec = "UP" if p > 0.5 else ("DOWN" if p < 0.5 else "HOLD")
            if dec != "HOLD":
                votes_dir[name] = p
            self._log_decision(ts, name, dec, p, row_close)
        self._ensemble_et_online(ts, feat, votes_dir, row_close)

    def _ensemble_et_online(self, ts, feat, votes_dir, row_close) -> None:
        try:
            of = self._online_features(feat)
            r_on = self.online.predict(of)
            p_on = float(r_on.get("proba_up", 0.5) or 0.5)
            dec_on = "UP" if p_on > 0.5 else ("DOWN" if p_on < 0.5
                                              else "HOLD")
            # Vote online = INFO uniquement : il n'entre PAS dans la règle
            # 4/5 (docstring online_learner : verrouillée sur les batch).
            self._log_decision(ts, "Online", dec_on, p_on, row_close,
                               feat_online=of)
        except Exception:
            pass
        # Ensemble 4/5 : >= 4 votes directionnels d'accord
        ups = sum(1 for p in votes_dir.values() if p > 0.5)
        downs = sum(1 for p in votes_dir.values() if p < 0.5)
        dec_ens = ("UP" if ups >= 4 else
                   "DOWN" if downs >= 4 else "HOLD")
        p_ens = (ups / (ups + downs)) if (ups + downs) else 0.5
        self._log_decision(ts, "Ensemble4sur5", dec_ens, p_ens, row_close)

    def _online_features(self, feat: pd.DataFrame) -> dict:
        row = feat.iloc[-1]
        out: dict = {}
        for cle, col in ONLINE_FEATURES_MAP.items():
            try:
                out[cle] = float(row.get(col, 0.0) or 0.0)
            except Exception:
                out[cle] = 0.0
        out["hmm_confidence"] = 0.0
        out["ensemble_votes"] = 0.0
        out["volume_ratio"] = 1.0
        out["risk_reward"] = 0.0
        h = pd.Timestamp.now(tz="UTC")
        out["session_london"] = 1.0 if 7 <= h.hour < 10 else 0.0
        out["session_overlap"] = 1.0 if 12 <= h.hour < 15 else 0.0
        out["session_newyork"] = 1.0 if 15 <= h.hour < 20 else 0.0
        out["hour_sin"] = math.sin(2 * math.pi * h.hour / 24)
        out["hour_cos"] = math.cos(2 * math.pi * h.hour / 24)
        out["day_of_week"] = float(h.dayofweek)
        return out

    def _log_decision(self, ts, name: str, dec: str, p: float,
                      prix: float, feat_online: dict | None = None) -> None:
        d = {"model": name, "decision": dec, "proba": round(p, 4),
             "prix_entree": prix, "ts_decision": str(ts)}
        if feat_online is not None:
            d["feat_online"] = dict(feat_online)   # pour learn_one à t+1
        self.pending.append(d)
        self.n_decisions += 1
        with open(self.f_dec, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": str(ts), "model": name,
                                "decision": dec, "proba": round(p, 4),
                                "prix": prix}) + "\n")

    # ── Révélation à la fermeture de t+1 ────────────────────────────
    def reveal(self, ts, close: float) -> None:
        for d in self.pending:
            ret = close / d["prix_entree"] - 1.0
            dir_sign = 1.0 if d["decision"] == "UP" else -1.0
            win = (dir_sign > 0 and ret > 0) or (dir_sign < 0 and ret < 0)
            pnl_net = dir_sign * ret - self.spread
            d.update({"ts_reveal": str(ts), "ret": round(ret, 8),
                      "win": bool(win), "pnl_net": round(pnl_net, 8)})
            self.n_issues += 1
            with open(self.f_issues, "a", encoding="utf-8") as f:
                f.write(json.dumps(d) + "\n")
            if d["model"] == "Online":
                # Boucle fermée : learn_one avec les features vues à t
                # (pas des zéros) — mesure honnête de l'adaptation online.
                try:
                    self.online.update(d.get("feat_online") or {},
                                       1 if win else 0)
                except Exception:
                    pass
        self.pending = []
        # Baseline UP — MÊME échantillon de bougies que les modèles :
        # si un WR de modèle ≈ cette baseline, c'est le drift copié.
        if self.dernier_close:
            ret_b = close / self.dernier_close - 1.0
            self.n_issues += 1
            if ret_b > 0:
                self.n_up_reel += 1
            else:
                self.n_down_reel += 1
            with open(self.f_issues, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": str(ts), "model": "BaselineUP", "decision": "UP",
                    "proba": 1.0, "prix_entree": self.dernier_close,
                    "ts_reveal": str(ts), "ret": round(ret_b, 8),
                    "win": bool(ret_b > 0),
                    "pnl_net": round(ret_b - self.spread, 8)}) + "\n")

    # ── Agrégation des issues par modèle ────────────────────────────
    def stats(self) -> dict:
        if not self.f_issues.exists():
            return {}
        par_modele: dict[str, list] = {}
        with open(self.f_issues, encoding="utf-8") as f:
            for ligne in f:
                try:
                    d = json.loads(ligne)
                except json.JSONDecodeError:
                    continue
                if d.get("decision") == "HOLD":
                    continue
                par_modele.setdefault(d["model"], []).append(d)
        out = {}
        for nom, rows in sorted(par_modele.items()):
            n = len(rows)
            wins = sum(1 for r in rows if r["win"])
            pnl = [r["pnl_net"] for r in rows]
            rets = [r["ret"] for r in rows]
            out[nom] = {
                "n": n,
                "win_rate": round(wins / n, 4) if n else 0.0,
                "p_binomiale": round(binom_pvalue(wins, n), 4),
                "pnl_net_cum": round(float(np.sum(pnl)), 6),
                "pnl_moyen": round(float(np.mean(pnl)), 8) if n else 0.0,
                                "max_dd": round(max_drawdown(pnl), 6),
                "ret_moyen_brut": round(float(np.mean(rets)), 8) if n else 0.0,
            }
        return out

    def checkpoint(self) -> None:
        """Sauvegarde portable — stats + état de reprise pour run long."""
        ck = {
            "symbol": self.symbol,
            "granularity": self.granularity,
            "spread": self.spread,
            "n_decisions": self.n_decisions,
            "n_issues": self.n_issues,
            "n_up_reel": self.n_up_reel,
            "n_down_reel": self.n_down_reel,
            "pending": len(self.pending),
            "t0_forward": self.t0_forward,
            "checked_at": time.time(),
            "stats": self.stats(),
        }
        with open(self.f_ck, "w", encoding="utf-8") as f:
            json.dump(ck, f, indent=2, default=str)

    def rapport(self) -> None:
        """Rapport autonome a remettre a Claude (analyse critique)."""
        st = self.stats()
        base = self._baseline_pct()
        elapsed = (time.time() - (self.t0_forward or time.time())) / 60.0
        lines = []
        a = lines.append
        a("# Rapport forward-demo - mesure de l'edge en conditions reelles")
        a("")
        a("- **Symbole** : `" + self.symbol + "` - granularite " + str(self.granularity) + "s")
        a("- **Duree**   : " + f"{elapsed:.1f}" + " min - " + str(self.n_decisions) + " decisions, " + str(self.n_issues) + " issues revelees")
        a("- **Spread aller simple** : " + f"{self.spread:.4%}")
        a("- **Base UP (baseline drift)** : " + f"{base:.1f}" + " % (" + str(self.n_up_reel) + " UP / " + str(self.n_down_reel) + " DOWN)")
        a("")
        a("> Seuil de reference : la baseline UP (drift brut). Un modele")
        a("dont le WR = baseline copie le drift -> **aucun edge**.")
        a("")
        a("## Resultats par modele (forward reel, issues horodatées)")
        a("")
        a("| Modele | Decisions | WR | Edge | p-value | PnL net | Verdict |")
        a("|---|---|---|---|---|---|---|")
        if not st:
            a("| - | - | - | - | - | - | aucune issue revelee |")
        for nom, s2 in st.items():
            wr = s2["win_rate"]
            is_base = nom == "BaselineUP"
            edge = (wr - base / 100.0) * 100.0 if not is_base else 0.0
            base_str = f"{base:.1f} %" if not is_base else "-"
            edge_str = f"{edge:+.1f} pp" if not is_base else "-"
            sig = (not is_base) and (s2["p_binomiale"] < 0.05 and edge > 0.5)
            verdict = ("VERT_EDGE" if sig else ("- drift" if is_base else "ROUGE_drift"))
            pnl = s2["pnl_net_cum"]
            s_pnl = "+" if pnl >= 0 else ""
            a(f"| {nom} | {s2['n']} | {wr*100:.1f} % | {base_str} | {edge_str} | {s2['p_binomiale']:.4f} | {s_pnl}{pnl:+.0f} | {verdict} |")
        a("")
        a("## Verdict global")
        a("")
        edges = []
        for nom, s2 in st.items():
            if nom == "BaselineUP":
                continue
            e = (s2["win_rate"] - base / 100.0) * 100.0
            if s2["p_binomiale"] < 0.05 and abs(e) > 0.5:
                edges.append(nom)
        if edges:
            a(f"Sur {self.n_issues} issues, {len(edges)} modele(s) ({'/'.join(edges)}) montrent un edge significatif (p<0.05 ET edge>0.5pp).")
            a("ATTENTION : valider absence de fuite temporelle + confronter a permutation_test_score.")
        else:
            a(f"Sur {self.n_issues} issues, AUCUN modele ne bat la baseline UP ({base:.1f} %) de façon significative (p<0.05 ET edge>0.5pp).")
            a("-> Aucun edge prédictif demontre en forward reel.")
        a("")
        a("Interpretation : WR ≈ baseline UP = drift capture. Le garde-fou")
        a("fonctionne : il empeche la conclusion trompeuse des WR nus.")
        a("")
        a("### A remettre a Claude")
        a("1. Un modele sort-il vraiment de la baseline UP (edge/signif) ?")
        a("2. PnL net positif malgre le spread ?")
        a("3. Max DD supportable ?")
        a("4. volume_ratio/session_* features (Online) apportent quoi ?")
        a("5. Confroncer WR live vs permutation_test_score sur ce flow.")
        a("")
        with open(self.f_rap, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("Rapport -> %s", self.f_rap)

    def _baseline_pct(self) -> float:
        n = self.n_up_reel + self.n_down_reel
        return (self.n_up_reel / n * 100.0) if n else 0.0

    # ── Boucle principale forward (temps réel) ────────────────────────
    async def run_live(self, duration_min: int) -> None:
        logger.info(
            "Forward live %s/%ss — %d min (train_count=%d)",
            self.symbol, self.granularity, duration_min, self.train_count,
        )
        # Entraînement sur historique STRICTEMENT antérieur au lancement.
        collector = DerivDataCollector(self.symbol, self.granularity)
        await collector.connect()
        df_train = await collector.get_candle_history_full(
            target_count=self.train_count,
        )
        await collector.close()
        if df_train is None or df_train.empty:
            raise RuntimeError("Historique d'entraînement vide — stop.")
        rap_train = self.train_models(df_train)
        for k, v in rap_train.items():
            logger.info("Train %-16s %s", k, v)
        self.buffer = df_train.copy()
        self.seen = set(df_train.index)
        logger.info("Forward démarré — poll toutes les %ss", self.granularity)
        deadline = time.time() + duration_min * 60.0
        poll_tail = min(60, self.train_count)   # ventouse de polling
        while time.time() < deadline:
            await asyncio.sleep(self.granularity)
            try:
                c = DerivDataCollector(self.symbol, self.granularity)
                await c.connect()
                tail = await c.get_candle_history_full(
                    target_count=poll_tail,
                )
                await c.close()
            except Exception as exc:
                logger.warning("Polling échoué : %s — retry", exc)
                continue
            if tail is None or tail.empty:
                continue
            # Uniquement les bougies JAMAIS vues (anti double-comptage).
            new = [i for i in tail.index if i not in self.seen]
            for idx in new:
                close = float(tail.loc[idx, "close"])
                self.reveal(idx, close)         # issue de la décision t-1
                self.buffer = pd.concat(
                    [self.buffer, tail.loc[[idx]]
                     ]
                ).sort_index()
                self.seen.add(idx)
                # Limite la fenêtre build_features (fenêtre glissante).
                max_keep = min(self.train_count, 1500)
                if len(self.buffer) > max_keep:
                    self.buffer = self.buffer.iloc[-max_keep:]
                self.decide(self.buffer)        # décision à clôture idx
            if new:
                self.checkpoint()
        self.rapport()
        logger.info("Forward terminé — %d décisions / %d issues",
                     self.n_decisions, self.n_issues)

    def train_models_sync(self, df: pd.DataFrame) -> dict:
        """Wrapper synchrone du ForwardTester pour usage hors-boucle."""
        return self.train_models(df)


# ── CLI ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forward-test DÉMO (edge réel)")
    p.add_argument("--symbol", default="CRASH500")
    p.add_argument("--granularity", type=int, default=60)
    p.add_argument("--train-count", type=int, default=3000)
    p.add_argument("--duration-min", type=int, default=240)
    p.add_argument("--spread", type=float, default=None,
                   help="Spread aller simple (défaut : calibré symbole)")
    p.add_argument("--out", default="forward")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spread = args.spread if args.spread is not None else spread_defaut(
        args.symbol
    )
    ft = ForwardTester(
        args.symbol, args.granularity, args.train_count, spread, args.out,
    )
    logger.info("Répertoire forward → %s", ft.out_dir)
    asyncio.run(ft.run_live(args.duration_min))


if __name__ == "__main__":
    main()
