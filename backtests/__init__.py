# -*- coding: utf-8 -*-
"""
backtests — conteneur des résultats de Walk-Forward Optimization.

Fournit une entrée stable `get_latest_backtest_summary()` pour les hooks
automatiques (journalisation, reporting, weekly_retrain, etc.).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


BACKTESTS_DIR = Path(__file__).resolve().parent


def _paths() -> List[Path]:
    return sorted(BACKTESTS_DIR.glob("wfo_summary_*.json"))


def latest_summary_path() -> Optional[Path]:
    paths = _paths()
    return paths[-1] if paths else None


def get_latest_backtest_summary() -> Optional[Dict[str, Any]]:
    """
    Retourne le résumé complet (déjà parsé) du dernier backtest WFO.
    Format identique au JSON sauvegardé par walk_forward_optimizer.run().
    """
    path = latest_summary_path()
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_summary_flat() -> Dict[str, Dict[str, Any]]:
    """
    Version "plates" : {model: {win_rate_moyen, profitable, ...}}
    pour un affichage / logging rapide.
    """
    full = get_latest_backtest_summary()
    if full is None:
        return {}
    return full


def latest_summary_markdown() -> str:
    """
    Génère un mini rapport lisible en texte brut (pour logs / slack / email).
    """
    full = get_latest_backtest_summary()
    if full is None:
        return "Aucun backtest WFO trouvé."

    lines = []
    lines.append("=" * 60)
    lines.append("DERNIER BACKTEST WALK-FORWARD — Résumé")
    lines.append("=" * 60)
    for model, stats in full.items():
        wr = stats.get("win_rate_moyen", 0)
        std = stats.get("win_rate_std", 0)
        tr = stats.get("n_trades_total", 0)
        profitable = bool(stats.get("profitable", False))
        status = "OK: PROFITABLE" if profitable else "FAIL: SOUS SEUIL"
        lines.append("")
        lines.append(f"{model:<16} {status}")
        lines.append(f"  Win Rate moyen : {wr:.1%}")
        lines.append(f"  Écart-type     : {std:.1%}")
        lines.append(f"  Trades total   : {tr}")
        lines.append(f"  Folds          : {stats.get('n_folds', 0)}")
    lines.append("")
    lines.append("=" * 60)
    any_ok = any(stats.get("profitable", False) for stats in full.values())
    if any_ok:
        lines.append("OK: Au moins un modèle est profitable.")
    else:
        lines.append("WARN: Aucun modèle ne dépasse le seuil actuellement.")
    return "\n".join(lines)


def enumerate_backtests() -> List[Dict[str, Any]]:
    """
    Liste tous les backtests existants avec métadonnées minimales.
    Utile pour l'audit et les diagnostics automatiques.
    """
    out = []
    for path in _paths():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        stem = path.stem.replace("wfo_summary_", "")
        out.append({
            "file": path.name,
            "symbol_timeframe": stem,
            "timestamp": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
            "models": list(data.keys()),
            "best_model": None,
            "best_wr": None,
        })
        best_model = None
        best_wr = -1.0
        for model, stats in data.items():
            wr = stats.get("win_rate_moyen", 0)
            if wr > best_wr:
                best_wr = wr
                best_model = model
        out[-1]["best_model"] = best_model
        out[-1]["best_wr"] = best_wr
    return out


def save_manual_summary(tag: str, summary: Dict[str, Any]) -> Path:
    """
    Persiste manuellement un résumé (utile pour les runners externes).
    Retourne le chemin du fichier écrit.
    """
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"wfo_summary_{tag}_{ts}.json"
    path = BACKTESTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    return path
