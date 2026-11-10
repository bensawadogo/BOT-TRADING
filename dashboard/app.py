"""Dashboard du bot de tendance : affiche uniquement ce que le bot a écrit.

    python main.py dashboard

Sources : journal des décisions (TREND_JOURNAL), état du coupe-circuit (TREND_STATE) et
courbe d'équité (trend_equity.csv, à côté du journal). Aucune donnée simulée.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trend_bot import config  # noqa: E402
from trend_bot.runner import equity_path  # noqa: E402

journal = Path(os.getenv("TREND_JOURNAL", config.JOURNAL_PATH))
etat = Path(os.getenv("TREND_STATE", config.STATE_PATH))

st.set_page_config(page_title="Bot de tendance", layout="wide")
st.title("Bot de tendance MT5")

if not journal.exists():
    st.info("Aucun cycle enregistré. Lance `python main.py` (simulation) ou "
            "`python main.py trend --live --loop` (compte démo).")
    st.stop()

j = pd.read_csv(journal)
s = json.loads(etat.read_text()) if etat.exists() else {}
dernier = j[j["date"] == j["date"].max()].drop_duplicates("marche", keep="last")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dernière clôture traitée", str(j["date"].max()), s.get("mode", ""))
if s.get("equity"):
    dd = 1 - s["equity"] / s["peak"] if s.get("peak") else 0.0
    c2.metric("Équité", f"{s['equity']:,.2f}")
    c3.metric("Drawdown depuis le plus haut", f"{dd:.1%}")
c4.metric("Coupe-circuit", "DÉCLENCHÉ — reset manuel" if s.get("tripped") else "armé")

courbe = equity_path(journal)
if courbe.exists():
    e = pd.read_csv(courbe, parse_dates=["horodatage_utc"]).set_index("horodatage_utc")
    st.subheader("Équité à chaque cycle")
    st.line_chart(e["equite"])
    jours = e["jour"].nunique()
    st.caption(f"{jours} jour(s) de cotation traité(s). Objectif : 3 mois de démo (≈ 63 jours) "
               "avant tout compte réel.")

st.subheader("Dernier cycle")
colonnes = ["marche", "symbole", "signal", "regime", "poids", "lots_actuels",
            "lots_cibles", "action", "resultat", "raison"]
st.dataframe(dernier[[c for c in colonnes if c in dernier.columns]], hide_index=True,
             use_container_width=True)

echecs = j[j["resultat"].fillna("").str.startswith("ÉCHEC")]
petits = dernier[dernier["resultat"].fillna("").str.startswith("sous le lot minimum")]
if len(petits):
    st.warning(f"{len(petits)} marché(s) sous le lot minimum au dernier cycle : "
               + ", ".join(petits["marche"]) + ". L'équité est trop faible pour les trader.")
if len(echecs):
    st.error(f"{len(echecs)} ordre(s) refusé(s) dans le journal. Dernier : "
             f"{echecs.iloc[-1]['symbole']} — {echecs.iloc[-1]['resultat']}")
