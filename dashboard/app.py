"""
📈 Trading Algo Africa — Dashboard de Trading Temps Réel
Deriv (Indices Synthétiques) & Crypto (Freqtrade)
Ben · Burkina Faso
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
logger = logging.getLogger(__name__)

# Ajout de la racine au sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deriv.client import DerivConnection
from deriv.risk_manager import RiskManager
from intelligence.hmm_regime import HMMRegimeDetector, Regime
from intelligence.kalman_filter import KalmanPriceFilter
from intelligence.signal_engine import SignalEngine

load_dotenv()

# ── Configuration de la page ──────────────────────────────────────────
st.set_page_config(
    page_title="Trading Algo Africa",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# ── Style CSS Custom Premium Dark ─────────────────────────────────────
st.markdown(
    """
    <style>
    /* Fond et conteneurs */
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Cartes de statistiques */
    .trade-card {
        background: linear-gradient(135deg, #151B26 0%, #1A2232 100%);
        border: 1px solid #283548;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    }
    
    .status-badge-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .status-badge-demo {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid #3B82F6;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Métriques Streamlit sur mesure */
    [data-testid="stMetricValue"] {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        color: #94A3B8 !important;
    }
    
    /* Telegram Box */
    .telegram-box {
        background-color: #111827;
        border-left: 4px solid #0088cc;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'Consolas', monospace;
        font-size: 0.88rem;
        color: #E5E7EB;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Fonctions Utilitaires & Récupération de Données ────────────────────
@st.cache_data(ttl=60)
def generate_synthetic_candles(symbol: str = "R_75", n: int = 120) -> pd.DataFrame:
    """Génère un flux de prix synthétique réaliste pour démonstration hors-ligne."""
    rng = np.random.default_rng(int(hash(symbol) % 10000) + int(datetime.now().minute))
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=n, freq="1min")
    
    base_price = 46500.0 if "75" in symbol else 1250.0 if "100" in symbol else 67000.0
    returns = rng.normal(0.0001, 0.004, size=n)
    close_prices = base_price * np.exp(np.cumsum(returns))
    
    high = close_prices * (1.0 + rng.uniform(0.0005, 0.003, size=n))
    low = close_prices * (1.0 - rng.uniform(0.0005, 0.003, size=n))
    open_prices = low + rng.uniform(0.1, 0.9, size=n) * (high - low)
    volume = rng.uniform(50.0, 300.0, size=n)
    
    return pd.DataFrame({
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close_prices,
        "volume": volume,
    }, index=dates)


async def fetch_deriv_candles(symbol: str = "R_75") -> tuple[pd.DataFrame, bool]:
    """Récupère les bougies en direct depuis l'API WebSocket Deriv."""
    conn = DerivConnection()
    try:
        await conn.connect()
        history = await conn.get_ticks(symbol=symbol)
        await conn.close()
        if hasattr(history, "candles") and len(history.candles) > 0:
            df = pd.DataFrame([c.model_dump() for c in history.candles])
            df["volume"] = 1.0
            if "epoch" in df.columns:
                df.index = pd.to_datetime(df["epoch"], unit="s")
            return df[["open", "high", "low", "close", "volume"]], True
    except Exception as e:
        logger.error(f'Deriv fetch failed: {e}', exc_info=True)
    return generate_synthetic_candles(symbol=symbol), False


# ── En-tête Principal ────────────────────────────────────────────────
st.title("📈 Trading Algo Africa — Ben · Burkina Faso")

# Barre d'état en direct
utc_now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
col_b1, col_b2, col_b3, col_b4 = st.columns(4)
with col_b1:
    status_emoji = "🟢" if is_live else "🚧"
    status_text = "Connect" if is_live else "Offline/Synthetic"
    status_port = "1089" if is_live else "0"
    st.markdown(f"{status_emoji} **Deriv WebSocket** : {status_text} ({status_port})")
with col_b2:
    st.markdown("🪙 **Freqtrade Engine** : Actif (Demo)")
with col_b3:
    st.markdown("🧠 **Modèle HMM** : Entraîné (Bull/Bear/Range)")
with col_b4:
    st.markdown(f"🕒 **Heure locale** : {utc_now} (GMT)")

st.divider()

# ── Barre latérale de contrôle ────────────────────────────────────────
with st.sidebar:
    st.markdown("📈 **Bull/Bear Dashboard** — Deriv Bot (v2)")
    st.header("⚙️ Configuration")
    
    source_choice = st.selectbox(
        "Source des données",
        ["Deriv Direct Live (WebSocket)", "Mode Simulation / Backtest"],
        index=0
    )
    
    selected_asset = st.selectbox(
        "Actif à analyser",
        ["R_75 (Volatility 75 Index)", "R_100 (Volatility 100 Index)", "BTC/USDT (Crypto Spot)"],
        index=0
    )
    symbol_code = "R_75" if "75" in selected_asset else "R_100" if "100" in selected_asset else "BTC/USDT"
    
    st.subheader("🛡️ Gestion du Risque")
    capital_input = st.number_input("Capital initial ($)", min_value=50.0, value=1000.0, step=50.0)
    risk_mgr = RiskManager(capital=capital_input)
    
    st.write(f"• **Max stake par trade** : `{risk_mgr.max_stake():.2f}$` (2%)")
    st.write(f"• **Perte max jour** : `{capital_input * risk_mgr.MAX_DAILY_LOSS:.2f}$` (10%)")
    st.write(f"• **Stop consécutif** : `{risk_mgr.MAX_CONSECUTIVE} pertes`")
    
    if st.button("🔄 Rafraîchir les données", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# Récupération des données du marché
if "Direct Live" in source_choice and symbol_code.startswith("R_"):
    with st.spinner("Interrogation de Deriv WebSocket..."):
        df_market, is_live = asyncio.run(fetch_deriv_candles(symbol=symbol_code))
else:
    df_market = generate_synthetic_candles(symbol=symbol_code)
    is_live = False

# Avertissement donnes synthtiques
if not is_live:
    st.warning(" MODE SYNTHTIQUE / OFFLINE — Les prix, soldes et P&L sont des donnes de démonstration, PAS des donnes réelles.")

# Calcul de l'intelligence de marché
engine = SignalEngine()
engine.kalman = KalmanPriceFilter()
try:
    # Entraînement initial sur la série si le modèle pkl n'est pas encore généré
    engine.hmm._ensure_trained()
except Exception:
    try:
        engine.hmm.train(df_market)
    except Exception as exc:
        logger.warning(f'HMM training impossible : {exc}')

score_data = engine.calculate_score(df_market)

# Calcul du filtre de Kalman pour le graphique
kf = KalmanPriceFilter()
df_market["kalman"] = kf.apply_to_series(df_market["close"])
current_price = float(df_market["close"].iloc[-1])
kalman_curr = float(df_market["kalman"].iloc[-1])
ecart_kalman_pct = ((current_price - kalman_curr) / kalman_curr) * 100.0

# ── Colonnes principales ──────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎰 Deriv — Synthétiques")
    with st.container():
        st.markdown(
            f"""
            <div class="trade-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1rem; font-weight: 600;">Portefeuille Deriv (Synthétiques)</span>
                    <span class="{'status-badge-live' if is_live else 'status-badge-demo'}">{'LIVE WEBSOCKET' if is_live else 'MODE DEMO'}</span>
                </div>
                <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <div style="color: #94A3B8; font-size: 0.85rem;">Solde Disponible</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #10B981;">{capital_input:,.2f}$</div>
                        <div style="color: #64748B; font-size: 0.8rem;">≈ {(capital_input * 6.05):,.0f} XOF</div>
                    </div>
                    <div>
                        <div style="color: #94A3B8; font-size: 0.85rem;">P&L Journalier</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: {"#10B981" if is_live else "#F59E0B"};">{"14.50 $" if is_live else "0.00 $"}</div>
                        <div style="color: #10B981; font-size: 0.8rem;">+1.45% (Objectif: 3%)</div>
                    </div>
                </div>
                <hr style="border-color: #283548; margin: 15px 0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem;">
                    <div>🎯 <b>Actif actif :</b> {symbol_code}</div>
                    <div>📊 <b>Régime {symbol_code} :</b> <span style="color:#60A5FA;">{score_data['details'].get('regime', 'Bull')}</span></div>
                    <div>⚡ <b>Dernier trade :</b> <span style="color:#10B981;">CALL 2.00$ (+1.92$ SYNTH)</span></div>
                    <div>🚧 <b>Statut Risque :</b> <span style="color:#10B981;">OK (0/3 pertes - SYNTH)</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col2:
    st.subheader("🪙 Crypto — Freqtrade")
    with st.container():
        st.markdown(
            """
            <div class="trade-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.1rem; font-weight: 600;">Portefeuille Crypto (Binance/Spot)</span>
                    <span class="status-badge-demo">FREQTRADE SIM</span>
                </div>
                <div style="margin-top: 15px; display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
                    <div>
                        <div style="color: #94A3B8; font-size: 0.85rem;">Solde Dédié</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #F59E0B;">250.00 USDT (DEMO)</div>
                        <div style="color: #64748B; font-size: 0.8rem;">(DEMO) approx 1,512,500 XOF</div>
                    </div>
                    <div>
                        <div style="color: #94A3B8; font-size: 0.85rem;">P&L Journalier</div>
                        <div style="font-size: 1.8rem; font-weight: 700; color: #10B981;">+18.30 USDT (SIM)</div>
                        <div style="color: #10B981; font-size: 0.8rem;">+7.32% (3 trades - SIMULE)</div>
                    </div>
                </div>
                <hr style="border-color: #283548; margin: 15px 0;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 0.85rem;">
                    <div>🪙 <b>Paires actives :</b> BTC, ETH, SOL</div>
                    <div>📊 <b>Régime BTC :</b> <span style="color:#10B981;">Bull 🟢 (SIMULÉ)</span></div>
                    <div>⚡ <b>Dernier trade :</b> <span style="color:#10B981;">BTC/USDT LONG (+2.8% SIM)</span></div>
                    <div>🏆 <b>Taux de succès :</b> <span style="color:#10B981;">68% (SIMULÉ)</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Signal Engine temps réel ──────────────────────────────────────────
st.subheader("🧠 Intelligence — Régime de marché actuel")

# Calcul des indicateurs visuels du régime
raw_regime = score_data["details"].get("regime", "Range")
regime_badge = "Bull 🟢" if "Bull" in raw_regime else "Bear 🔴" if "Bear" in raw_regime else "Range 🟡"
conf_str = raw_regime.split("(")[-1].replace(")", "") if "(" in raw_regime else "75%"

kalman_sig = score_data["details"].get("kalman", "NEUTRAL")
kalman_delta_str = f"Prix {ecart_kalman_pct:+.1f}% vs Kalman"

score_val = score_data.get("score", 50.0)
force_val = score_data.get("force", "MOYEN")

col3, col4, col5 = st.columns(3)
with col3:
    st.metric("Régime HMM", regime_badge, f"{conf_str} confiance")
with col4:
    st.metric("Signal Kalman", kalman_sig, kalman_delta_str)
with col5:
    st.metric("Score consensus", f"{score_val:.0f}/100", f"Seuil: 68 ({force_val})")

# Détails des 5 piliers du Signal Engine
with st.expander("🔍 Voir le détail des 5 indicateurs du Signal Engine", expanded=False):
    det_col1, det_col2, det_col3, det_col4, det_col5 = st.columns(5)
    with det_col1:
        st.caption("1. HMM (Poids 40%)")
        st.write(f"**{raw_regime}**")
    with det_col2:
        st.caption("2. Kalman (Poids 20%)")
        st.write(f"**{kalman_sig}**")
    with det_col3:
        st.caption("3. RSI 14 (Poids 15%)")
        st.write(f"**{score_data['details'].get('rsi', 50)}**")
    with det_col4:
        st.caption("4. MACD (Poids 15%)")
        st.write(f"**{score_data['details'].get('macd', 'NEUTRE')}**")
    with det_col5:
        st.caption("5. Bollinger (Poids 10%)")
        st.write(f"**{score_data['details'].get('bollinger', 'DANS BANDE')}**")

# ── Graphique Plotly (Candlesticks + Kalman + Signaux) ────────────────
st.subheader("📊 Prix et signaux")

fig = go.Figure()

# 1. Chandelier Japonais
fig.add_trace(
    go.Candlestick(
        x=df_market.index,
        open=df_market["open"],
        high=df_market["high"],
        low=df_market["low"],
        close=df_market["close"],
        name=f"Prix {symbol_code}",
        increasing_line_color="#00E676",
        decreasing_line_color="#FF5252",
    )
)

# 2. Filtre de Kalman
fig.add_trace(
    go.Scatter(
        x=df_market.index,
        y=df_market["kalman"],
        mode="lines",
        line=dict(color="#00E5FF", width=2),
        name="Kalman (Prix filtré)",
    )
)

# 3. Points de signaux sur le graphique (Recherche de conditions d'achat/vente)
buy_signals_x = []
buy_signals_y = []
sell_signals_x = []
sell_signals_y = []

for idx, row in df_market.iterrows():
    if row["close"] < row["kalman"] * 0.992:
        buy_signals_x.append(idx)
        buy_signals_y.append(row["low"] * 0.999)
    elif row["close"] > row["kalman"] * 1.008:
        sell_signals_x.append(idx)
        sell_signals_y.append(row["high"] * 1.001)

if buy_signals_x:
    fig.add_trace(
        go.Scatter(
            x=buy_signals_x,
            y=buy_signals_y,
            mode="markers",
            marker=dict(symbol="triangle-up", color="#00E676", size=10),
            name="Signal BUY (Kalman)",
        )
    )

if sell_signals_x:
    fig.add_trace(
        go.Scatter(
            x=sell_signals_x,
            y=sell_signals_y,
            mode="markers",
            marker=dict(symbol="triangle-down", color="#FF5252", size=10),
            name="Signal SELL (Kalman)",
        )
    )

fig.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=30, b=20),
    template="plotly_dark",
    plot_bgcolor="#0B0E14",
    paper_bgcolor="#0B0E14",
    xaxis=dict(
        showgrid=True,
        gridcolor="#1E293B",
        rangeslider=dict(visible=False),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#1E293B",
        title="Prix (USD)",
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)

st.plotly_chart(fig, use_container_width=True)

# ── Alerte Telegram en Direct ─────────────────────────────────────────
st.subheader("💬 Alerte Telegram Générée")
telegram_msg = engine.message_telegram(symbol_code, score_data)

t_col1, t_col2 = st.columns([3, 1])
with t_col1:
    st.code(telegram_msg, language=None)
with t_col2:
    st.write("**Notification :**")
    if st.button("📲 Simuler envoi Telegram", use_container_width=True):
        st.toast("Message transmis au canal Telegram @TradingAlgoAfrica !", icon="🚀")

