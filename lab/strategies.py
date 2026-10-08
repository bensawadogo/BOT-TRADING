"""
Traduction FIDÈLE des algorithmes du dépôt, en signaux causaux (bougie i → données <= i).

A6 / A5 : tradingview/hmm_kalman_consensus_v6.pine / indicator.pine
B       : deriv/strategies/regime_momentum_strategy.py (classe d'origine appelée telle quelle)
C       : intelligence/signal_engine.py (score) + seuils de deriv/strategies/rise_fall.py
D       : crypto/strategies/HMMRegimeStrategy.py (freqtrade)

Les indicateurs Pine n'ont pas de règle de sortie : on leur applique la sortie
de B (stop 1,5 ATR, objectif 2,5 ATR), identique pour tous ceux qui n'en ont pas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from lab.engine import ExitRules, Signal

SORTIE_ATR = dict(sl_atr=1.5, tp_atr=2.5)


def _kalman_pine(close: np.ndarray, gain: float = 0.3) -> np.ndarray:
    k, e, out = close[0], 1.0, np.empty_like(close)
    for i, x in enumerate(close):
        kg = gain / (gain + e)
        k = k + kg * (x - k)
        e = (1.0 - kg) * e
        out[i] = k
    return out


def _pine_base(df):
    c, h, l = df["close"], df["high"], df["low"]
    kp = pd.Series(_kalman_pine(c.to_numpy(float)), index=df.index)
    ema_f, ema_s = c.ewm(span=20, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_r = atr / atr.rolling(50).mean()
    bull = (ema_f > ema_s) & (atr_r < 1.5)
    bear = (ema_f < ema_s) & (atr_r < 1.5)
    rsi = ta.momentum.RSIIndicator(c, 14).rsi()
    macd = ta.trend.MACD(c, 26, 12, 9)
    bb = ta.volatility.BollingerBands(c, 20, 2)
    return c, kp, bull, bear, rsi, macd.macd(), macd.macd_signal(), bb


def strat_A6(df, **_):
    c, kp, bull, bear, rsi, m, ms, bb = _pine_base(df)
    low, mid = bb.bollinger_lband(), bb.bollinger_mavg()
    score = (50 + np.where(rsi < 35, 20, np.where(rsi > 65, -20, 0))
             + np.where(m > ms, 15, -15)
             + np.where(c < kp * 0.99, 15, np.where(c > kp * 1.01, -15, 0))
             + np.where(c < low, 10, -5)
             + np.where(bull, 15, np.where(bear, -15, 0)))
    cross_up = (c > low) & (c.shift() <= low.shift())
    cross_dn = (c < mid) & (c.shift() >= mid.shift())
    buy, sell = (score >= 65) & cross_up, (score <= 35) & cross_dn
    sig = {i: Signal(+1, **SORTIE_ATR) for i in np.nonzero(buy.to_numpy())[0]}
    sig.update({i: Signal(-1, **SORTIE_ATR) for i in np.nonzero(sell.to_numpy())[0]})
    return sig, ExitRules(max_hold=48)


def strat_A5(df, **_):
    c, kp, bull, bear, rsi, m, ms, bb = _pine_base(df)
    low, up = bb.bollinger_lband(), bb.bollinger_hband()
    score = (50 + np.where(rsi < 35, 20, np.where(rsi > 65, -20, 0))
             + np.where(m > ms, 15, -15)
             + np.where(c < kp * 0.99, 15, np.where(c > kp * 1.01, -15, 0))
             + np.where(c < low, 10, np.where(c > up, -10, 0))
             + np.where(bull, 15, np.where(bear, -15, 0)))
    cross_up = (c > low) & (c.shift() <= low.shift())
    cross_dn = (c < up) & (c.shift() >= up.shift())
    buy, sell = (score >= 65) & cross_up, (score <= 35) & cross_dn
    sig = {i: Signal(+1, **SORTIE_ATR) for i in np.nonzero(buy.to_numpy())[0]}
    sig.update({i: Signal(-1, **SORTIE_ATR) for i in np.nonzero(sell.to_numpy())[0]})
    return sig, ExitRules(max_hold=48)


def strat_B(df, regimes, **_):
    """Appelle la classe d'origine RegimeMomentumStrategy bougie par bougie."""
    from deriv.strategies.regime_momentum_strategy import RegimeMomentumStrategy
    st, sig = RegimeMomentumStrategy(), {}
    reg, conf = regimes["regime"].to_numpy(), regimes["confiance"].to_numpy()
    for i in range(60, len(df)):
        s = st.analyse(df.iloc[max(0, i - 199):i + 1],
                       {"regime": reg[i], "confiance": float(conf[i])})
        if s is not None:
            sig[i] = Signal(+1, sl_atr=st.ATR_STOP_MULT, tp_atr=st.ATR_TARGET_MULT)
    return sig, ExitRules(max_hold=48)


def strat_C(df, regimes, seuil: int = 68, **_):
    """Score SignalEngine (HMM 20, Kalman 10, RSI 10, MACD 7, Bollinger 5)."""
    from intelligence.kalman_filter import KalmanPriceFilter
    c = df["close"]
    reg, conf = regimes["regime"].to_numpy(), regimes["confiance"].to_numpy()
    rsi = ta.momentum.RSIIndicator(c, 14).rsi().to_numpy()
    macd = ta.trend.MACD(c)
    md, ms = macd.macd().to_numpy(), macd.macd_signal().to_numpy()
    bb = ta.volatility.BollingerBands(c, 20)
    bl, bh = bb.bollinger_lband().to_numpy(), bb.bollinger_hband().to_numpy()
    cl = c.to_numpy(float)
    kf, sig = KalmanPriceFilter(), {}
    for i in range(len(df)):
        kx = kf.update(cl[i])          # même filtre, appliqué au fil de l'eau
        if i < 50:
            continue
        score = 50.0
        if reg[i] == "Bull" and conf[i] > 0.7:
            score += 20
        elif reg[i] == "Bear" and conf[i] > 0.7:
            score -= 20
        ecart = (cl[i] - kx) / kx
        score += 10 if ecart < -0.01 else -10 if ecart > 0.01 else 0
        score += 10 if rsi[i] < 35 else -10 if rsi[i] > 65 else 0
        score += 7 if md[i] > ms[i] else -7
        score += 5 if cl[i] < bl[i] else -5 if cl[i] > bh[i] else 0
        if score >= seuil:
            sig[i] = Signal(+1, **SORTIE_ATR)
        elif score <= 100 - seuil:
            sig[i] = Signal(-1, **SORTIE_ATR)
    return sig, ExitRules(max_hold=48)


def strat_D(df, regimes, minutes_par_bougie: int = 5, **_):
    """HMM + vitesse Kalman ; sortie en Range, ROI {0:3%, 30:1.5%, 60:0.5%}, stop -4%."""
    from intelligence.kalman_filter import KalmanPriceFilter
    kf = KalmanPriceFilter().filter(df["close"])   # filtre récursif : causal
    kp, vel = kf["kalman_price"].to_numpy(), kf["kalman_velocity"].to_numpy()
    reg, cl = regimes["regime"].to_numpy(), df["close"].to_numpy(float)
    long_ = (reg == "Bull") & (vel > 0) & (cl > kp)
    short = (reg == "Bear") & (vel < 0) & (cl < kp)
    sig = {i: Signal(+1) for i in np.nonzero(long_)[0]}
    sig.update({i: Signal(-1) for i in np.nonzero(short)[0]})
    b = lambda m: max(0, m // minutes_par_bougie)  # noqa: E731
    roi = {b(0): 0.03, b(30): 0.015, b(60): 0.005}
    return sig, ExitRules(max_hold=10_000, exit_signal=(reg == "Range"),
                          roi=roi, stoploss_pct=0.04)


STRATEGIES = {"A6": strat_A6, "A5": strat_A5, "B": strat_B, "C": strat_C, "D": strat_D}


def strat_B2(df, regimes, **_):
    """B corrigé : régime / confiance / ADX ne filtrent que l'ARMEMENT.

    Dans B d'origine, le moindre passage de la confiance HMM sous 70 % ou du
    régime hors Bull réinitialise la machine à états — or c'est précisément ce
    qui arrive pendant le pullback qu'elle attend, donc elle ne déclenche
    presque jamais. Ici, une fois armée, la machine suit le pullback puis la
    cassure (mêmes EMA 14/18/24, pullback 1-3, stop 1,5 ATR, objectif 2,5 ATR).
    """
    c, h, l = df["close"], df["high"], df["low"]
    e1, e2, e3 = (c.ewm(span=s, adjust=False).mean().to_numpy() for s in (14, 18, 24))
    adx = ta.trend.ADXIndicator(h, l, c, 14).adx().to_numpy()
    reg, conf = regimes["regime"].to_numpy(), regimes["confiance"].to_numpy()
    cl, hi = c.to_numpy(float), h.to_numpy(float)
    sig, phase, pb, win_hi = {}, "SCAN", 0, None
    for i in range(60, len(df)):
        if phase == "SCAN":
            if (reg[i] == "Bull" and conf[i] >= 0.70 and adx[i] >= 20
                    and e1[i] > e2[i] > e3[i]):
                phase, pb = "ARMED", 0
            continue
        if cl[i] < cl[i - 1]:                       # pullback
            pb += 1
            if pb > 3:
                phase = "SCAN"
                continue
            win_hi = hi[i - pb:i + 1].max()
            phase = "WINDOW"
        elif phase == "WINDOW" and cl[i] > win_hi:  # cassure
            sig[i] = Signal(+1, sl_atr=1.5, tp_atr=2.5)
            phase = "SCAN"
    return sig, ExitRules(max_hold=48)


STRATEGIES["B2"] = strat_B2
