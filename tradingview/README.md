# Trading Algo Africa — Indicateur TradingView (Pine Script v5)

Ce dossier contient l'implémentation de la stratégie de consensus (HMM + Kalman + RSI/MACD/Bollinger) directement sur **TradingView**.

## Fichier : `indicator.pine`

### Fonctionnalités intégrées :
1. **Filtre de Kalman (1D)** : Tracé en bleu lissant le bruit de marché.
2. **Proxy Régime HMM** : Coloration du fond du graphique :
   - Vert transparent : Régime **Bull** (EMA20 > EMA50 et volatilité maîtrisée).
   - Rouge transparent : Régime **Bear** (EMA20 < EMA50).
   - Gris transparent : Régime **Range**.
3. **Score Consensus (0-100)** :
   - RSI 14 (±20 pts)
   - MACD (±15 pts)
   - Écart Kalman ±1% (±15 pts)
   - Position Bandes de Bollinger (±10 pts)
   - Régime de tendance (±15 pts)
4. **Tableau de bord sur graphique** (en haut à droite) :
   - Régime de marché en direct
   - Score consensus actuel
   - Valeur RSI 14
   - Écart % au prix Kalman
   - Signal actif (BUY, SELL, WAIT)
5. **Alertes TradingView** :
   - `BUY` et `SELL` configurables pour déclencher des alertes visuelles, sonores ou des webhooks.

## Guide d'installation sur TradingView :
1. Ouvrir [TradingView.com](https://www.tradingview.com) et charger un graphique (ex: `BTCUSDT`, ou index synthétique Deriv si disponible).
2. Cliquer sur l'onglet **Pine Editor** en bas de l'écran.
3. Créer un nouveau script et copier-coller le contenu de [indicator.pine](file:///c:/BOT-TRADING/tradingview/indicator.pine).
4. Cliquer sur **Add to chart** (Ajouter au graphique).
5. Pour configurer des alertes :
   - Faire un clic droit sur le graphique -> **Add Alert**.
   - Choisir la condition `Trading Algo Africa — HMM+Kalman+Consensus` et sélectionner `BUY` ou `SELL`.
