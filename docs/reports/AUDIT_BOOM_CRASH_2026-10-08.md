# Audit BOOM/CRASH sur données réelles — 08/10/2026

Données : 100 000 bougies M1 réelles par indice (30/07 → 08/10/2026, aucun trou),
téléchargées via le nouvel endpoint public Deriv. Coûts réels relevés par
`proposal` le jour même. Scripts : `scripts/fetch_candles_public.py`,
`scripts/audit_boom_crash.py` (reproductibles).

## 1. Le contrat Rise/Fall n'existe pas sur Boom/Crash
`contracts_for` sur BOOM300N, BOOM500, BOOM1000, CRASH500, CRASH1000 :
**ACCU, MULTUP, MULTDOWN uniquement**. CALL/PUT existent sur R_75, 1HZ100V…
→ la stratégie « drift post-spike en Rise/Fall » ne peut pas être exécutée.

## 2. Le signal spike n'a aucun pouvoir prédictif
| Indice | Taux de base (minute quelconque) | Après spike (règle live) | p(signal > base) |
|---|---|---|---|
| BOOM500 | 58,2 % | 57,0 % (1 049 trades) | 0,80 |
| CRASH500 | 58,7 % | 60,1 % (1 050 trades) | 0,19 |

Les 57-60 % de « réussite » mesurent la dérive de l'indice, pas le signal
(même artefact que l'audit CRASH500 du 14/09). Les 62,5 % / p = 0,093 du
13/09 étaient testés contre un seuil fixe de 55,6 %, jamais contre ce taux de base.

## 3. Avec les Multipliers (seuls contrats possibles), tout perd
Mise 10 $, multiplicateurs ×100 à ×400, durées 5 à 60 min, entrée après
spike ou au hasard : **PnL moyen négatif dans toutes les configurations**
(≈ −0,03 à −0,63 $/trade). La dérive est compensée par les spikes contraires,
qui sautent au-delà du stop-out ; la commission fait le reste.

## Conclusion
Pas d'edge exploitable sur BOOM/CRASH avec cette approche. Ne pas trader cette
stratégie, ni en démo ni en réel. Optimiser ses paramètres sur ces données
reviendrait à sur-ajuster du bruit.

## À noter pour la suite
- L'ancien endpoint `wss://ws.derivws.com/websockets/v3?app_id=1089` a répondu
  HTTP 520 ; le nouvel API est `api.derivws.com` (données publiques :
  `/trading/v1/options/ws/public`, trading : OTP via REST + App ID enregistré).
  Le client du bot devra être migré avant tout trading.
