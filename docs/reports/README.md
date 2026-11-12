# Rapports datés

Chaque rapport fige une conclusion à sa date ; il n'est pas réécrit après coup. Le code
qu'ils citent sous `deriv/`, `intelligence/`, `crypto/`, `tradingview/`, `lab/regime.py`,
`lab/strategies.py`, `lab/run.py` ou `scripts/` (hors `fetch_fred.py` et `couts_mt5.py`)
est archivé : `git checkout archive/deriv-ensemble`.

| Date | Rapport | Conclusion |
|---|---|---|
| 08/10/2026 | `AUDIT_BOOM_CRASH_2026-10-08.md` | BOOM/CRASH : aucun edge, contrat Rise/Fall inexistant |
| 08/10/2026 | `BACKTEST_ALGOS_TENDANCE_2026-10-08.md` | Algos intraday du dépôt (M5, H1, H4) : aucun edge après coûts |
| 08/10/2026 | `SYSTEME_REGIME_FIBO_2026-10-08.md` | HMM causal + Fibonacci : pas d'amélioration exploitable |
| 08/10/2026 | `TREND_BOT_BACKTEST_2026-10-08.md` | Bot de tendance : Sharpe 0,55 sur 1985-2026, faible depuis 2008 |
| 02/11/2026 | `ANALYSE_SCALPING_2026-11-02.md` | Scalping : le spread dépasse l'avantage mesurable partout |
| 02/11/2026 | `TREND_BOT_COUTS_REELS_2026-11-02.md` | Bot de tendance aux coûts réels de Deriv MT5 (spreads, swaps) |
| 03/11/2026 | `RECHERCHE_BOTS_QUI_MARCHENT_2026-11-03.md` | Ce que font les systèmes qui gagnent, et le plan pour s'en inspirer |
| 03/11/2026 | `COMBINER_LES_ALGOS_2026-11-03.md` | Tendance + régime + Fibonacci : un rôle par algo, protocole de 5 essais |
| 06/11/2026 | `ESSAIS_E0_E4_2026-11-06.md` | Les 5 essais de combinaison : aucun ne bat le bot d'origine, il reste en production |
| 06/11/2026 | `COMPTE_SANS_SWAP_2026-11-06.md` | Compte Deriv sans swap : frais d'administration, 2,8 %/an et non 5,1 % |
| 10/11/2026 | `VERIFICATION_LOGIQUE_2026-11-10.md` | Logique vérifiée (52 tests, causalité, live = backtest) ; l'avantage est tout dans 1985-1995, zéro depuis 1996 ; pistes extérieures (futures IBKR) |
| 10/11/2026 | `PORTAGE_2026-11-10.md` | Signal de portage : rien chez Deriv (Sharpe dégonflé 0,80) ; sur futures sans swap, portage/vol Sharpe 0,63, dégonflé 0,99 — premier résultat au-dessus du seuil, support non détenu |
| 10/11/2026 | `HEURE_ET_CONFIRMATION_2026-11-10.md` | Spread Deriv quasi fixe de 7 h à 16 h (cycle reste à 7 h, éviter 20 h) ; délai de confirmation 3/5/10 jours : toujours moins bien que le bot (P > T 0,03-0,07), classé |
| 10/11/2026 | `SCAN_MARCHES_2026-11-10.md` | 124 marchés Deriv (crypto, actions, indices, métaux, ETF, agricoles, croisés) : aucun avantage de tendance, 31 perdants par les swaps |
| 12/11/2026 | `STOPS_2026-11-12.md` | Stops par position : les larges ne se déclenchent presque jamais, le serré gagne 0,03 de Sharpe sans être significatif (P > T 0,81) et ne réduit pas les profits rendus (54 %) ; stop catastrophe chez le courtier à ajouter avant le réel |
