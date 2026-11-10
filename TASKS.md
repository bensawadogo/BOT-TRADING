# TASKS — feuille de route

*Mise à jour : 10/11/2026. L'historique de l'ancien bot Deriv (missions 1 à 5, ensemble de
7 modèles, BOOM/CRASH) est dans le tag `archive/deriv-ensemble`.*

## Maintenant
- [ ] **Démo de 3 mois** : boucle relancée le 10/11/2026 (`logs/trend_bot.log`). Le 09/11, 13 cycles
  refusés (Algo Trading coupé) puis le PC a redémarré et tué le bot : aucun ordre à ce jour. Restent à faire :
  - [x] bouton Algo Trading activé dans MT5 (fait le 10/11 à 01 h 40 UTC) ;
  - [x] tâche planifiée « BOT-TRADING trend_bot » : ouverture de session + 06:45 UTC avec réveil du PC,
    un seul exemplaire (`scripts/lancer_bot.ps1`) ; le bot bloque la veille automatique tant qu'il tourne ;
  - [ ] de ton côté : connexion automatique Windows après redémarrage, minuteurs de réveil autorisés,
    mises à jour Windows hors 6 h-20 h UTC ;
  - [ ] jalon : 63 jours de cotation dans `data/trend_equity.csv`, suivis avec `main.py dashboard`.
- [ ] **Coûts de détention** : le compte Deriv sans swap existe mais facture des frais
  d'administration (2,8 %/an au lieu de 2,4 % avec swaps, `COMPTE_SANS_SWAP_2026-11-06.md`).
  Vérifier dans le Trader's hub s'il est ouvert à ta résidence, et chercher un courtier
  dont la grille de frais sur le forex majeur est plus basse (à lire avant de conclure).

## Fait le 10/11/2026
- [x] Vérification de la logique : 52 tests verts, causalité et live = backtest confirmés ; 3 écarts
  mineurs notés (retournement non reporté par le filtre d'annonces, lot minimum, veille PC).
  Mesure par sous-période : l'avantage est tout dans 1985-1995, zéro depuis 1996 ; `CLAUDE.md` et
  `TREND_BOT.md` corrigés pour l'afficher en premier (`VERIFICATION_LOGIQUE_2026-11-10.md`).
- [x] Balayage de 124 marchés Deriv aux coûts réels (`scripts/scan_marches.py`) : aucun avantage de
  tendance, 31 marchés perdants par les swaps, crypto = bêta (`docs/reports/SCAN_MARCHES_2026-11-10.md`).
  Le panier ne change pas.

## Fait le 06/11/2026
- [x] Protocole E0-E4 exécuté : aucune variante ne bat le bot d'origine, il reste en
  production (`docs/reports/ESSAIS_E0_E4_2026-11-06.md`).
- [x] Journal de décision : chaque ligne porte sa raison en clair (colonnes `signal_brut`,
  `exposition`, `raison`).

## Pistes ouvertes (à mesurer avant d'y croire)
- [x] **Signal de portage (carry)** mesuré le 10/11 (`lab/carry.py`, `PORTAGE_2026-11-10.md`) : chez
  Deriv aucune variante ne passe (dégonflé 0,80) ; sur futures sans swap, portage/vol Sharpe 0,63
  (dégonflé 0,99), décorrélé de la tendance (0,30). Le surcoût de swap Deriv coûte 0,22 de Sharpe
  à la tendance seule. Rien ne change pour le bot.
- [ ] Si la piste futures est retenue : backtest tendance + portage sur les **séries de contrats**
  (roulement, basis), coûts IBKR relevés, puis compte démo IBKR (`pysystemtrade` comme référence).
- [ ] Le même système sur **futures** (pas de swap) via Interactive Brokers, qui accepte le Burkina
  Faso : relever commissions, roulement, marge et lot minimum des micro-contrats, puis rejouer
  le backtest avec ces coûts. Capital réaliste 20-50 k$ (`VERIFICATION_LOGIQUE_2026-11-10.md` §5).

## Ensuite (seulement si la démo est conforme au backtest)
- [ ] Comparer chaque mois l'équité de la démo à celle du backtest sur la même période.
- [ ] Rejouer `scripts/couts_mt5.py tendance` une fois par mois (spreads et swaps changent).
- [ ] Compte réel : seulement avec de l'argent que tu peux perdre, `MT5_ALLOW_REAL=1`,
  et une équité suffisante pour le lot minimum (voir `docs/TREND_BOT.md`).

## Connu, non bloquant
- Pétrole et Nasdaq : historique MT5 depuis 01/2024 (≈ 720 jours ouvrés < 1 000) ; leur
  filtre de régime est inactif jusqu'à ce que l'historique suffise (le bot le signale).
- Le backtest ignore le portage (différentiel de taux) : il compare des prix au comptant
  et ne retire que la marge du courtier sur les swaps.

## Abandonné (ne pas relancer sans nouvel argument chiffré)
- Ajouter cryptos, actions, indices, métaux, ETF, agricoles ou paires croisées au panier :
  mesuré le 10/11/2026 sur 124 marchés, aucun avantage (`SCAN_MARCHES_2026-11-10.md`). NVDA et SOL
  (t ≈ 2) sont les 2 tirages attendus du hasard sur 124 : ne pas les retester isolément.
- Changer l'heure du cycle (session américaine) ou ajouter un délai de confirmation du signal :
  mesurés le 10/11/2026, spreads quasi fixes et confirmation toujours perdante
  (`HEURE_ET_CONFIRMATION_2026-11-10.md`).
- Fibonacci en timing d'entrée, tendance EWMAC, désaccord lent/rapide, régime adouci :
  mesurés le 06/11/2026, aucun gain (`ESSAIS_E0_E4_2026-11-06.md`).
- Bot Deriv ensemble 7 modèles / options binaires / BOOM-CRASH : aucun edge (audits 14/09
  et 08/10/2026).
- Scalping : le spread dépasse l'avantage mesurable sur tous les marchés testés
  (`docs/reports/ANALYSE_SCALPING_2026-11-02.md`).
