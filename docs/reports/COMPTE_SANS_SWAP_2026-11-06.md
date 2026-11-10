# Le compte sans swap de Deriv n'est pas gratuit : 2,8 %/an, pas 5,1 % (06/11/2026)

*Le rapport du 02/11 estimait 5,1 %/an « sans swap » en supposant un coût nul. Deriv
facture en réalité des frais d'administration sur les positions tenues. Grille lue sur
`blog.deriv.com/blog/admin-fees-for-deriv-mt5-swap-free-accounts` (06/11/2026), modélisée
dans `trend_bot/data.py` (`ADMIN_SANS_SWAP`) et `trend_bot/backtest.py` (`admin=`).*

## 1. Ce que Deriv facture sur un compte MT5 sans swap

- **Frais d'administration par lot et par nuit**, dus une fois la position tenue plus de
  **15 jours** (instruments financiers) dans le même sens ; 5 jours sur les indices dérivés.
  Aucun frais à la fermeture. Fermer une partie de la position ne remet pas le compteur à zéro.
- **Un seul sens est facturé** pour beaucoup de paires : EURUSD long 8,80 $/lot/nuit, court 0 ;
  USDJPY long 0, court 14,89 $ ; US Tech 100 long 4,90 $, court 0.
- Le compte sans swap existe pour le forex et le Nasdaq (US Tech 100 est dans la grille).
  Il n'est pas proposé aux résidents de l'UE. Rien n'indique, dans les sources lues, s'il
  l'est au Burkina Faso : la vérification se fait dans le Trader's hub ou auprès du support.

En fraction annuelle du notionnel (cours actuels), pour les 11 marchés du bot :

| Marché | Frais admin long | Frais admin court | Swap standard long | Swap standard court |
|---|---|---|---|---|
| EURUSD | 2,88 % | 0 | −3,73 % | +0,87 % |
| GBPUSD | 0,51 % | 0,66 % | −1,48 % | −1,12 % |
| USDJPY | 0 | 5,43 % | +2,30 % | −5,08 % |
| AUDUSD | 0 | 1,48 % | −1,44 % | −3,05 % |
| USDCAD | 0 | 2,63 % | +0,70 % | −3,21 % |
| USDCHF | 0 | 4,40 % | +3,08 % | −7,15 % |
| NZDUSD | 4,61 % | 1,41 % | −4,16 % | −1,66 % |
| USDMXN | 3,26 % | 0 | −0,48 % | +0,03 % |
| USDNOK | 2,41 % | 1,49 % | −2,82 % | −2,00 % |
| USDSEK | 0,01 % | 4,03 % | +0,65 % | −5,71 % |
| NAS100 | 6,82 % | 0 | −6,16 % | +1,89 % |

Le frais d'administration reprend en gros le côté coûteux du swap standard, sans jamais
rien créditer de l'autre côté.

## 2. Effet sur le bot (E0, 11 marchés, spreads réels, 1985-2026)

| Scénario | CAGR | Sharpe | DD max | Sharpe 2008-26 |
|---|---|---|---|---|
| Compte standard (spreads + swaps, rapport du 02/11) | 2,4 % | 0,33 | 29,4 % | 0,10 |
| **Sans swap + frais admin Deriv (grâce 15 jours)** | **2,8 %** | **0,36** | 25,6 % | **0,17** |
| Sans swap + frais admin, sans délai de grâce | 1,5 % | 0,23 | 29,2 % | 0,03 |
| Sans swap, frais admin supposés nuls (ancien chiffre) | 5,1 % | 0,64 | 19,0 % | 0,33 |
| E1 (tendance continue) + frais admin, grâce 15 jours | 1,3 % | 0,19 | 26,1 % | 0,10 |

Le compte sans swap rapporte à peine plus que le compte standard (+0,4 point par an, écart
indiscernable du bruit). Le délai de grâce de 15 jours compte beaucoup, mais une tendance
dure des mois : la plupart des jours de détention sont facturés. E1, qui ne change presque
pas de position, ne s'en sort pas mieux.

## 3. Conséquences

- **Ce que le bot gagne net aux coûts de Deriv : environ 2,4 à 2,8 %/an**, avec un Sharpe
  de 0,10 à 0,17 sur 2008-2026. Sur 1 000 $, cela fait ≈ 25 $ par an avant toute perte
  d'exécution. Ce n'est pas un revenu qui finance un abonnement ou un SaaS.
- **Le coût est dans l'instrument, pas dans le signal.** Les essais E0-E4 ont montré
  qu'aucune combinaison d'algos ne change cela.
- **Autres courtiers (non vérifié)** : la plupart des comptes « islamiques » MT5 remplacent
  aussi le swap par des frais (ACY : frais fixes après 4 jours ; AvaTrade : 5 jours ;
  OneRoyal : 10 jours ouvrés ; XBTFX : frais égaux au swap après 15 jours). Admirals dit
  que le swap est supprimé sur les indices, avec un petit frais sur les paires exotiques.
  Ces informations viennent de pages commerciales de courtiers : la grille de chacun doit
  être lue avant de conclure.
- **Pistes à tester, pas à croire** : (1) un courtier sans frais sur le forex majeur ;
  (2) un marché où le frais est nul dans le sens de la position (la grille est asymétrique,
  mais l'exploiter serait choisir le sens d'après le coût : à mesurer avant tout).

## 4. Limites

- Notionnel d'un lot : cours actuels appliqués à tout l'historique ; Nasdaq supposé à 1 ×
  l'indice (cohérent avec le swap standard relevé, mais à confirmer sur la spécification MT5).
- Frais comptés par jour de bourse (annuel ÷ 252) : le week-end et le mercredi (triple) ne
  sont pas modélisés.
- La grille lue date de la page du blog (publication indiquée 25/05/2026) ; elle peut changer.
- Le backtest ignore le portage (différentiel de taux), comme les rapports précédents.
