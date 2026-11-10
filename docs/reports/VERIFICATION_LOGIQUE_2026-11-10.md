# Vérification de la logique du projet, et d'où vient le résultat (10/11/2026)

*Relecture complète de `trend_bot/` (signaux, risque, portefeuille, runner, backtest, courtier
MT5), suite de tests (52/52, 3 min 47), et une mesure qui manquait : le panier de production
par sous-période, avec et sans filtre de régime, aux coûts réels de Deriv (spreads + surcoût
de swap, `data/couts_mt5.json` du 02/11). Journal : `logs/regime_1996.log`.*

## 1. Ce qui est correct

| Point vérifié | Où | Constat |
|---|---|---|
| Causalité des signaux | `signals.py` | signes 1/3/12 mois, vol EWMA : clôtures ≤ t uniquement ; NaN tant que l'horizon manque |
| Causalité du régime | `risk.py` | entraînement sur les 1 000 jours *avant* le 1er du mois, états *en ligne* le mois suivant ; étiquetage sur la fenêtre d'entraînement. Tests `regime_series_causal`, `regime_now_egale_le_backtest` |
| Live = backtest | `runner.py` / `backtest.py` | mêmes fonctions, même calendrier lundi-vendredi, régime par marché sur sa série ; test `live_egale_backtest` |
| Bougies terminées | `brokers/mt5.py` | `copy_rates_from_pos(…, 1, n)` : la bougie du jour (position 0) est exclue |
| Rendement t → t+1 | `backtest.py` | poids décidés à la clôture t, rendement de t+1 ; frais sur chaque variation de poids |
| Garde-fous | `runner.py`, `mt5.py` | préflight Algo Trading ; compte réel refusé sans `MT5_ALLOW_REAL` ; positions filtrées par numéro magique ; netting et hedging ; lot jamais sur-dimensionné ; coupe-circuit persistant (`peak` 10 000, équité 9 761 : drawdown 2,4 %) |
| Journal | `runner.py` | chaque décision porte sa raison en clair |

## 2. Trois écarts mineurs, notés, non corrigés

1. **Retournement et annonces.** La règle d'attente teste `|cible| > |actuel|` : un retournement
   exact (+0,3 → −0,3) n'est pas reporté bien qu'il ouvre une exposition nouvelle. Cas rare ;
   si on y tient, `≥` suffit, avec son test.
2. **Lot minimum.** Le backtest trade des poids continus ; à 9 761 $, plusieurs marchés passent
   sous le lot minimum et ne sont pas tradés (NAS100 : ≈ 73 000 $ d'équité nécessaires). Le live
   est un sous-ensemble du backtest. C'est le capital, pas le code (`docs/TREND_BOT.md`).
3. **Machine.** Le 09/11 : 13 cycles refusés (Algo Trading coupé, garde-fou correct), un trou de
   1 h 54 (11 h 33 → 13 h 27, veille probable), puis redémarrage à 15 h 40 qui a tué le bot.
   Relancé le 10/11 à 01 h 30 UTC ; Algo Trading activé à 01 h 40.

## 3. D'où vient le résultat : tout est dans 1985-1995

Panier de production (11 marchés FRED, forex + Nasdaq), coûts Deriv, un seul backtest découpé :

| Période | Avec régime (production) | | | Sans régime | | | P(avec > sans) |
|---|---|---|---|---|---|---|---|
| | Sharpe | CAGR | DD | Sharpe | CAGR | DD | |
| 1985-2026 (chiffre affiché jusqu'ici) | +0,28 | +2,0 % | 30 % | +0,22 | +1,5 % | 42 % | 0,82 |
| **1985-1995** | **+0,83** | +7,7 % | 17 % | +0,86 | +8,4 % | 16 % | |
| **1996-2026** | **+0,06** | +0,1 % | 30 % | −0,03 | −0,6 % | 42 % | 0,90 |
| 2008-2026 | +0,04 | 0,0 % | 30 % | −0,12 | −1,3 % | 40 % | 0,96 |
| 2017-2026 | −0,17 | −1,5 % | 19 % | −0,32 | −2,6 % | 28 % | |

(P = part des rééchantillonnages par blocs de 21 jours où « avec » bat « sans ».)

- **L'avantage mesuré est entièrement dans 1985-1995.** Depuis 30 ans, aux coûts de Deriv, le
  bot est à zéro (+0,06), et négatif depuis 2017.
- **Le filtre de régime fait son travail de frein, pas de moteur** : il réduit le drawdown de
  42 % à 30 % et évite une partie des pertes (P = 0,90 à 0,96 qu'il aide depuis 1996), mais
  il ne crée pas de rendement : +0,1 %/an depuis 1996.
- Les chiffres diffèrent un peu des rapports précédents (0,33 et 0,10 le 06/11) : ici le régime
  vient du cache `data/regimes_1985.pkl` aligné sur le calendrier commun. L'ordre de grandeur
  et la conclusion sont les mêmes.

## 4. Le défaut de cohérence du projet

La règle 5 de `CLAUDE.md` exige qu'une stratégie batte le hasard, aux coûts réels, **sur une
période jamais utilisée pour la régler**. La stratégie de production ne le fait pas sur
1996-2026. Les documents affichaient pourtant « Sharpe 0,54 » puis « 0,33 » en tête et
reléguaient le 2008-2026 en fin de tableau : l'ordre de présentation était l'inverse de
l'ordre d'importance. Ce n'est pas une erreur de code ; c'est une erreur de présentation
et de décision.

**Conséquence** : la démo de 3 mois valide la *mécanique* (ordres, journal, garde-fous,
coûts réels). Elle ne peut pas valider un revenu, puisque l'espérance mesurée à ces coûts est
nulle. `CLAUDE.md` et `docs/TREND_BOT.md` sont corrigés ce jour pour afficher d'abord
1996-2026.

## 5. Autres moyens d'arriver au but (recherche GitHub et forums, 10/11/2026)

Le but : un revenu régulier pour financer un SaaS. Ce que l'extérieur dit, sources à l'appui :

1. **Le même système sur futures, pas sur CFD.** `pysystemtrade` (Rob Carver, ex-AHL, open
   source, ≈ 40 futures via Interactive Brokers, résultats publiés depuis 2014) est la référence
   de la tendance diversifiée en Python. Les futures n'ont **pas de swap** : le coût de
   détention qui annule notre bot (`COMPTE_SANS_SWAP_2026-11-06.md`, `SCAN_MARCHES_2026-11-10.md`)
   disparaît ; il reste commissions et roulement. **Interactive Brokers accepte les résidents
   du Burkina Faso** (liste officielle « Available Countries », lue le 10/11/2026).
   Limite, dite par Carver lui-même (« Diversification and small account size », 2016) : on
   ne peut pas tenir une fraction de contrat ; il recommande ≥ 4 contrats au signal maximal,
   sinon le système devient « binaire arbitraire ». Les *micro* futures (1/10 d'un E-mini)
   abaissent le seuil ; les courtiers citent 1 500 à 2 500 $ pour ouvrir, mais une tendance
   diversifiée à 10 %/an de vol demande plutôt 20 à 50 k$. Avec 10 k$ : 3 à 5 micro-contrats,
   moins diversifié que le backtest, sans swap. **À mesurer avant tout** : coûts IBKR réels,
   roulement, marge.
2. **Ce que valent les systèmes de tendance aujourd'hui** : l'indice SG Trend (les gérants
   professionnels) a fait +27,3 % en 2022 et **+2,4 % en 2025** (algotrader.ch, vérifié
   octobre 2026). Un bon système de tendance, bien exécuté, rend quelques pourcents par an
   en moyenne, avec des années à −10 %.
3. **Ce que disent les forums de trading algorithmique** (r/algotrading, tradealgo.com,
   2026) : ≈ 80 % des stratégies bonnes en backtest échouent en réel ; les survivants font 8 à
   15 %/an ; le biais du survivant cache la majorité des échecs. Un projet GitHub très proche
   du nôtre (Aurum, or XAUUSD, walk-forward, Sharpe dégonflé, filtre de pré-enregistrement)
   conclut : « aucun edge statistiquement démontré ». Notre résultat est la norme, pas une
   anomalie.
4. **Ce qui ne marche pas, déjà mesuré ici** : options binaires et indices synthétiques Deriv
   (audits 14/09 et 08/10), scalping (02/11), combinaisons d'algos (06/11), 124 autres marchés
   (10/11).

**Conclusion pratique** : aucun de ces chemins ne produit un revenu régulier sur 10 k$. L'ordre
de grandeur honnête est 0 à 3 %/an avec des années perdantes. Le revenu du SaaS doit venir
d'ailleurs ; le bot reste un projet de recherche propre, un journal de décisions et un
portfolio technique. La seule piste qui change la donne (futures sans swap chez IBKR) demande
plus de capital et une mesure préalable des coûts, comme toute piste de ce projet.

## 5 bis. Chiffres relevés pour la piste « futures » et le signal de portage (10/11/2026, soir)

Pages officielles lues le 10/11/2026 (à revérifier à l'ouverture d'un compte) :

| Poste | Valeur relevée | Source |
|---|---|---|
| Commission IBKR, micro FX (M6E, M6A, M6B…) et or 1 oz | 0,15 $ / contrat (≤ 1 000 contrats/mois), + frais de bourse | interactivebrokers.com, Commissions Futures |
| Commission IBKR, micro DAX / micro ESTX50 | 0,25 € / contrat | idem |
| Marge de nuit M6E (micro EUR/USD, 12 500 €) | ≈ 350 à 450 $ | IBKR, Futures Margin Requirements (deux relevés) |
| Marge de nuit MES (micro S&P 500) | ≈ 3 400 à 4 000 $ | idem |
| Marge de nuit MNQ (micro Nasdaq 100) | ≈ 6 300 à 7 500 $ | idem |
| Swap / financement de nuit | **aucun** (le portage est dans le prix du contrat et le roulement trimestriel) | nature du contrat |

Lecture : un micro EUR/USD vaut ≈ 14 500 $ de notionnel pour 0,15 $ de commission par passage,
contre, chez Deriv, un spread comparable **plus** 2,9 à 3,7 %/an de détention. Sur 10 k$, le
bot tiendrait 3 à 5 micro-contrats de change et aucun indice (MNQ seul demande 6 à 7 k$ de
marge) : diversification réduite, mais coût de détention nul. **Rien de tout cela n'est encore
un backtest** : il faut les séries des contrats (roulement inclus), les coûts ci-dessus, et le
protocole du 06/11.

**Signal de portage (carry)** : Carver et les gérants de tendance l'ajoutent comme second
signal (Sharpe en hausse, asymétrie en baisse ; CME « An Introduction to Global Carry », blog
Carver). Les taux interbancaires 3 mois des devises du panier sont sur FRED
(`IR3TIB01EZM156N`, `IR3TIB01JPM156N`, `IR3TIB01UKM156N`, `IR3TIB01AUM156N`, `IR3TIB01CAM156N`,
`IR3TIB01CHM156N`, `IR3TIB01NZM156N`, `IR3TIB01MXM156N`) : un test est possible dans `lab/` sans
toucher au bot. Réserve : chez Deriv, le portage est déjà prélevé en swap, souvent au-delà du
différentiel ; le signal n'a de sens que sur un support sans swap (futures) ou si le swap est
inférieur au différentiel, à vérifier marché par marché.

## 6. Limites de cette note

- Les sous-périodes sont choisies après coup (1996, 2008, 2017) : elles illustrent, elles ne
  testent pas. Le point robuste est que **toute** période commençant après 1995 donne ≈ 0.
- Coûts d'aujourd'hui appliqués à tout l'historique ; portage (différentiel de taux) ignoré.
- Les sources externes sont des pages web lues le 10/11/2026 : l'éligibilité IBKR et ses
  coûts se vérifient à l'ouverture de compte, pas sur une page.
