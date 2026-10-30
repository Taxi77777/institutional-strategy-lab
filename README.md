# 🪖 Institutional Strategy Lab — l'armée des 20 spécialistes

**Recherche automatique de stratégies intraday à 80–95 % de réussite, validées sur des données jamais vues.
Forex, or et crypto · compatible avec tous les courtiers · signaux Telegram · 100 % gratuit.**

## L'idée

Les institutions ont leurs algorithmes ; voici le tien, construit avec leur méthode :

1. **20 spécialistes**, chacun avec sa famille d'indicateurs : Tendance · Ichimoku · Momentum · Oscillateurs ·
   Volatilité · Bandes · Volume · VWAP · Volume profile · Fibonacci · AMD et sessions · Structure de marché ·
   Chandeliers · Statisticien · Mathématicien · Retour à la moyenne · Horaires · Inter-marchés · Macro · Carnet d'ordres.
2. **Plus de 600 variantes d'indicateurs** (divergences RSI / MACD / OBV, Supertrend, Hull, régressions, Hurst,
   entropie, sweeps de liquidité, avance du marché meneur BTC / EURUSD / SPY…), découpées en **près de 3 000 conditions**, combinées par 2 et par 3 :
   **des millions de combinaisons testées** à chaque recherche.
3. **Méthode anti-hasard des fonds quantitatifs** :
   - apprentissage sur les 60 % les plus anciens, validation sur les 20 % suivants, **verdict sur les 20 % les plus récents, jamais vus** ;
   - correction des tests multiples (Benjamini-Hochberg) : sur des millions d'essais, le hasard « trouve » toujours quelque chose ;
   - trades sans chevauchement, frais inclus, exécution prudente ;
   - **purge et embargo** entre les périodes (López de Prado) : aucun trade appris ne « voit » la période suivante ;
   - **robustesse** : gagnante sur la majorité des actifs pris un par un, et ≥ 75 % de réussite sur chaque moitié de la
     période jamais vue (stabilité dans le temps) ;
   - test de contrôle : sur des prix au hasard, le labo ne doit **rien** trouver (vérifié par les tests automatiques).
4. **Règle de sélection** : taux de réussite **≥ 80 %** (objectif 95 %) **et** gain moyen positif après frais.
   Un taux élevé avec un total perdant ne passe pas.
5. **L'armée vote** : un signal part seulement si au moins **3 spécialistes** ont une stratégie validée active dans le
   même sens, sans majorité contraire. Le vote lui-même est testé sur les données jamais vues.
6. **Setup élite** : une stratégie prouvée à **≥ 90 % sur ≥ 100 trades jamais vus** (borne basse prudente ≥ 85 %) peut signaler seule, sans attendre 3 spécialistes.
   Tant qu'aucune stratégie ne passe, l'armée reste silencieuse. C'est voulu.
   En direct, l'armée lit les mêmes données que la recherche (Coinbase pour la crypto, volume forex neutralisé) : aucun écart entre ce qui a été testé et ce qui est surveillé.
7. **L'Inventeur (21ᵉ agent)** cherche en permanence des idées : à chaque recherche il invente ~150 nouveaux indicateurs
   en combinant ceux des spécialistes (accords, désaccords, accélérations, croisements, rangs), les fait tester avec les
   mêmes règles, garde en mémoire (`ideas.json`) celles qui gagnent et les fait évoluer la semaine suivante
   (mutation, croisement), comme une sélection naturelle.
8. **Multi-unités (23ᵉ agent)** : tendance 1 h, 4 h et journalière (bougies clôturées uniquement), que chaque spécialiste peut ajouter à ses conditions — l'entrée en 15 min dans le sens de la tendance supérieure, comme les desks institutionnels.
10. **Sources publiques reconnues (24ᵉ à 27ᵉ agents)** : les 61 figures de chandeliers de TA-Lib, les facteurs Alpha158 de Microsoft Qlib, les « 101 Formulaic Alphas » de WorldQuant (version série temporelle) et les Smart Money Concepts (swings confirmés, BOS / CHoCH, order blocks, FVG, liquidité) — formules réécrites, sans regard vers le futur (testé).
9. **Favoris TradingView (22ᵉ agent)** : WaveTrend (VuManChu Cipher B), Squeeze Momentum, UT Bot, Chandelier Exit, QQE,
   Range Filter, SSL Channel, Fisher Transform, Schaff Trend Cycle, Kalman, Fair Value Gaps, Nadaraya-Watson (version qui
   ne repeint pas) — réécrits à partir de leurs formules publiques (aucune copie de script).
9. **L'algorithme final** = le comité des stratégies validées et leur vote, régénéré chaque semaine dans
   `strategies.json` : il devient plus fort à mesure que l'Inventeur trouve de meilleures idées et que les stratégies
   qui faiblissent en réel sont retirées.

## Ce que tu reçois sur Telegram

- 🪖 **Conseil des 20** à 8 h, 14 h et 20 h (Paris) : pour BTC, ETH, SOL, EURUSD, GBPUSD, USDJPY, or et argent,
  combien de spécialistes sont haussiers ou baissiers, et qui n'est pas d'accord.
- 🟢🔴 **Signal** (contrôle toutes les 5 minutes, **bougies 15 minutes uniquement** ; forex : historique Dukascopy + prix interbancaires temps réel TrueFX) : catégorie, ACHETER / VENDRE, entrée au prix du marché, sortie de sécurité,
  prise de gain et heure de fermeture **en prix et en %** (valable chez n'importe quel courtier), spécialistes d'accord et
  pas d'accord, taux de réussite prouvé, 10 000 simulations Monte Carlo.
- ✅🛑 **Résultat** de chaque signal et bilan réel de l'armée. Une stratégie qui tombe sous 70 % en réel (après 20 trades)
  est retirée automatiquement.

## Données (gratuites, sans inscription)

Recherche **et signaux en direct** : **Dukascopy** pour tout le forex et les métaux (bougies 1 min regroupées en 5 min, 15 min, 30 min et 1 h, 3 ans en 1 h, avec le volume ; vérifié : Yahoo produisait de faux setups forex), **Coinbase** (crypto, 8 mois en
15 min, 3 ans en 1 h), Yahoo Finance (forex 1 h sur 2 ans, et en direct). En 5, 15 et 30 min, une stratégie doit réussir sur au moins
**200 trades jamais vus**. Carnets d'ordres publics Coinbase et Bitstamp (crypto, en
direct uniquement : aucun historique gratuit n'existe), S&P 500 et VIX pour le régime de marché.

## Installation (déjà faite)

- **Recherche** : *Actions* → **Labo - recherche de strategies** (**tourne sans arrêt** : chaque recherche relance la suivante avec de nouvelles idées ; mettre la variable `CONTINU` à `stop` pour l'arrêter).
  Publie `strategies.json` (l'algorithme) et `ideas.json` (la mémoire de l'Inventeur).
- **Armée en direct** : tourne depuis le dépôt `crypto-signals-timesfm` (workflow *Armee des 20*), qui a déjà le jeton
  Telegram ; elle recharge chaque heure les stratégies publiées ici. Le workflow `live.yml` de ce dépôt reste disponible à
  la main si on y ajoute le secret `TELEGRAM_BOT_TOKEN`.

Réglages (variables GitHub, facultatives) : `MIN_WIN` (0.80), `CONSENSUS` (3), `BEAM` (40), `TIMEFRAMES` (15M,5M,30M,1H).

## ⚠️ Honnêteté

Un taux de réussite de 80–95 % s'obtient avec un objectif proche et un stop plus éloigné : chaque gain est petit, une
perte en coûte plusieurs. C'est pourquoi le labo exige aussi un **gain total positif**. Les résultats passés, même
validés hors échantillon, ne garantissent rien. Signaux uniquement, aucun ordre, pas un conseil financier.
