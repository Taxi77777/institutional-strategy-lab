# 🪖 Institutional Strategy Lab — l'armée des spécialistes

**Recherche automatique et continue de stratégies 1 heure à plus de 92 % de réussite, objectif 99 à 100 %, prouvées sur des données jamais vues.
Forex, or, argent et crypto · compatible avec tous les courtiers · signaux Telegram · 100 % gratuit.**

## Le principe

1. **Des spécialistes**, chacun avec sa famille d'indicateurs (~1 000 variantes au total) :
   Tendance · Ichimoku · Momentum · Oscillateurs · Volatilité · Bandes · Volume · VWAP · Volume profile · Fibonacci ·
   AMD et sessions · Structure de marché · Chandeliers · Statisticien · Mathématicien · Retour à la moyenne · Horaires ·
   Inter-marchés · Macro · Carnet d'ordres · Favoris TradingView · Multi-unités (1 h / 4 h / jour) ·
   Chandeliers TA-Lib (61 figures) · Alpha158 (Microsoft Qlib) · 101 Alphas (WorldQuant) · Smart Money Concepts ·
   Dérivés crypto (financement des perpétuels) · Chiffres ronds · Volatilité du week-end · Suite TA-Lib complète (58) ·
   Concepts des maîtres (Hosoda, Sasaki, Péloille, Steidlmayer, Dalton, Shannon, Wyckoff, journée de tendance).
2. **Le noyau Ichimoku** : des stratégies qui partent toujours d'un signal Ichimoku — passé (Chikou), présent (Tenkan,
   Kijun, prix face au nuage), futur (nuage projeté, déjà connu à la clôture) — confirmé par 1 ou 2 conditions des
   meilleurs autres spécialistes.
3. **L'Inventeur** crée ~250 nouveaux indicateurs à chaque recherche (accords, écarts, accélérations…), garde ceux qui
   gagnent (`ideas.json`) et les fait évoluer.
4. **L'Éclaireur** fait chaque jour la veille de GitHub et d'arXiv (`veille.json`) : il oriente l'Inventeur vers les
   familles qui montent et envoie un résumé sur Telegram. Il ne télécharge ni n'exécute jamais de code. Les idées
   étudiées et la décision du conseil sont notées dans `eclaireur_integre.json`.
5. **Méthode anti-hasard des fonds quantitatifs** : apprentissage sur les 60 % les plus anciens, validation sur les 20 %
   suivants, verdict sur les 20 % les plus récents jamais vus ; purge et embargo entre les périodes ; correction des
   tests multiples (Benjamini-Hochberg) ; trades sans chevauchement, frais inclus, exécution prudente ; robustesse
   (**≥ 92 % sur chaque paire** ayant au moins 5 trades jamais vus, majorité des actifs gagnants, chaque année jamais vue, ≥ 75 % sur chaque moitié de la période jamais vue) ; contrôle : sur des prix au
   hasard le labo ne trouve rien (testé).
6. **Règle de sélection** : **≥ 92 % de réussite** (exigence du trader, objectif 99 à 100 %), ≥ 30 trades jamais vus **et** gain moyen positif après frais.
   Pas de prise de gain sous 0,5 ATR (sur des prix au hasard elle « réussit » déjà ~87 %).

## Données (gratuites, sans inscription)

- **Forex et métaux : Dukascopy** (banque suisse), 3 ans de bougies 1 h pour la recherche ; en direct, les heures
  publiées par Dukascopy complétées par le **temps réel TrueFX** (10 grandes paires, ou plus avec un compte TrueFX gratuit
  dans les secrets `TRUEFX_USER` / `TRUEFX_PASS`). Yahoo n'est plus utilisé pour le forex : un test sur 2 ans a montré
  qu'il produisait de faux setups (`validation.json`).
- **Crypto : Coinbase** (3 ans en 1 h, recherche et direct), taux de financement Kraken Futures (secours Hyperliquid),
  carnets d'ordres Coinbase / Bitstamp en direct.
- S&P 500 et VIX pour le régime de marché.

## Ce que tu reçois sur Telegram

- 🟢🔴 **Signal 1 h** (contrôle toutes les 5 min) : entrée, stop, prise de gain et heure de fermeture **en prix et en %**,
  la stratégie et son taux prouvé. Il part si le **vote de l'armée** (≥ 3 spécialistes d'accord, vote lui-même testé
  ≥ 92 %) ou un **setup élite** (≥ 92 % sur ≥ 100 trades jamais vus) — **aucun signal sous 92 %**, et seulement sur une paire où la stratégie a prouvé ses 92 % (au moins 5 trades jamais vus) — exigence du trader est actif, sans majorité contraire, un seul trade
  à la fois par paire.
- ✅🛑 **Résultat** de chaque signal et bilan réel ; une stratégie qui tombe sous 70 % en réel (après 20 trades) est retirée.
- Rien d'autre : pas de briefing ni de résumé (variable `INFOS` = `1` pour remettre le conseil de 8 h / 14 h / 20 h et la veille de l'Éclaireur).
- Tant qu'aucune stratégie ne passe, l'armée reste silencieuse. C'est voulu.

## Fonctionnement (GitHub Actions)

- **Labo - recherche de stratégies** (`mine.yml`) : forex 1 h et crypto 1 h en parallèle, chaque recherche relance la
  suivante sans arrêt (variable `CONTINU` = `stop` pour arrêter), prix Dukascopy gardés en cache ; publie `strategies.json`.
- **Éclaireur** (`scout.yml`) : veille quotidienne. **Vérification courtier** (`validate.yml`) : rejoue la meilleure
  stratégie sur Dukascopy et Yahoo. **Tests** (`tests.yml`).
- **Armée en direct** : tourne depuis le dépôt `crypto-signals-timesfm` (workflow *Armée*), qui a le jeton Telegram, et
  recharge chaque heure les stratégies publiées ici.

Réglages (variables GitHub, facultatives) : `MIN_WIN` (0.92), `CONSENSUS` (3), `BEAM` (60), `TIMEFRAMES` (1H).

## ⚠️ Honnêteté

Un taux de réussite de 80–95 % s'obtient avec un objectif proche et un stop plus éloigné : chaque gain est petit, une
perte en coûte plusieurs. C'est pourquoi le labo exige aussi un **gain total positif**. Les résultats passés, même
validés hors échantillon, ne garantissent rien. Signaux uniquement, aucun ordre, pas un conseil financier.
