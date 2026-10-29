# 🪖 Institutional Strategy Lab — l'armée des 20 spécialistes

**Recherche automatique de stratégies intraday à 80–95 % de réussite, validées sur des données jamais vues.
Forex, or, crypto et actions US · compatible avec tous les courtiers · signaux Telegram · 100 % gratuit.**

## L'idée

Les institutions ont leurs algorithmes ; voici le tien, construit avec leur méthode :

1. **20 spécialistes**, chacun avec sa famille d'indicateurs : Tendance · Ichimoku · Momentum · Oscillateurs ·
   Volatilité · Bandes · Volume · VWAP · Volume profile · Fibonacci · AMD et sessions · Structure de marché ·
   Chandeliers · Statisticien · Mathématicien · Retour à la moyenne · Horaires · Inter-marchés · Macro · Carnet d'ordres.
2. **Près de 600 variantes d'indicateurs** (divergences RSI / MACD / OBV, Supertrend, Hull, régressions, Hurst,
   entropie, sweeps de liquidité…), découpées en **près de 3 000 conditions**, combinées par 2 et par 3 :
   **des millions de combinaisons testées** à chaque recherche.
3. **Méthode anti-hasard des fonds quantitatifs** :
   - apprentissage sur les 60 % les plus anciens, validation sur les 20 % suivants, **verdict sur les 20 % les plus récents, jamais vus** ;
   - correction des tests multiples (Benjamini-Hochberg) : sur des millions d'essais, le hasard « trouve » toujours quelque chose ;
   - trades sans chevauchement, frais inclus, exécution prudente ;
   - test de contrôle : sur des prix au hasard, le labo ne doit **rien** trouver (vérifié par les tests automatiques).
4. **Règle de sélection** : taux de réussite **≥ 80 %** (objectif 95 %) **et** gain moyen positif après frais.
   Un taux élevé avec un total perdant ne passe pas.
5. **L'armée vote** : un signal part seulement si au moins **3 spécialistes** ont une stratégie validée active dans le
   même sens, sans majorité contraire. Le vote lui-même est testé sur les données jamais vues.
6. **Tant qu'aucune stratégie ne passe, l'armée reste silencieuse.** C'est voulu.

## Ce que tu reçois sur Telegram

- 🪖 **Conseil des 20** à 8 h, 14 h et 20 h (Paris) : pour BTC, ETH, SOL, EURUSD, GBPUSD, USDJPY, or, SPY et QQQ,
  combien de spécialistes sont haussiers ou baissiers, et qui n'est pas d'accord.
- 🟢🔴 **Signal** (contrôle toutes les 15 minutes) : catégorie, ACHETER / VENDRE, entrée au prix du marché, sortie de sécurité,
  prise de gain et heure de fermeture **en prix et en %** (valable chez n'importe quel courtier), spécialistes d'accord et
  pas d'accord, taux de réussite prouvé, 10 000 simulations Monte Carlo.
- ✅🛑 **Résultat** de chaque signal et bilan réel de l'armée. Une stratégie qui tombe sous 70 % en réel (après 20 trades)
  est retirée automatiquement.

## Données (gratuites, sans inscription)

Yahoo Finance (bougies 15 min sur 60 jours, 1 h sur 2 ans), carnets d'ordres publics Coinbase et Bitstamp (crypto, en
direct uniquement : aucun historique gratuit n'existe), S&P 500 et VIX pour le régime de marché.

## Installation

1. Ajoute le secret **`TELEGRAM_BOT_TOKEN`** (le même que Bot Pro) dans *Settings → Secrets and variables → Actions*.
   La conversation est déjà appairée (destination chiffrée avec le jeton).
2. *Actions* → **Labo - recherche de strategies** → *Run workflow* (puis automatique chaque dimanche soir).
3. *Actions* → **Armee - signaux en direct** → *Run workflow* (tourne ensuite 24 h/24).

Réglages (variables GitHub, facultatives) : `MIN_WIN` (0.80), `CONSENSUS` (3), `BEAM` (40), `TIMEFRAMES` (15M,1H).

## ⚠️ Honnêteté

Un taux de réussite de 80–95 % s'obtient avec un objectif proche et un stop plus éloigné : chaque gain est petit, une
perte en coûte plusieurs. C'est pourquoi le labo exige aussi un **gain total positif**. Les résultats passés, même
validés hors échantillon, ne garantissent rien. Signaux uniquement, aucun ordre, pas un conseil financier.
