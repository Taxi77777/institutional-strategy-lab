"""Reglages du laboratoire (tous modifiables par variables d'environnement / variables GitHub)."""
import logging
import math
import os


def envf(name, default):
    v = os.environ.get(name, "").strip()
    if v == "":
        return default
    if isinstance(default, str):
        return v
    try:
        x = float(v.replace(",", "."))
        if not math.isfinite(x):
            raise ValueError
        return type(default)(x)
    except (ValueError, OverflowError):
        logging.warning("variable %s=%r invalide : %s par defaut", name, v, default)
        return default


def envl(name, default):
    return [s.strip().upper() for s in envf(name, default).split(",") if s.strip()]


UNIVERSE = {
    # actifs « universels » : cotes chez tous les courtiers (forex, metaux, grandes cryptos, indices / actions US)
    "fx": envl("FX", "EURUSD,GBPUSD,USDJPY,USDCHF,USDCAD,AUDUSD,NZDUSD,EURJPY,GBPJPY,EURGBP,AUDJPY,EURCHF,"
                     "GBPCHF,CADJPY,CHFJPY,EURAUD,EURCAD,GBPAUD,AUDNZD,XAUUSD,XAGUSD"),
    "cr": envl("CRYPTO", "BTC,ETH,SOL,XRP,BNB,DOGE,ADA,AVAX,LINK,LTC,DOT,TRX,BCH,NEAR,XLM,HBAR,ATOM,UNI,AAVE,SUI"),
    "stocks": envl("STOCKS", ""),                # actions retirees a la demande du trader (forex et crypto seulement)
}

CFG = {
    "TIMEFRAMES": envl("TIMEFRAMES", "15M"),          # demande du trader : 15 minutes uniquement
    "MIN_WIN": envf("MIN_WIN", 0.80),        # taux de reussite minimum (donnees jamais vues)
    "TARGET_WIN": envf("TARGET_WIN", 0.95),  # objectif affiche
    "MIN_TRADES_TEST": envf("MIN_TRADES_TEST", 30),
    "MIN_TRADES_TEST_15M": envf("MIN_TRADES_TEST_15M", 200),   # 5 / 15 / 30 min : au moins 200 trades jamais vus
    "FDR": envf("FDR", 0.05),                # taux de fausses decouvertes toleré (Benjamini-Hochberg)
    "CONSENSUS": envf("CONSENSUS", 3),       # specialistes d'accord minimum pour un signal
    "ELITE_WIN": envf("ELITE_WIN", 0.90),    # setup elite : peut signaler seul (>= 90 % ...
    "ELITE_WILSON": envf("ELITE_WILSON", 0.85),  # ... borne basse prudente >= 85 % ...
    "ELITE_N": envf("ELITE_N", 100),         # ... sur au moins 100 trades jamais vus)
    "BEAM": envf("BEAM", 40),                # largeur de la recherche (conditions gardees a chaque etage)
    "MAX_DEPTH": envf("MAX_DEPTH", 3),       # conditions combinees au maximum
    "SIMS": envf("SIMS", 10000),             # simulations Monte Carlo par strategie retenue
    "COST_BPS": {"fx": envf("COST_BPS_FX", 1.5), "cr": envf("COST_BPS_CR", 10.0),
                 "stocks": envf("COST_BPS_STOCKS", 3.0)},
    "STATE_DIR": envf("STATE_DIR", "."),
    "LOOP_MINUTES": min(envf("LOOP_MINUTES", 320), 320),
    "PAUSE": envf("PAUSE", 0.15),
}

# sorties testees (en multiples de l'ATR) : objectif de gain, stop, duree maxi en bougies
EXITS = [
    {"id": "e1", "tp": 0.5, "sl": 1.5, "h": 16},
    {"id": "e2", "tp": 0.5, "sl": 2.0, "h": 24},
    {"id": "e3", "tp": 0.75, "sl": 2.0, "h": 24},
    {"id": "e4", "tp": 1.0, "sl": 1.0, "h": 12},
    {"id": "e5", "tp": 1.0, "sl": 2.5, "h": 32},
    {"id": "e6", "tp": 1.5, "sl": 4.0, "h": 64},   # grands mouvements : les frais pesent moins en 5 / 15 / 30 min
    {"id": "e7", "tp": 2.0, "sl": 4.0, "h": 96},
    # pas de sortie a gain < 0,5 ATR : sur des prix au hasard elle « reussit » deja ~87 % (faux taux, teste)
]
