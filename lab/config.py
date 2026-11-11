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
    "cr": envl("CRYPTO", "BTC,ETH,SOL,XRP,BNB,DOGE,ADA,AVAX,LINK,LTC,DOT,TRX,BCH,NEAR,XLM,HBAR,ATOM,UNI,AAVE,SUI,"
                         "FIL,ETC,ICP,SHIB,ARB,OP,APT,INJ,GRT,CRV,LDO,SAND,MANA,ALGO,XTZ,FET,IMX,STX,AXS,"
                         "COMP,SNX,YFI,1INCH,BAT,ZEC,DASH,QNT,CHZ,ENS,APE,BLUR,LRC,SUSHI,ZRX,ANKR,STORJ,MINA,FLOW,"
                         "KAVA,ROSE"),   # 60 cryptos (demande du trader), Coinbase + order flow Binance
    "stocks": envl("STOCKS", ""),                # actions retirees a la demande du trader (forex et crypto seulement)
}

MODE = os.environ.get("LAB_MODE", "tendance90").strip().lower()
# exigence du trader (3 nov. 10 h 51) : « tendance90 » = >= 90 % de trades gagnants ET gros gains : sortie en deux
# temps (moitie encaissee au 1er objectif, stop au prix d'entree, l'autre moitie court la tendance au stop suiveur)
# choix du trader (3 nov. 10 h) : « tendance » = vraie tendance (depart detecte, stop suiveur, on laisse courir) ;
# « gain » = objectifs fixes >= stop ; « win » = ancien mode 92 % (petits objectifs)
CFG = {
    "MODE": MODE,                            # gain : objectif >= stop, classement par gain total ; win : ancien mode 92 %
    "MIN_PF": envf("MIN_PF", 1.3),           # mode gain : facteur de profit minimal (gains / pertes) sur les trades jamais vus
    "PF_FLOOR_1": envf("PF_FLOOR_1", 1.05),  # mode gain : tri de depart de la recherche (une condition)
    "PF_FLOOR_2": envf("PF_FLOOR_2", 1.15),  # mode gain : tri des combinaisons et de l'apprentissage
    "ELITE_PF": envf("ELITE_PF", 1.5),       # mode gain : setup elite (peut signaler seul, >= 100 trades jamais vus)
    "HEURES_SIGNAUX": [envf("HEURE_DEBUT", 9.0), envf("HEURE_FIN", 21.0)],   # ouverture du matin -> fin du volume le soir (Paris), demande du 6 nov.
    "VOLUME_MIN_CRYPTO": envf("VOLUME_MIN_CRYPTO", 10e6),  # 6 nov. : cryptos avec un vrai volume (>= 10 M$ / 24 h)
    "KRONOS_ACCORD": int(envf("KRONOS_ACCORD", 0)),   # 11 nov. : Kronos ne bloque plus (bon sens 50 % sur 60 000
                                                       # previsions) ; son avis reste affiche dans le message
    "LIQUIDITE_SEULE": int(envf("LIQUIDITE_SEULE", 1)),   # demande du 5 nov. : liquidite + prix avance uniquement
    "TIMEFRAMES": envl("TIMEFRAMES", "1H,30M,15M,4H,1D"),    # demande du trader (5 nov.) : 1 h + 4 h + 15 min
    "MIN_WIN": envf("MIN_WIN", 0.90 if MODE == "tendance90" else 0.92),        # exigence du trader : le labo ne garde que les strategies >= 92 % (jamais vues)
    # tri de depart de la recherche (pieces a assembler) : volontairement plus large que la barre finale, car une
    # combinaison a 92 % est faite d'indicateurs qui, seuls, font moins. La barre finale reste MIN_WIN.
    "SEARCH_FLOOR_1": envf("SEARCH_FLOOR_1", 0.75),
    "SEARCH_FLOOR_2": envf("SEARCH_FLOOR_2", 0.80),
    "TARGET_WIN": envf("TARGET_WIN", 0.99),  # objectif du trader : 99 a 100 % (la plus haute passe toujours devant)
    "MIN_TRADES_TEST": envf("MIN_TRADES_TEST", 30),
    "FDR": envf("FDR", 0.05),                # taux de fausses decouvertes toleré (Benjamini-Hochberg)
    "CONSENSUS": envf("CONSENSUS", 3),       # specialistes d'accord minimum pour un signal
    "SEULEMENT_ACTIVEES": int(envf("SEULEMENT_ACTIVEES", 1)),   # seules les pistes activees (liquidite + Kronos) donnent un signal
                                                          # activees (activation.json) et leurs resultats
    "INFOS": envf("INFOS", 0),               # briefings / veille sur Telegram : coupes par defaut (signaux + resultats seulement)
    "SIGNAL_MIN_WIN": envf("SIGNAL_MIN_WIN", 0.90 if MODE == "tendance90" else 0.92),   # exigence du trader : aucun signal sous 92 % de reussite prouvee
    "ELITE_WIN": envf("ELITE_WIN", 0.90 if MODE == "tendance90" else 0.92),    # setup elite : peut signaler seul (>= 92 % ...
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
EXITS_WIN = [
    {"id": "e1", "tp": 0.5, "sl": 1.5, "h": 16},
    {"id": "e2", "tp": 0.5, "sl": 2.0, "h": 24},
    {"id": "e3", "tp": 0.75, "sl": 2.0, "h": 24},
    {"id": "e4", "tp": 1.0, "sl": 1.0, "h": 12},
    {"id": "e5", "tp": 1.0, "sl": 2.5, "h": 32},
    {"id": "e6", "tp": 1.5, "sl": 4.0, "h": 64},
    {"id": "e7", "tp": 2.0, "sl": 4.0, "h": 96},
    # pas de sortie a gain < 0,5 ATR : sur des prix au hasard elle « reussit » deja ~87 % (faux taux, teste)
]
# mode gain (choix du trader) : l'objectif est TOUJOURS au moins egal au stop (pas de petits gains)
EXITS_GAIN = [
    {"id": "e1", "tp": 1.0, "sl": 1.0, "h": 12},
    {"id": "e2", "tp": 1.5, "sl": 1.0, "h": 24},
    {"id": "e3", "tp": 2.0, "sl": 1.0, "h": 32},
    {"id": "e4", "tp": 1.5, "sl": 1.5, "h": 32},
    {"id": "e5", "tp": 3.0, "sl": 1.5, "h": 64},
    {"id": "e6", "tp": 2.0, "sl": 2.0, "h": 48},
    {"id": "e7", "tp": 4.0, "sl": 2.0, "h": 96},
]
# vraie tendance : aucun objectif fixe (tp tres loin, jamais vise), stop de depart puis stop suiveur en ATR
EXITS_TREND = [
    {"id": "e1", "tp": 99, "sl": 1.0, "trail": 1.0, "h": 48},
    {"id": "e2", "tp": 99, "sl": 1.0, "trail": 1.5, "h": 72},
    {"id": "e3", "tp": 99, "sl": 1.5, "trail": 1.5, "h": 96},
    {"id": "e4", "tp": 99, "sl": 1.5, "trail": 2.0, "h": 120},
    {"id": "e5", "tp": 99, "sl": 2.0, "trail": 2.0, "h": 168},
    {"id": "e6", "tp": 99, "sl": 2.0, "trail": 3.0, "h": 240},
    {"id": "e7", "tp": 99, "sl": 1.0, "trail": 2.0, "h": 120},
]
# >= 90 % et gros gains : sortie en deux temps (tp1 = 1er objectif pour la moitie, puis stop suiveur sur le reste)
EXITS_SCALE = [
    {"id": "e1", "tp": 0.5, "tp1": 0.5, "frac": 0.5, "sl": 1.0, "trail": 1.0, "h": 48},
    {"id": "e2", "tp": 0.5, "tp1": 0.5, "frac": 0.5, "sl": 1.5, "trail": 1.5, "h": 72},
    {"id": "e3", "tp": 0.75, "tp1": 0.75, "frac": 0.5, "sl": 1.5, "trail": 1.5, "h": 96},
    {"id": "e4", "tp": 1.0, "tp1": 1.0, "frac": 0.5, "sl": 1.5, "trail": 2.0, "h": 120},
    {"id": "e5", "tp": 1.0, "tp1": 1.0, "frac": 0.5, "sl": 2.0, "trail": 2.0, "h": 168},
    {"id": "e6", "tp": 0.75, "tp1": 0.75, "frac": 0.5, "sl": 2.0, "trail": 3.0, "h": 240},
    {"id": "e7", "tp": 1.0, "tp1": 1.0, "frac": 0.5, "sl": 1.0, "trail": 1.5, "h": 96},
]
EXITS = {"tendance": EXITS_TREND, "gain": EXITS_GAIN, "tendance90": EXITS_SCALE}.get(MODE, EXITS_WIN)
_ids = [x.strip().lower() for x in os.environ.get("EXIT_IDS", "").split(",") if x.strip()]
if _ids:                                    # recherche parallele : une famille de sorties par job
    EXITS = [e for e in EXITS if e["id"] in _ids]
