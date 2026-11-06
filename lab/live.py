"""L'armee en direct : controle toutes les 5 minutes des bougies 1 heure (forex : Dukascopy + temps reel TrueFX ;
crypto : Coinbase). Un signal part si le vote de l'armee (>= CONSENSUS specialistes d'accord, vote teste >= 80 %)
ou un setup elite (>= 90 % sur >= 100 trades jamais vus) est actif, sans majorite contraire, un seul trade par paire.
Briefing commun (« conseil des specialistes ») a 8 h, 14 h et 20 h (Paris), veille de l'Eclaireur chaque matin."""
import datetime
import hashlib
import json
import logging
import os
import signal
import time
from zoneinfo import ZoneInfo

import numpy as np

from . import data, ideas as IDEAS, indicators as IND, positionnement as POS, telegram as TG
from .config import CFG, EXITS, UNIVERSE
from .miner import BENCH, align, family, is_pa, kronos_feats, macro_series, funding_refs, orderflow_refs

FUND = {}

log = logging.getLogger("lab")
PARIS = ZoneInfo("Europe/Paris")
STATE_FILE = "lab_state.json"
STEP = {"1H": 3600, "4H": 14400, "15M": 900, "30M": 1800, "1D": 86400}
KEEP = {"1H": 2500, "4H": 900, "15M": 1500, "30M": 1500, "1D": 500}      # 4 h : ~7 mois ; 15 min : ~3 semaines (fenetres max ~300 bougies)
SESSION_NOM = {"asie": "🌏 Session Asie", "londres": "🇬🇧 Session Londres", "new_york": "🇺🇸 Session New York",
               "soir": "🌙 Session du soir"}
CACHE = {}
FAIL = {}
STOP = [False]
SEP = "━━━━━━━━━━━━━━━"
MAJORS = [("cr", "BTC"), ("cr", "ETH"), ("cr", "SOL"), ("fx", "EURUSD"), ("fx", "GBPUSD"), ("fx", "USDJPY"),
          ("fx", "XAUUSD"), ("fx", "XAGUSD")]
BANNER = {"cr": "🪙 <b>CRYPTO</b>", "fx": "💱 <b>FOREX</b>", "stocks": "📈 <b>ACTIONS / INDICES US</b>"}


def banner(kind, sym):
    return "🥇 <b>OR / ARGENT</b>" if sym in ("XAUUSD", "XAGUSD") else BANNER[kind]


def fmt(x):
    if x >= 1000:
        return "%.2f" % x
    if x >= 20:
        return "%.3f" % x
    if x >= 1:
        return "%.5f" % x
    return "%.6g" % x


def paris(ts):
    return datetime.datetime.fromtimestamp(ts, PARIS)


# ------------------------------------------------------------------ etat
def load_state():
    st = {}
    for p in (os.path.join(CFG["STATE_DIR"], STATE_FILE), STATE_FILE):
        try:
            with open(p) as f:
                st = json.load(f)
            break
        except Exception:
            continue
    for k, t in (("alerts", dict), ("open", list), ("track", list), ("live", dict), ("retired", list), ("done", dict)):
        if not isinstance(st.get(k), t):
            st[k] = t()
    return st


def save_state(st):
    cut = time.time() - 4 * 86400
    st["alerts"] = {k: v for k, v in st["alerts"].items() if isinstance(v, (int, float)) and v > cut}
    st["track"] = st["track"][-2000:]
    st["telegram"] = dict(TG.STATS)
    os.makedirs(CFG["STATE_DIR"], exist_ok=True)
    p = os.path.join(CFG["STATE_DIR"], STATE_FILE)
    with open(p + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(p + ".tmp", p)


STRAT_URL = os.environ.get("LAB_STRATEGIES_URL",
                           "https://raw.githubusercontent.com/Taxi77777/institutional-strategy-lab/main/strategies.json")
LAB_CACHE = {"ts": 0, "v": None}
NUM_URL = STRAT_URL.replace("strategies.json", "strategies_numerologie.json")
ACT_URL = STRAT_URL.replace("strategies.json", "activation.json")


def load_strategies():
    """Strategies du labo : version publiee sur GitHub (rechargee chaque heure), sinon le fichier local."""
    if LAB_CACHE["v"] is not None and time.time() - LAB_CACHE["ts"] < 3600:
        return LAB_CACHE["v"]
    v = None
    try:
        r = data.SESSION.get(STRAT_URL, params={"t": int(time.time() // 3600)}, timeout=20)
        if r.ok:
            v = r.json()
    except Exception:
        v = None
    if not isinstance(v, dict):
        try:
            with open("strategies.json") as f:
                v = json.load(f)
        except Exception:
            v = {"classes": []}
    v["activees"] = activated(v)
    LAB_CACHE.update(ts=time.time(), v=v)
    return v


def activated(v):
    """Strategies activees par le trader sous la barre des 92 % (activation.json) : prises dans le plafond publie,
       avec leurs zones exactes ; signal seulement sur les paires ou elles gagnent sur les trades jamais vus."""
    try:
        r = data.SESSION.get(ACT_URL, params={"t": int(time.time() // 3600)}, timeout=20)
        act = r.json() if r.ok else None
    except Exception:
        act = None
    if not isinstance(act, dict):
        try:
            with open("activation.json") as f:
                act = json.load(f)
        except Exception:
            act = {}
    out, seen, evn, dirn = [], set(), {}, {}
    for rule in act.get("regles", []):
        if rule.get("auto"):                        # meilleures pistes du plafond, mises a jour a chaque recherche
            for cl in v.get("classes", []):
                if cl["kind"] != rule["kind"] or cl["tf"] != rule.get("tf", "1H"):
                    continue
                taken = 0
                # d'abord les meilleures pistes de chaque session (signaux a toutes les heures, demande du 5 nov.),
                # puis le reste par priorite du trader : liquidite
                per_s = rule.get("par_session", 0)
                sess = []
                for nm in rule.get("sessions", list(SESSION_NOM)):
                    sess += sorted([r for r in cl.get("sessions", []) if r.get("session") == nm],
                                   key=lambda r: (-is_liquidity(r), -r["test_win"]))[:per_s]
                cont = sorted(cl.get("continuation", []), key=lambda r: -r["test_win"])[:rule.get("par_continuation", 0)]
                pool = cont + sess + sorted(cl.get("activables", []) + cl.get("plafond", []) + cl.get("sessions", [])
                                            + cl.get("continuation", []) + cl.get("ventes", []) + cl.get("achats", []),
                                            key=lambda r: (-is_liquidity(r), -r["test_win"]))
                for pass_, p in [(0, p) for p in pool] + [(1, p) for p in pool]:   # 2e passage : places libres
                    if taken >= rule.get("max", 3):
                        break
                    ps = (p.get("robustesse") or {}).get("par_session") or {}
                    if (p["agent"] in rule.get("exclure", []) or any(len(c) != 4 for c in p["conds"])
                            or (rule.get("heures_de_volume") and ps and ps.get("asie", {}).get("part", 0) > 0.5)
                            or (rule.get("liquidite_seule") and not is_pa(p))
                            or p["test_win"] < rule["taux_min"] or p["test_n"] < rule["trades_min"]
                            or p.get("test_avg", 0) <= 0
                            or p.get("validation_win", 1) < rule.get("validation_min", 0)):
                        continue
                    sig = cl["tf"] + family(p)                # une meme piste peut vivre en 1 h, 4 h et 15 min
                    same = ("stats", cl["kind"], cl["tf"], p["dir"], p["exit"], round(p["test_win"], 4), p["test_n"])
                    if sig in seen or same in seen:
                        continue                    # meme piste (memes trades) ecrite autrement : une seule fois
                    pa = (p.get("robustesse") or {}).get("par_actif", {})
                    ok = sorted(sym for sym, x in pa.items() if x.get("trades", 0) >= rule["paire_trades_min"]
                                and x.get("reussite", 0) >= rule["paire_taux_min"] and x.get("gain_moyen", 0) > 0)
                    ex = next((e for e in EXITS if e["id"] == p["exit"]), None)
                    if not ok or ex is None:
                        continue
                    ev = event_of(p)                        # pistes du meme evenement rare (fixing, NFP...) :
                    if pass_ == 0 and ev and evn.get((cl["kind"], cl["tf"], ev), 0) >= rule.get("max_meme_evenement", 99):
                        continue                            # elles se declenchent a la meme heure, place aux autres
                    if pass_ == 0 and dirn.get((cl["kind"], cl["tf"], p["dir"]), 0) >= rule.get("max", 3) // 2:
                        continue                            # achat ET vente : moitie des places pour chaque sens d'abord
                    dirn[(cl["kind"], cl["tf"], p["dir"])] = dirn.get((cl["kind"], cl["tf"], p["dir"]), 0) + 1
                    if ev:
                        evn[(cl["kind"], cl["tf"], ev)] = evn.get((cl["kind"], cl["tf"], ev), 0) + 1
                    seen.add(sig)
                    seen.add(same)
                    taken += 1
                    nom = "%s%s%s · %.0f %% (%d trades jamais vus)" % (
                        "🚀 Continuation · " if p.get("agent") == "noyau_continuation" else
                        "💧 Liquidité · " if is_liquidity(p) else "", rule["nom"],
                        " · " + SESSION_NOM[p["session"]] if p.get("session") in SESSION_NOM else "",
                        100 * p["test_win"], p["test_n"])
                    out.append({"id": "activee-%s-%s%s" % (cl["kind"], "" if cl["tf"] == "1H" else cl["tf"].lower() + "-",
                                                           hashlib.md5(sig[len(cl["tf"]):].encode()).hexdigest()[:8]),
                                "kind": cl["kind"], "tf": cl["tf"], "agent": p["agent"], "dir": p["dir"], "exit": ex,
                                "conds": [list(c) for c in p["conds"]],
                                "stats": {"test": {"win": p["test_win"], "n": p["test_n"], "avg": p["test_avg"]}},
                                "robustesse": p.get("robustesse", {}), "activee": nom, "paires_ok": ok})
            continue
        for cl in v.get("classes", []):
            if cl["kind"] != rule["kind"] or cl["tf"] != rule.get("tf", "1H"):
                continue
            for p in cl.get("plafond", []):
                names = [tuple(c[:2]) for c in p["conds"]]
                if (p["agent"] != rule["agent"] or p["dir"] != rule["dir"] or p["exit"] != rule["exit"]
                        or any(len(c) != 4 for c in p["conds"])
                        or not all(tuple(x) in names for x in rule.get("doit_contenir", []))
                        or p["test_win"] < rule["taux_min"] or p["test_n"] < rule["trades_min"]):
                    continue
                pa = (p.get("robustesse") or {}).get("par_actif", {})
                ok = sorted(sym for sym, x in pa.items() if x.get("trades", 0) >= rule["paire_trades_min"]
                            and x.get("reussite", 0) >= rule["paire_taux_min"] and x.get("gain_moyen", 0) > 0)
                ex = next((e for e in EXITS if e["id"] == p["exit"]), None)
                if not ok or ex is None:
                    continue
                slug = "".join(ch if ch.isalnum() else "-" for ch in rule["nom"].lower())
                out.append({"id": "activee-%s" % slug, "kind": cl["kind"], "tf": cl["tf"], "agent": p["agent"],
                            "dir": p["dir"], "exit": ex, "conds": [list(c) for c in p["conds"]],
                            "stats": {"test": {"win": p["test_win"], "n": p["test_n"], "avg": p["test_avg"]}},
                            "robustesse": p.get("robustesse", {}), "activee": rule["nom"], "paires_ok": ok})
                break
            if out and out[-1]["activee"] == rule["nom"]:
                break
    return out


def heat_check(sym, kind, dr, fx):
    """Heatmap du carnet (crypto, en direct) : un achat est bloque si un gros mur vendeur est a moins d'un demi-ATR
       au-dessus (une vente : mur acheteur en dessous). Chaque releve est aussi enregistre (historique a tester)."""
    from lab import kraken_fx as KF
    if kind != "cr" and sym not in KF.PAIRS:
        return True, None                           # forex hors Kraken : pas de carnet public
    hm = HEAT.get(sym)
    if not hm or time.time() - hm[0] > 240:
        hm = (time.time(), data.heatmap(sym, kind=kind))
        HEAT[sym] = hm
        record_heat(sym, hm[1])
    h = hm[1]
    if not h:
        return True, None
    half = 0.5 * fx["atr"] / fx["close"]
    block = h.get("mur_vente_dist" if dr > 0 else "mur_achat_dist")
    zone = "±1 %%" if kind == "cr" else "±0,25 %%"
    txt = ("🧱 Heatmap%s : mur acheteur %s · mur vendeur %s · déséquilibre " + zone + " : %+.0f %%") % (
        "" if kind == "cr" else " (carnet Kraken)",
        "à -%.2f %%" % (100 * h["mur_achat_dist"]) if "mur_achat_dist" in h else "aucun",
        "à +%.2f %%" % (100 * h["mur_vente_dist"]) if "mur_vente_dist" in h else "aucun",
        100 * h.get("desequilibre_1" if kind == "cr" else "desequilibre_0.25", 0))
    return not (block is not None and block < half), txt


def heat_record_all():
    """Releve horaire de la heatmap des cryptos suivies (meme sans signal) : constitue l'historique."""
    from lab import kraken_fx as KF
    hour = int(time.time() // 3600)
    fxs = [s_ for s_ in UNIVERSE.get("fx", []) if s_ in KF.PAIRS]
    for kind_, sym in [("cr", s_) for s_ in UNIVERSE.get("cr", [])] + [("fx", s_) for s_ in fxs]:
        if STOP[0]:
            return
        if HEAT.get(("rec", sym)) == hour:
            continue
        try:
            h = data.heatmap(sym, kind=kind_)
        except Exception:
            h = None
        HEAT[sym] = (time.time(), h)
        record_heat(sym, h)


def record_heat(sym, h, path=None):
    """Historique de la heatmap (une ligne par releve) : dans quelques mois le labo pourra la tester pour de vrai."""
    if not h:
        return
    path = path or os.path.join(CFG["STATE_DIR"], "heatmap.jsonl")
    hour = int(time.time() // 3600)
    if HEAT.get(("rec", sym)) == hour:
        return                                      # un releve par heure et par crypto
    HEAT[("rec", sym)] = hour
    try:
        with open(path, "a") as f:
            f.write(json.dumps(dict(h, sym=sym, ts=int(time.time()))) + "\n")
    except Exception as ex:
        log.warning("heatmap %s : %s", sym, ex)


LIQ_AGENTS = {"liquidite", "smc", "orderflow", "noyau_liquidite"}


EVENTS = ("fixing", "nfp", "cme", "fomc", "gotobi", "fin_de_mois")


def event_of(r):
    """Evenement rare dont depend la piste (toutes ses entrees tombent a ce moment-la), sinon None."""
    txt = json.dumps(r.get("conds", []))
    return next((e for e in EVENTS if e in txt), None)


def is_liquidity(r):
    """Piste de liquidite (priorite du trader) : noyau liquidite, ou une condition liquidite / SMC / order flow."""
    return int(r.get("agent") in LIQ_AGENTS or any(c[0] in LIQ_AGENTS for c in r.get("conds", [])))


def in_hours(t):
    """Demande du trader : signaux seulement quand il y a du volume, de 9 h a 18 h 30 (heure de Paris)."""
    p = paris(t)
    a, b = CFG["HEURES_SIGNAUX"]
    return a * 60 <= p.hour * 60 + p.minute <= b * 60


KAVIS = {"t": 0, "v": {}}
KAVIS_URL = STRAT_URL.replace("strategies.json", "donnees/kronos/avis.json")


def kronos_avis(sym):
    """Sens prevu par Kronos (5 prochaines bougies jour, en ATR) ; rafraichi toutes les 30 min, None si inconnu."""
    if time.time() - KAVIS["t"] > 1800:
        try:
            r = data.SESSION.get(KAVIS_URL, params={"t": int(time.time() // 1800)}, timeout=20)
            if r.ok:
                j = r.json()
                KAVIS["v"] = j.get("actifs", {}) if time.time() - j.get("genere", 0) < 36 * 3600 else {}  # perime : ignore
        except Exception as ex:
            log.warning("avis Kronos : %s", ex)
        KAVIS["t"] = time.time()
    return KAVIS["v"].get(sym)


def activated_signals(st, lab, kind, tf, sym, fx):
    """Signal d'une strategie activee par le trader (a part du vote de l'armee)."""
    if kind == "fx" and (not in_hours(time.time()) or paris(time.time()).weekday() >= 5):
        return 0                                    # forex hors des heures de volume : aucun signal (crypto : 24 h/24)
    for s in sorted(lab.get("activees", []), key=lambda a: -is_liquidity(a)):
        if (s["kind"] != kind or s["tf"] != tf or sym not in s["paires_ok"] or s["id"] in set(st["retired"])
                or not active(s, fx["last"])):
            continue
        if any(o["sym"] == sym for o in st["open"]) or any(r["args"][3] == sym for r in PENDING):
            return 0
        if kind == "cr":                            # demande du trader : seulement des cryptos avec un vrai volume
            a_ = fx.get("a")
            n_ = max(1, int(86400 // STEP[tf]))
            v24 = float((a_[-n_:, 5] * a_[-n_:, 4]).sum()) if a_ is not None and len(a_) else 0.0
            if v24 < CFG["VOLUME_MIN_CRYPTO"]:
                st.setdefault("diag", {}).setdefault("volume_trop_faible", {})[sym] = "%.1f M$ sur 24 h" % (v24 / 1e6)
                return 0
        kv = kronos_avis(sym)
        if CFG["KRONOS_ACCORD"] and (kv is None or kv["atr"] * s["dir"] <= 0):
            dz = st.setdefault("diag", {}).setdefault("kronos_pas_d_accord", {})
            dz["%s %s" % (sym, tf)] = "%s %s : piste %s, Kronos %s" % (
                paris(time.time()).strftime("%d/%m %H:%M"), sym, "achat" if s["dir"] > 0 else "vente",
                "sans avis" if kv is None else "%+.2f ATR" % kv["atr"])
            continue                                # demande du trader : tout le monde dans le meme sens que Kronos
        key = "act|%s|%s|%d|%d" % (sym, tf, s["dir"], int(fx["ts"]))
        if key in st["alerts"]:
            continue
        ok, heat = heat_check(sym, kind, s["dir"], fx)
        if not ok:                                  # la heatmap n'a pas encore d'historique teste : elle previent, elle
            heat = "⚠️ MUR PROCHE DANS LE SENS DU TRADE (prudence) · " + (heat or "")      # ne bloque plus une piste prouvee
            st.setdefault("diag", {}).setdefault("murs_signales", []).append(
                "%s %s %s" % (paris(time.time()).strftime("%d/%m %H:%M"), sym, tf))
            st["diag"]["murs_signales"] = st["diag"]["murs_signales"][-20:]
        st["alerts"][key] = time.time()
        kt = ("🤖 Kronos (IA) d'accord : %s %+.1f ATR (%+.1f %%) sur 5 jours" % (
            "hausse" if kv["atr"] > 0 else "baisse", kv["atr"], kv["pct"]) if kv else None)
        heat = " · ".join(x for x in (kt, heat) if x) or None
        a_ = fx.get("a")
        n_ = max(1, int(86400 // STEP[tf]))
        vol = float((a_[-n_:, 5] * a_[-n_:, 4]).sum()) if a_ is not None and len(a_) else 0.0   # volume 24 h en $
        PENDING.append({"win": s.get("test_win", 0), "vol": vol, "args": (st, kind, tf, sym, s["dir"], fx, dict(s, heat=heat),
                                                                   {s["agent"]: s}, {}, lab)})
        return 1
    return 0


PENDING = []
EVAL = {}


def flush_pending():
    """Demande du trader : les signaux d'un meme controle sont envoyes classes, le meilleur pourcentage d'abord,
       puis le plus gros volume ; chaque message porte son rang (n° 1 = le plus solide)."""
    rows = sorted(PENDING, key=lambda r: (-round(r["win"], 3), -r["vol"]))
    PENDING.clear()
    for i, r in enumerate(rows):
        st, kind, tf, sym, d_, fx, s, ag, ex, lab = r["args"]
        rang = "🏆 Rang %d/%d du controle (%.0f %% de reussite, volume %.1f M$ sur 24 h)" % (
            i + 1, len(rows), 100 * r["win"], r["vol"] / 1e6)
        s = dict(s, heat=" · ".join(x for x in (rang, s.get("heat")) if x))
        if any(o["sym"] == sym for o in st["open"]):
            continue
        send_signal(st, kind, tf, sym, d_, fx, s, ag, ex, lab, activee=True)


# ------------------------------------------------------------------ donnees en direct
def hist(sym, kind, tf):
    key = (sym, kind, tf)
    c = CACHE.get(key)
    now = time.time()
    if not c or now - c["full"] > 6 * 3600:
        if now - FAIL.get(key, 0) < 1800:           # echec recent : on n'arrete pas toute l'armee 10 minutes dessus
            raise RuntimeError("source indisponible, nouvel essai dans %d min" % ((1800 - now + FAIL[key]) // 60 + 1))
        try:
            a = fetch_live(sym, kind, tf, full=True)
        except Exception:
            FAIL[key] = now
            raise
        a = np.asarray(a, dtype=float).reshape(-1, 6)       # actif sans donnees (ex. retire de Coinbase) : tableau vide
        CACHE[key] = {"a": a[-KEEP[tf]:], "full": now, "vol": data.VOLUME.get(sym) if kind == "fx" else None}
    else:
        try:
            new = fetch_live(sym, kind, tf, full=False)
        except Exception as ex:                     # suite indisponible (week-end, reseau) : on garde l'existant
            log.warning("%s %s suite : %s", sym, tf, ex)
            new, data.VOLUME[sym] = np.zeros((0, 6)), None
        rows = {int(r[0]): r for r in c["a"]}
        rows.update({int(r[0]): r for r in new})
        c["a"] = np.array([rows[k] for k in sorted(rows)], dtype=float).reshape(-1, 6)[-KEEP[tf]:]
        if kind == "fx" and data.VOLUME.get(sym) is not None:      # volume des ticks : propre a chaque unite
            old_v = c.get("vol")
            vd = dict(zip(old_v[0].tolist(), old_v[1].tolist())) if old_v is not None else {}
            nv = data.VOLUME[sym]
            vd.update(zip(nv[0].tolist(), nv[1].tolist()))
            ks = sorted(vd)[-KEEP[tf] - 50:]
            c["vol"] = (np.array(ks, dtype=float), np.array([vd[k] for k in ks], dtype=float))
    return CACHE[key]["a"]


KMIN = {}


def kraken_minutes(sym):
    """Forex en temps reel sans inscription : minutes construites a partir des transactions publiques de Kraken
       (EURUSD, GBPUSD, AUDUSD, USDJPY, USDCAD, USDCHF). Dukascopy ne publie chaque heure qu'une fois terminee :
       sans ce relais, les bougies 15 / 30 min arrivaient trop tard et le signal etait ecarte."""
    from lab import kraken_fx as KF
    if sym not in KF.PAIRS:
        return []
    c = KMIN.get(sym)
    if c and time.time() - c[0] < 60:
        return c[1]
    since = str(int((time.time() - 3 * 3600) * 10 ** 9))
    trades = []
    for _ in range(10):
        try:
            data_, last = KF._trades(sym, since)
        except Exception:
            break
        trades += data_
        if len(data_) < 1000:
            break
        since = last
    mins = {}
    for t in trades:
        px, ts = float(t[0]), float(t[2])
        m = int(ts // 60 * 60)
        if m + 60 > time.time():
            continue                                # minute en cours : pas encore complete
        o = mins.get(m)
        mins[m] = [px, px, px, px] if o is None else [o[0], max(o[1], px), min(o[2], px), px]
    out = [(t, o, h, l, c_, 0.0) for t, (o, h, l, c_) in sorted(mins.items())]
    KMIN[sym] = (time.time(), out)
    return out


def fetch_live(sym, kind, tf, full):
    """Memes sources que la recherche : Coinbase pour la crypto, Yahoo pour le forex (volume neutralise la ou la
       recherche utilise Dukascopy)."""
    if kind == "cr":
        try:
            days = int(KEEP[tf] * STEP[tf] / 86400) + 2 if full else 2
            return data.coinbase(sym, STEP[tf], days)
        except Exception as ex:
            log.warning("Coinbase %s %s : %s (Yahoo en secours)", sym, tf, ex)
    if kind == "fx" and STEP[tf] > 3600:            # 4 h : fichiers mensuels 1 h (8 fichiers au lieu de 200 jours)
        step = STEP[tf]
        rows, vol = {}, {}
        if full:
            try:
                base = data.dukascopy_hours(sym, int(KEEP[tf] * step / 86400 * 7 / 5) + 10)
            except Exception as ex:                 # fichiers mensuels refuses : fichiers journaliers (plus long)
                log.warning("%s 4 h mensuel : %s (journalier)", sym, ex)
                return data.dukascopy_live(sym, step, int(KEEP[tf] * step / 86400 * 7 / 5) + 4,
                                           recent=data.truefx_minutes(sym, step) or kraken_minutes(sym))
            vb = data.VOLUME.get(sym)
            rows.update({int(r[0]): r for r in base})
            if vb is not None:
                vol.update(zip(vb[0].tolist(), vb[1].tolist()))
        rec = data.dukascopy_live(sym, 3600, 5, recent=data.truefx_minutes(sym, 3600) or kraken_minutes(sym))
        vr = data.VOLUME.get(sym)
        rows.update({int(r[0]): r for r in rec})
        if vr is not None:
            vol.update(zip(vr[0].tolist(), vr[1].tolist()))
        h = np.array([rows[k] for k in sorted(rows)], dtype=float)
        ks = sorted(vol)
        data.VOLUME[sym] = (np.array(ks, dtype=float), np.array([vol[k] for k in ks], dtype=float))   # 1 h, additionne
        a = data._resample(h, step)
        if step >= 86400:
            a = a[((a[:, 0] // 86400 + 3) % 7) != 6]    # 1 jour : sans la petite bougie du dimanche soir
        if len(a) and h[-1, 0] < a[-1, 0] + step - 3600 and time.time() < a[-1, 0] + step + 2 * 3600:
            a = a[:-1]                              # derniere bougie 4 h pas encore complete : jamais de signal dessus
        return a
    if kind == "fx":                                # forex : uniquement Dukascopy (Yahoo donnait de faux setups)
        days = int(KEEP[tf] * STEP[tf] / 86400 * 7 / 5) + 4 if full else 5      # suite : 5 derniers jours seulement
        return data.dukascopy_live(sym, STEP[tf], days, recent=data.truefx_minutes(sym, STEP[tf]) or kraken_minutes(sym))
    return data.bars(sym, kind, tf) if full else data.bars(sym, kind, tf, rng="1mo" if tf == "1H" else "5d")


MACRO = {"ts": 0, "v": {}}


def macro():
    if time.time() - MACRO["ts"] > 6 * 3600:
        MACRO["v"] = macro_series(data.daily)
        MACRO["ts"] = time.time()
    return MACRO["v"]


FEAT = {}


POSC = {}
OFC = {}
HEAT = {}


def features(sym, kind, tf, now, specs=None):
    a = hist(sym, kind, tf)
    a = a[a[:, 0] + STEP[tf] <= now]                # bougies cloturees uniquement
    if len(a) < 300:
        NOTE[(sym, kind, tf)] = "seulement %d bougies" % len(a)
        return None
    NOTE.pop((sym, kind, tf), None)
    key = (sym, kind, tf, len(specs or {}))
    if key in FEAT and FEAT[key]["ts"] == float(a[-1, 0]) and FEAT[key]["close"] == float(a[-1, 4]):
        return FEAT[key]                            # pas de nouvelle bougie : rien a recalculer
    refs = None
    if sym != BENCH[kind]:
        try:
            b = hist(BENCH[kind], kind, tf)
            refs = {"bench": align(b[:, 0], b[:, 4], a[:, 0])}
        except Exception:
            refs = None
    refs = dict(refs or {})
    refs["sym"] = sym                               # flux des operateurs : panier deja en cache (aucune requete de plus)
    pan = {}
    for (s2, k2, t2), cc in list(CACHE.items()):
        if k2 == kind and t2 == tf and len(cc["a"]):
            b2 = cc["a"][cc["a"][:, 0] + STEP[tf] <= now]
            if len(b2):
                pan[s2] = align(b2[:, 0], b2[:, 4], a[:, 0])
    pan[sym] = a[:, 4]
    refs["panier"] = pan
    ps = POSC.get(kind)
    if not ps or time.time() - ps[0] > 6 * 3600:      # COT / DVOL : rafraichis toutes les 6 h
        ps = (time.time(), POS.sources(kind, years=4))   # direct : 4 ans suffisent (percentile sur 3 ans), rapide
        POSC[kind] = ps
    refs.update(ps[1])
    if kind == "cr":                                # order flow crypto : Binance, rafraichi a chaque heure
        of = OFC.get((sym, tf))
        if not of or time.time() - of[0] > min(1800, STEP[tf]):
            try:
                of = (time.time(), data.binance_flow(sym, days=int(KEEP[tf] * STEP[tf] / 86400) + 2,
                                                     interval="15m" if STEP[tf] < 3600 else "1h"))
            except Exception:
                of = (time.time(), None)
            OFC[(sym, tf)] = of
        refs["orderflow"] = orderflow_refs(sym, kind, a[:, 0], flow=of[1]) if of[1] is not None else None
    else:
        v_ = CACHE.get((sym, kind, tf), {}).get("vol")
        if v_ is not None:
            data.VOLUME[sym] = v_                    # volume de CETTE unite de temps (1 h, 4 h ou 15 min)
        refs["orderflow"] = orderflow_refs(sym, kind, a[:, 0], live=True)
    mac = {k: align(v[0], v[1], a[:, 0]) for k, v in macro().items()}
    if kind == "cr":
        refs = dict(refs or {})
        fc = FUND.get(sym)
        if not fc or time.time() - fc[0] > 3600:
            try:
                fc = (time.time(), data.funding(sym, days=60))
            except Exception:
                fc = (time.time(), None)
            FUND[sym] = fc
        refs["funding"] = funding_refs(sym, a[:, 0], fetch=lambda s_: fc[1] if fc[1] is not None else (_ for _ in ()).throw(RuntimeError("indisponible")))
    F, atr = IND.compute(a, kind, refs, mac)
    F["kronos"] = kronos_feats(kind, tf, sym, a[:, 0])
    for nm, sp in (specs or {}).items():
        x = IDEAS.apply_spec(F, sp)
        if x is not None:
            F["inventeur"][nm] = x
    last = {(ag, nm): float(x[-1]) for ag, d in F.items() for nm, x in d.items()}
    FEAT[key] = {"a": a, "last": last, "atr": float(atr[-1]), "ts": float(a[-1, 0]), "close": float(a[-1, 4])}
    if KEEP_FULL[0]:                                # controle (sonde) : toutes les valeurs, pas seulement la derniere
        FEAT[key]["F"] = F
    return FEAT[key]


KEEP_FULL = [False]
NOTE = {}


PROCHES = {}


def near(st, lab, kind, tf, sym, fx):
    """Proximite du signal : pour chaque piste active sur cet actif, nombre de conditions deja reunies et condition
       manquante (les 15 plus proches gardees dans l'etat : on voit ou le prochain signal va tomber)."""
    px = PROCHES
    for s in lab.get("activees", []):
        if s["kind"] != kind or s["tf"] != tf or sym not in s["paires_ok"]:
            continue
        ok, miss = 0, []
        for ag, nm, lo, hi in s["conds"]:
            v = fx["last"].get((ag, nm))
            if v is not None and np.isfinite(v) and lo < v <= hi:
                ok += 1
            else:
                miss.append("%s=%s (zone %s a %s)" % (nm[:40], "?" if v is None or not np.isfinite(v) else "%.3g" % v,
                                                        "%.3g" % lo, "%.3g" % hi))
        px["%s %s %s" % (sym, tf, s["id"][-8:])] = {"reunies": "%d/%d" % (ok, len(s["conds"])), "manque": miss[:2],
                                                    "piste": s["activee"][:70], "sens": "achat" if s["dir"] > 0 else "vente",
                                                    "heure": paris(time.time()).strftime("%d/%m %H:%M"),
                                                    "score": ok / max(1, len(s["conds"])), "t": time.time()}
    if len(px) > 400:
        for k in sorted(px, key=lambda k: px[k]["t"])[:len(px) - 400]:
            px.pop(k)
    st.setdefault("diag", {})["plus_proches"] = sorted(({"actif": k, **v} for k, v in px.items() if v["score"] < 1),
                                                       key=lambda x: (x["score"], x["t"]), reverse=True)[:15]


def active(strat, last):
    for ag, nm, lo, hi in strat["conds"]:
        v = last.get((ag, nm))
        if v is None or not np.isfinite(v) or not (v > lo and v <= hi):
            return False
    return True


# ------------------------------------------------------------------ le vote de l'armee
def council_votes(lab, kind, tf, sym, fx, retired):
    votes = {1: {}, -1: {}}
    for cl in lab["classes"]:
        if cl["kind"] != kind or cl["tf"] != tf:
            continue
        for s in cl["strategies"]:
            if s["id"] in retired:
                continue
            if s.get("agent") == "numerologie":
                continue                            # strategie a part : ne vote jamais avec l'armee
            if active(s, fx["last"]):
                best = votes[s["dir"]].get(s["agent"])
                if not best or s["wilson_test"] > best["wilson_test"]:
                    votes[s["dir"]][s["agent"]] = s
    return votes


def gain_mode():
    return CFG.get("MODE", "gain") in ("gain", "tendance")


def is_strong(s):
    """Strategie assez forte pour mener un signal : mode gain -> gain moyen positif et facteur de profit >= MIN_PF ;
       mode win -> >= SIGNAL_MIN_WIN de reussite."""
    t = s["stats"]["test"]
    if gain_mode():
        return t.get("avg", 0) > 0 and t.get("pf", 0) >= CFG["MIN_PF"]
    return t.get("win", 0) >= CFG["SIGNAL_MIN_WIN"]


def pair_proven(s, sym):
    """Strategie prouvee sur cette paire : >= 5 trades jamais vus et >= 92 % (une perte toleree si peu de trades)."""
    pa = (s.get("robustesse") or {}).get("par_actif", {}).get(sym)
    if not pa or pa.get("trades", 0) < 5:
        return False
    if gain_mode():                                 # gros gains : la strategie gagne de l'argent sur cette paire
        return pa.get("gain_moyen", 0) > 0
    n = pa["trades"]
    losses = n - round(pa["reussite"] * n)
    return losses <= max(1, int((1 - CFG["SIGNAL_MIN_WIN"]) * n))


def numerology_signals(st, lab, kind, tf, sym, fx):
    """Strategie numerologie, a part : un signal seulement si une de ses strategies est active, >= 92 % prouve
       (setup elite) et prouvee sur cette paire."""
    sent = 0
    for cl in lab["classes"]:
        if cl["kind"] != kind or cl["tf"] != tf:
            continue
        for s in cl["strategies"]:
            if s.get("agent") != "numerologie" or s["id"] in set(st["retired"]) or not active(s, fx["last"]):
                continue
            if not (is_elite(s) and pair_proven(s, sym) and is_strong(s)):
                continue
            if any(o["sym"] == sym for o in st["open"]):
                return sent
            key = "num|%s|%s|%d|%d" % (sym, tf, s["dir"], int(fx["ts"]))
            if key in st["alerts"]:
                continue
            st["alerts"][key] = time.time()
            send_signal(st, kind, tf, sym, s["dir"], fx, s, {"numerologie": s}, {}, lab, elite=True)
            sent += 1
            return sent
    return sent


def is_elite(s):
    """Setup elite : >= 92 % sur >= 100 trades jamais vus, borne basse prudente >= 85 %, gain positif."""
    t = s["stats"]["test"]
    if gain_mode():                                 # gros gains : >= 100 trades jamais vus, facteur de profit eleve,
        return (t.get("n", 0) >= CFG["ELITE_N"] and t.get("pf", 0) >= CFG["ELITE_PF"]       # gain prudent positif
                and t.get("avg", 0) - 1.64 * t.get("se", 1) > 0)
    return (t.get("win", 0) >= CFG["ELITE_WIN"] and t.get("n", 0) >= CFG["ELITE_N"] and t.get("avg", 0) > 0
            and s.get("wilson_test", 0) >= CFG["ELITE_WILSON"])


def wanted(lab, kind, tf):
    return ((any(cl["kind"] == kind and cl["tf"] == tf and cl["strategies"] for cl in lab["classes"])
             and not CFG["SEULEMENT_ACTIVEES"])
            or any(a["kind"] == kind and a["tf"] == tf for a in lab.get("activees", [])))


def _mem_path():
    root = os.environ.get("DUKA_CACHE", "")
    return os.path.join(root, "armee_memoire.pkl") if root else None


def mem_save():
    """Garde les historiques et sources deja calcules pour la prochaine armee (demarrage presque immediat)."""
    p = _mem_path()
    if not p:
        return
    import pickle
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p + ".tmp", "wb") as f:
            pickle.dump({"t": time.time(), "CACHE": dict(CACHE), "VOLUME": dict(data.VOLUME), "POSC": dict(POSC),
                         "MACRO": dict(MACRO), "OFC": dict(OFC), "FUND": dict(FUND)}, f, protocol=4)
        os.replace(p + ".tmp", p)
    except Exception as ex:
        log.warning("memoire armee : %s", ex)


def mem_load():
    p = _mem_path()
    if not p or not os.path.exists(p):
        return 0
    import pickle
    try:
        with open(p, "rb") as f:
            m = pickle.load(f)
    except Exception as ex:
        log.warning("memoire armee illisible : %s", ex)
        return 0
    age = time.time() - m.get("t", 0)
    if age > 36 * 3600:
        return 0
    for k, c in m.get("CACHE", {}).items():         # complete ensuite par les seules dernieres bougies
        CACHE[k] = dict(c, full=time.time() - 3600 if age < 5 * 3600 else 0)
    data.VOLUME.update(m.get("VOLUME", {}))
    POSC.update(m.get("POSC", {}))
    OFC.update(m.get("OFC", {}))
    FUND.update(m.get("FUND", {}))
    if m.get("MACRO", {}).get("ts"):
        MACRO.update(m["MACRO"])
    return len(m.get("CACHE", {}))


def prefetch(lab):
    """Demarrage rapide : tous les historiques telecharges en parallele (6 a la fois) au lieu d'un par un."""
    import concurrent.futures as cf
    jobs = [(sym, kind, tf) for tf in CFG["TIMEFRAMES"] for kind in ("cr", "fx") if UNIVERSE.get(kind)
            and wanted(lab, kind, tf) for sym in UNIVERSE[kind]]
    t0 = time.time()

    def one(j):
        try:
            hist(*j)
            return 1
        except Exception as ex:
            log.warning("prechargement %s %s %s : %s", j[0], j[1], j[2], ex)
            return 0
    # crypto (Coinbase) 6 a la fois, forex (Dukascopy, deja 8 fichiers en parallele par paire) 3 a la fois : en meme temps
    with cf.ThreadPoolExecutor(6) as ex_cr, cf.ThreadPoolExecutor(3) as ex_fx:
        futs = [(ex_cr if j[1] == "cr" else ex_fx).submit(one, j) for j in jobs]
        ok = sum(f.result() for f in futs)
    log.info("prechargement : %d/%d historiques en %.0f s", ok, len(jobs), time.time() - t0)
    return {"historiques": "%d/%d" % (ok, len(jobs)), "secondes": round(time.time() - t0)}


SAVED = [0.0]


def warm(lab, kind, tf, syms, now):
    """Calcule en parallele (4 a la fois) les indicateurs des actifs dont une nouvelle bougie vient de se fermer :
       avant, 60 cryptos l'une apres l'autre prenaient plus de 10 minutes et les dernieres arrivaient trop tard."""
    import concurrent.futures as cf

    def one(sym):
        try:
            features(sym, kind, tf, now, lab.get("idees"))
        except Exception:
            pass                                    # l'erreur sera notee par la boucle normale
    todo = []
    for sym in syms:
        c = CACHE.get((sym, kind, tf))
        f = FEAT.get((sym, kind, tf, len(lab.get("idees") or {})))
        if not c or not f or f["ts"] + 2 * STEP[tf] <= now:     # bougie fermee pas encore calculee
            todo.append(sym)
    if len(todo) > 1:
        with cf.ThreadPoolExecutor(4) as ex:
            list(ex.map(one, todo))


def scan(st, lab, tf, now, kinds=None):
    sent = 0
    for kind, syms in sorted(UNIVERSE.items(), key=lambda kv: kv[0] != "cr"):   # crypto d'abord (24 h/24)
        if kinds and kind not in kinds:
            continue
        if not ((any(cl["kind"] == kind and cl["tf"] == tf and cl["strategies"] for cl in lab["classes"])
                 and not CFG["SEULEMENT_ACTIVEES"])
                or any(a["kind"] == kind and a["tf"] == tf for a in lab.get("activees", []))):
            continue
        warm(lab, kind, tf, syms, now)              # nouvelle bougie : indicateurs de tous les actifs calcules en parallele
        for sym in syms:
            if STOP[0]:
                return sent
            f0 = FEAT.get((sym, kind, tf, len(lab.get("idees") or {})))
            if f0 and EVAL.get((sym, kind, tf)) == f0["ts"] and f0["ts"] + 2 * STEP[tf] > now:
                continue                            # bougie deja CONTROLEE et pas de nouvelle fermee : rien a refaire
            sv = st.setdefault("diag", {}).setdefault("surveillance", {})
            try:
                fx = features(sym, kind, tf, now, lab.get("idees"))
            except Exception as ex:
                log.warning("%s %s : %s", sym, tf, ex)
                sv.setdefault("erreurs", {})[sym] = "%s : %s" % (paris(time.time()).strftime("%d/%m %H:%M"), str(ex)[:110])
                continue
            finally:
                time.sleep(CFG["PAUSE"])
            if fx:
                EVAL[(sym, kind, tf)] = fx["ts"]            # cette bougie est controlee (signal ou non)
            if time.time() - SAVED[0] > 120:                # etat visible pendant le premier tour (long)
                SAVED[0] = time.time()
                save_state(st)
            sv.get("erreurs", {}).pop(sym, None)            # actif de nouveau lu : l'erreur est effacee
            sv.get("erreurs", {}).pop("%s %s" % (sym, tf), None)
            sv["%s %s" % (kind, tf)] = {"dernier_controle": paris(time.time()).strftime("%d/%m %H:%M"), "dernier_actif": sym,
                        "actifs": len(syms), "pistes_actives": sum(1 for a in lab.get("activees", [])
                                                                   if a["kind"] == kind and a["tf"] == tf),
                        "derniere_bougie": paris(fx["ts"]).strftime("%d/%m %H:%M") if fx else None}
            sv.pop(kind, None)
            if fx is None:
                sv.setdefault("erreurs", {})["%s %s" % (sym, tf)] = NOTE.get((sym, kind, tf), "donnees vides")
            # forex : Dukascopy publie chaque heure avec un leger retard ; crypto : jusqu'a la moitie de la bougie (30 min
            # en 1 h), le prix d'entree reste celui de la cloture et l'heure limite est donnee dans le message
            late = max(2400, STEP[tf] // 4) if kind == "fx" else max(450, min(1800, STEP[tf] // 2))
            if tf == "1D":
                late = 20 * 3600                    # bougie du jour fermee la nuit : signal envoye a 9 h (heures de volume)
            t_chk = max(now, time.time()) if abs(time.time() - now) < 3600 else now    # heure reelle (tour long)
            if not fx or t_chk - fx["ts"] - STEP[tf] > late:
                continue                            # bougie trop ancienne : jamais de signal en retard
            near(st, lab, kind, tf, sym, fx)                          # pistes presque pretes (suivi pour le trader)
            sent += activated_signals(st, lab, kind, tf, sym, fx)       # activee par le trader (sous 92 %)
            if CFG["SEULEMENT_ACTIVEES"]:
                continue                            # le trader ne veut que les strategies activees
            votes = council_votes(lab, kind, tf, sym, fx, set(st["retired"]))
            if kind == "cr":
                ob = data.order_book(sym)
                if ob and abs(ob[0]) >= 0.15:
                    d_ = 1 if ob[0] > 0 else -1
                    if votes[d_]:
                        votes[d_]["carnet"] = {"agent": "carnet", "wilson_test": 0, "ob": ob[0]}
            for dr in (1, -1):
                army = next((cl.get("armee", {}).get("achat" if dr > 0 else "vente") for cl in lab["classes"]
                             if cl["kind"] == kind and cl["tf"] == tf), None)
                agents = votes[dr]
                strong = {a: s for a, s in agents.items() if a != "carnet"}
                if not strong or len(votes[-dr]) >= len(agents):
                    continue                        # personne de solide, ou autant de specialistes contre
                army_ok = (army and army.get("avg", 0) > 0 and army.get("n", 0) >= 20 and len(agents) >= CFG["CONSENSUS"]
                           and (army.get("pf", 0) >= CFG["MIN_PF"] if gain_mode()
                                else army.get("win", 0) >= CFG["SIGNAL_MIN_WIN"]))
                elite = [s for s in strong.values() if is_elite(s)]
                if not army_ok and not elite:
                    continue                        # ni vote de l'armee prouve >= 92 %, ni setup elite
                if not any(is_strong(s) for s in (elite or strong.values())):
                    continue                        # la strategie meneuse doit passer la barre (92 % ou gros gains)
                if not any(pair_proven(s, sym) for s in (elite or strong.values())):
                    continue                        # ... et l'avoir prouve sur CETTE paire (>= 5 trades jamais vus)
                if any(o["sym"] == sym for o in st["open"]):
                    continue                        # un seul trade a la fois par paire : pas d'empilement du meme risque
                key = "%s|%s|%d|%d" % (sym, tf, dr, int(fx["ts"]))
                if key in st["alerts"]:
                    continue
                st["alerts"][key] = time.time()
                lead = max(elite or strong.values(), key=lambda s: (s["stats"]["test"]["win"], s["wilson_test"]))
                send_signal(st, kind, tf, sym, dr, fx, lead, agents, votes[-dr], lab, elite=not army_ok)
                sent += 1
    return sent


def send_signal(st, kind, tf, sym, dr, fx, lead, agents, against, lab, elite=False, activee=False):
    ex, px, atr = lead["exit"], fx["close"], fx["atr"]
    sl, tp = px - dr * ex["sl"] * atr, px + dr * ex["tp"] * atr
    t_exit = fx["ts"] + STEP[tf] * (ex["h"] + 1)
    t = lead["stats"]["test"]
    army = next((cl.get("armee", {}).get("achat" if dr > 0 else "vente") for cl in lab["classes"]
                 if cl["kind"] == kind and cl["tf"] == tf), None)
    L = [banner(kind, sym),
         ("🎯 <b>STRATÉGIE ACTIVÉE PAR TOI</b> · <b>%s</b> (sous ta barre des %.0f %%)" % (TG.esc(lead.get("activee", "")),
                                                                                     100 * CFG["MIN_WIN"])
          if activee else
          "🪖 <b>ARMÉE DU LABO</b> · stratégie gros gains validée (objectif ≥ stop, gagnante sur des trades jamais vus) : <b>%s</b>"
          % TG.esc(IND.AGENTS.get(lead.get("agent", ""), lead.get("agent", ""))) if gain_mode() else
          "🪖 <b>ARMÉE DU LABO</b> · stratégie validée (≥ %.0f %% sur des trades jamais vus) : <b>%s</b>" % (
              100 * CFG["MIN_WIN"], TG.esc(IND.AGENTS.get(lead.get("agent", ""), lead.get("agent", ""))))),
         "%s <b>%s %s</b>%s" % ("🟢" if dr > 0 else "🔴", "ACHETER" if dr > 0 else "VENDRE", TG.esc(data.label(sym, kind)),
                                "" if dr > 0 else "  <i>(tu gagnes si le prix baisse)</i>"),
         ("🎯 Prouvée à %.1f %% sur %d trades jamais vus, gagnante sur cette paire · bougies %s" % (
             100 * lead["stats"]["test"]["win"], lead["stats"]["test"]["n"], tf.lower()) if activee else
          "🔢 <b>Stratégie numérologie (à part)</b> · prouvée à plus de 92 %% · bougies %s" % tf.lower()
          if lead.get("agent") == "numerologie" else
          ("⭐ <b>Setup élite</b> (gros gains prouvés seul sur plus de 100 trades) · bougies %s" if gain_mode() else
           "⭐ <b>Setup élite</b> (prouvé seul à plus de 92 %%) · bougies %s") % tf.lower() if elite else
          "🪖 Armée : <b>%d spécialistes d'accord</b> · bougies %s" % (len(agents), tf.lower())), SEP,
         "💵 Entrée au prix du marché : <code>%s</code>  · à faire avant %s" % (fmt(px), paris(max(fx["ts"] + STEP[tf] + 600, time.time() + 600)).strftime("%H:%M")),
         "🛑 Sortie de sécurité : <code>%s</code>  (%+.2f %%)" % (fmt(sl), -dr * 100 * ex["sl"] * atr / px),
         ("💰 <b>1) Encaisse la moitié</b> à <code>%s</code>  (%+.2f %%) puis remonte ton stop <b>au prix d'entrée</b>.\n"
          "📈 <b>2) Laisse courir l'autre moitié</b> : stop suiveur à %.1f ATR du meilleur cours de clôture, je t'envoie "
          "chaque déplacement." % (fmt(px + dr * ex["tp1"] * atr), dr * 100 * ex["tp1"] * atr / px, ex["trail"])
          if ex.get("tp1") else
          "📈 <b>Pas d'objectif fixe : on laisse courir la tendance.</b> Stop suiveur à %.1f ATR du meilleur cours de "
          "clôture : je t'envoie chaque déplacement du stop." % ex["trail"] if ex.get("trail") else
          "🎯 Prendre le gain : <code>%s</code>  (%+.2f %%)" % (fmt(tp), dr * 100 * ex["tp"] * atr / px)),
         "⏰ Fermer de toute façon à %s (Paris)" % paris(t_exit).strftime("%d/%m %H:%M" if t_exit - time.time() > 20 * 3600 else "%H:%M"),
         "<i>Chez ton courtier le prix peut différer un peu : garde les mêmes % depuis ton prix d'entrée.</i>", SEP,
         ("✅ Meilleure stratégie : <b>%.0f %% de trades gagnants</b> (moitié encaissée) sur %d trades jamais vus, gain "
          "moyen <b>%+.2f × le risque</b> (frais inclus)" % (100 * t["win"], t["n"], t["avg"])
          if ex.get("tp1") else
          "✅ Meilleure stratégie : <b>%.0f %% de réussite</b>, gain moyen <b>%+.2f × le risque</b>, facteur de profit "
          "<b>%.2f</b> sur %d trades jamais vus (frais inclus)" % (100 * t["win"], t["avg"], t.get("pf", 0), t["n"])
          if ex.get("trail") else
          "✅ Meilleure stratégie : <b>%.0f %% de réussite</b>, gain visé <b>%.1f × le stop</b>, facteur de profit <b>%.2f</b> "
          "sur %d trades jamais vus (frais inclus)" % (100 * t["win"], ex["tp"] / ex["sl"], t.get("pf", 0), t["n"])
          if gain_mode() else
          "✅ Taux de réussite de la meilleure stratégie : <b>%.0f %%</b> sur %d trades jamais vus (gain moyen %+.2f × le risque, frais inclus)"
          % (100 * t["win"], t["n"], t["avg"]))]
    if lead.get("heat"):
        L.append(TG.esc(lead["heat"]))
    pa = (lead.get("robustesse") or {}).get("par_actif", {}).get(sym)
    if activee and pa:
        L.append("📊 Sur %s seul : %.0f %% sur %d trades jamais vus" % (TG.esc(data.label(sym, kind)), 100 * pa["reussite"],
                                                                         pa["trades"]))
    if army and not elite and not activee:
        L.append("🪖 Vote de l'armée testé : %.0f %% de réussite sur %d trades jamais vus" % (100 * army["win"], army["n"]))
    if lead.get("mc"):
        L.append("🎲 %d simulations de 20 trades : positif dans %.0f %% des cas" % (CFG["SIMS"], 100 * lead["mc"]["p_gain"]))
    L.append("👥 D'accord : " + TG.esc(", ".join(IND.AGENTS.get(a, a) for a in agents)))
    if against:
        L.append("⚠️ Pas d'accord : " + TG.esc(", ".join(IND.AGENTS.get(a, a) for a in against)))
    L.append("<i>Signal seulement : c'est toi qui passes l'ordre. Pas un conseil.</i>")
    TG.send("\n".join(L))
    st["open"].append({"kind": kind, "tf": tf, "sym": sym, "dir": dr, "entry": px, "sl": sl, "tp": tp, "t_entry": fx["ts"] + STEP[tf],
                       "t_exit": t_exit, "ids": [s["id"] for s in agents.values() if s.get("id")], "agents": list(agents),
                       "activee": lead.get("activee") if activee else None,
                       "trail": ex.get("trail"), "atr": atr, "sl0": sl, "sl_sent": sl,
                       "tp1": (px + dr * ex["tp1"] * atr) if ex.get("tp1") else None, "frac": ex.get("frac", 0.5)})
    save_state(st)


# ------------------------------------------------------------------ suivi des resultats
def track(st, now):
    still = []
    for o in st["open"]:
        if o.get("tf") not in STEP:                 # trade d'une ancienne unite de temps (15 min retire) : abandonne
            log.info("trade %s %s abandonne (unite retiree)", o.get("sym"), o.get("tf"))
            continue
        try:
            a = hist(o["sym"], o["kind"], o["tf"])
        except Exception:
            still.append(o)
            continue
        a = a[(a[:, 0] >= o["t_entry"]) & (a[:, 0] + STEP[o["tf"]] <= now)]
        dr, res = o["dir"], None
        if o.get("trail"):                          # vraie tendance : stop suiveur recalcule sur les bougies cloturees
            stop = o.get("sl0", o["sl"])
            t1, frac = o.get("tp1"), o.get("frac", 0.5)
            half = t1 is None                       # sans sortie en deux temps : le stop suit des le depart
            part = 0.0
            for b in a:
                if b[0] >= o["t_exit"]:
                    break
                if (dr > 0 and b[3] <= stop) or (dr < 0 and b[2] >= stop):
                    pnl = part + (1 - frac if t1 is not None else 1.0) * (stop - o["entry"]) * dr \
                        if half else (stop - o["entry"]) * dr
                    res = ("GAGNANT" if pnl > 0 else "PERDANT", pnl)
                    break
                if not half and ((dr > 0 and b[2] >= t1) or (dr < 0 and b[3] <= t1)):
                    half, part, stop = True, frac * (t1 - o["entry"]) * dr, o["entry"]
                    if not o.get("half_sent") and not (CFG["SEULEMENT_ACTIVEES"] and not o.get("activee")):
                        TG.send("%s\n💰 <b>Moitié encaissée</b> sur %s à <code>%s</code> : remonte ton stop au prix d'entrée "
                                "<code>%s</code>. Le reste court la tendance, le trade ne peut plus perdre." % (
                                    banner(o["kind"], o["sym"]), TG.esc(data.label(o["sym"], o["kind"])), fmt(t1),
                                    fmt(o["entry"])))
                        o["half_sent"], o["sl_sent"] = True, o["entry"]
                        save_state(st)
                    continue
                if half:
                    new = b[4] - dr * o["trail"] * o["atr"]
                    stop = max(stop, new) if dr > 0 else min(stop, new)
            o["sl"], o["half"], o["part"] = stop, half and t1 is not None, part
            if (res is None and now < o["t_exit"] and abs(stop - o.get("sl_sent", stop)) >= 0.25 * o["atr"]
                    and not (CFG["SEULEMENT_ACTIVEES"] and not o.get("activee"))):
                TG.send("%s\n🔁 <b>Déplace ton stop</b> sur %s : <code>%s</code>  (%s : %+.2f %%)" % (
                    banner(o["kind"], o["sym"]), TG.esc(data.label(o["sym"], o["kind"])), fmt(stop),
                    "gain protégé" if (stop - o["entry"]) * dr > 0 else "perte réduite", 100 * (stop - o["entry"]) * dr / o["entry"]))
                o["sl_sent"] = stop
                save_state(st)
            a = a[:0] if res is not None else a
        for b in (a if not o.get("trail") else []):
            if b[0] >= o["t_exit"]:
                break
            if (dr > 0 and b[3] <= o["sl"]) or (dr < 0 and b[2] >= o["sl"]):
                res = ("PERDANT", (o["sl"] - o["entry"]) * dr)
                break
            if (dr > 0 and b[2] >= o["tp"]) or (dr < 0 and b[3] <= o["tp"]):
                res = ("GAGNANT", (o["tp"] - o["entry"]) * dr)
                break
        if res is None and now >= o["t_exit"] and len(a):
            last = a[a[:, 0] < o["t_exit"]]
            if len(last):
                pnl = (last[-1, 4] - o["entry"]) * dr
                if o.get("half"):                   # moitie deja encaissee + reste a la cloture
                    pnl = o.get("part", 0.0) + (1 - o.get("frac", 0.5)) * pnl
                res = ("GAGNANT" if pnl > 0 else "PERDANT", pnl)
        if res is None:
            if now - o["t_exit"] < 3 * 86400:
                still.append(o)
            continue
        win = res[0] == "GAGNANT"
        pct = 100 * res[1] / o["entry"]
        for sid in o["ids"]:
            lv = st["live"].setdefault(sid, {"n": 0, "k": 0})
            lv["n"] += 1
            lv["k"] += int(win)
            if lv["n"] >= 20 and lv["k"] / lv["n"] < 0.70 and sid not in st["retired"]:
                st["retired"].append(sid)               # la strategie ne tient plus en direct : retiree
        act = o.get("activee")
        st["track"].append({"sym": o["sym"], "kind": o["kind"], "dir": dr, "win": win, "pct": round(pct, 3),
                            "day": paris(now).date().isoformat(), "activee": act})
        if CFG["SEULEMENT_ACTIVEES"] and not act:
            continue                                # ancien trade de l'armee : plus de message
        tot = [x for x in st["track"] if x.get("activee") == act] if act else st["track"]
        k = sum(1 for x in tot if x["win"])
        TG.send("%s\n%s\n%s <b>%s terminé : %s</b> (%+.2f %%)\n📒 Bilan %s : %d trades · %d gagnants · <b>%.0f %% de réussite</b>"
                % (banner(o["kind"], o["sym"]), "🎯 %s" % TG.esc(act).upper() if act else "🪖 ARMÉE DU LABO",
                   "✅" if win else "🛑", TG.esc(o["sym"]), res[0], pct,
                   "de la stratégie" if act else "de l'armée", len(tot), k, 100 * k / len(tot)))
    st["open"] = still


# ------------------------------------------------------------------ le conseil des specialistes (briefing)
READ = {
    "tendance": lambda g: np.sign(g(("tendance", "ema50"))), "ichimoku": lambda g: (1 if g(("ichimoku", "au_dessus_nuage_9_26")) > 0 else -1 if g(("ichimoku", "sous_nuage_9_26")) < 0 else 0),
    "momentum": lambda g: (1 if g(("momentum", "rsi14")) > 55 else -1 if g(("momentum", "rsi14")) < 45 else 0),
    "oscillateurs": lambda g: (-1 if g(("oscillateurs", "stoch_k14")) > 80 else 1 if g(("oscillateurs", "stoch_k14")) < 20 else 0),
    "bandes": lambda g: (1 if g(("bandes", "donchian20")) > 0.8 else -1 if g(("bandes", "donchian20")) < 0.2 else 0),
    "volume": lambda g: np.sign(g(("volume", "obv_pente"))), "vwap": lambda g: np.sign(g(("vwap", "dist_atr"))),
    "profil": lambda g: np.sign(g(("profil", "dist_poc"))),
    "fibonacci": lambda g: (1 if g(("fibonacci", "position_veille")) > 0.618 else -1 if g(("fibonacci", "position_veille")) < 0.382 else 0),
    "amd": lambda g: (1 if g(("amd", "balayage_bas")) > 0 else -1 if g(("amd", "balayage_haut")) > 0 else 0),
    "structure": lambda g: np.sign(g(("structure", "vs_pivot"))), "bougies": lambda g: g(("bougies", "englobante")),
    "statisticien": lambda g: (-1 if g(("statisticien", "zscore20")) > 1.5 else 1 if g(("statisticien", "zscore20")) < -1.5 else 0),
    "retour": lambda g: (-1 if g(("retour", "ecart_std5")) > 1.5 else 1 if g(("retour", "ecart_std5")) < -1.5 else 0),
    "intermarche": lambda g: np.sign(g(("intermarche", "force_relative20"))),
    "macro": lambda g: g(("macro", "spy_au_dessus_mm50")),
}


SCOUT_URL = STRAT_URL.replace("strategies.json", "veille.json")


def scout_digest(st, now):
    """Resume quotidien de l'Eclaireur : nouveautes GitHub et arXiv qui interessent le labo."""
    try:
        v = data.SESSION.get(SCOUT_URL, params={"t": int(now // 3600)}, timeout=20).json()
    except Exception:
        return False
    if not isinstance(v, dict) or not v.get("date") or st["done"].get("veille") == v["date"]:
        return False
    st["done"]["veille"] = v["date"]
    items = v.get("trouvailles", [])
    gh = [x for x in items if x["source"] == "GitHub"][:3]
    ax = [x for x in items if x["source"] == "arXiv"][:3]
    names = lambda ags: ", ".join(IND.AGENTS.get(a, a).split(" (")[0] for a in ags[:3])  # noqa: E731
    L = ["🔭 <b>ÉCLAIREUR · veille du %s</b>" % TG.esc(v["date"]), SEP,
         "Familles d'indicateurs à la mode : <b>%s</b>" % TG.esc(names(list(v.get("poids", {}))) or "-"),
         "→ l'Inventeur teste plus d'idées dans ces familles."]
    for x in gh:
        L.append("⭐ %d · <a href=\"%s\">%s</a> : %s" % (x["etoiles"], TG.esc(x["url"]), TG.esc(x["titre"]), TG.esc(x["resume"][:110])))
    for x in ax:
        L.append("📄 <a href=\"%s\">%s</a>" % (TG.esc(x["url"]), TG.esc(x["titre"][:120])))
    L.append("<i>Idées à tester par le labo, pas des signaux.</i>")
    TG.send("\n".join(L))
    return True


def briefing(st, lab, now):
    L = ["🪖 <b>CONSEIL DES %d SPÉCIALISTES</b> · %s · bougies 1 h" % (len(IND.AGENTS), paris(now).strftime("%d/%m %H:%M")), SEP]
    n_strat = sum(len(c["strategies"]) for c in lab["classes"])
    if n_strat:
        L.append("🧪 Labo : <b>%d stratégies validées</b> (≥ %.0f %% de réussite sur des données jamais vues)" % (n_strat, 100 * CFG["MIN_WIN"]))
    else:
        L.append("🧪 Labo : <b>aucune stratégie n'atteint encore %.0f %%</b> avec un gain positif → l'armée reste silencieuse (voulu)" % (100 * CFG["MIN_WIN"]))
    for kind, sym in MAJORS:
        try:
            fx = features(sym, kind, CFG["TIMEFRAMES"][0], now)
        except Exception:
            fx = None
        if not fx:
            continue
        g = lambda k, fx=fx: fx["last"].get(k, np.nan)  # noqa: E731
        up, dn, flat, why_up, why_dn = 0, 0, 0, [], []
        for ag, fn in READ.items():
            try:
                v = float(fn(g))
            except Exception:
                v = 0.0
            if not np.isfinite(v) or v == 0:
                flat += 1
            elif v > 0:
                up += 1
                why_up.append(IND.AGENTS[ag].split(" (")[0])
            else:
                dn += 1
                why_dn.append(IND.AGENTS[ag].split(" (")[0])
        if kind == "cr":
            ob = data.order_book(sym)
            if ob:
                (why_up if ob[0] > 0.05 else why_dn if ob[0] < -0.05 else []).append("Carnet d'ordres")
                up, dn = up + (ob[0] > 0.05), dn + (ob[0] < -0.05)
        verdict = "plutôt HAUSSIER" if up >= dn + 3 else "plutôt BAISSIER" if dn >= up + 3 else "pas d'accord (attendre)"
        L.append("%s <b>%s</b> : %d ⬆️ · %d ⬇️ · %d ➖ → %s" % ("🟢" if "HAUSS" in verdict else "🔴" if "BAISS" in verdict else "⚪",
                                                        TG.esc(data.label(sym, kind)), up, dn, flat, verdict))
        minority = why_dn if up > dn else why_up
        if minority and verdict != "pas d'accord (attendre)":
            L.append("   contre : " + TG.esc(", ".join(minority[:4])))
    tot = st["track"]
    if tot:
        k = sum(1 for x in tot if x["win"])
        L.append(SEP)
        L.append("📒 Bilan de l'armée : %d trades · %.0f %% de réussite" % (len(tot), 100 * k / len(tot)))
    L.append("<i>Briefing commun des spécialistes · pas un conseil</i>")
    TG.send("\n".join(L))


# ------------------------------------------------------------------ boucle
def _stop(*_):
    STOP[0] = True


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    signal.signal(signal.SIGTERM, _stop)
    st = load_state()
    TG.setup(st)
    lab = load_strategies()
    st["diag"] = {"strategies": sum(len(c["strategies"]) for c in lab["classes"]), "labo": lab.get("genere"),
                  "telegram": "ok" if TG.CHAT["id"] else "destination inconnue : envoyer le code d'appairage au bot"}
    save_state(st)
    try:
        lab = load_strategies()
        repris = mem_load()
        st["diag"]["demarrage"] = dict(prefetch(lab), repris_de_la_veille=repris)
        mem_save()
        save_state(st)
    except Exception as ex:
        log.warning("prechargement : %s", ex)
    end = time.time() + CFG["LOOP_MINUTES"] * 60
    done15 = done_trk = 0
    while not STOP[0] and time.time() < end:
        now = time.time()
        try:
            if not TG.CHAT["id"] and TG.pair_check(st):
                TG.send("✅ <b>Labo institutionnel connecté</b> : les signaux des stratégies activées arriveront ici.")
                save_state(st)
            par = paris(now)
            if CFG["INFOS"] and par.hour in (8, 14, 20) and par.minute < 30 and st["done"].get("brief") != "%s-%d" % (par.date(), par.hour):
                st["done"]["brief"] = "%s-%d" % (par.date(), par.hour)
                briefing(st, lab, now)
                save_state(st)
            q = int(now // 300)
            if CFG["INFOS"] and par.hour >= 9 and st["done"].get("veille_essai") != str(par.date()) and now % 300 < 60:
                st["done"]["veille_essai"] = str(par.date())
                if scout_digest(st, now):
                    save_state(st)
            if q != done15 and now % 300 >= 40:          # controle toutes les 5 minutes
                done15 = q
                lab = load_strategies()
                for kd in ("cr", "fx"):                  # toute la crypto d'abord (rapide), puis le forex
                    for tf_ in CFG["TIMEFRAMES"]:
                        scan(st, lab, tf_, time.time(), kinds=(kd,))   # heure reelle : derniere bougie fermee
                        flush_pending()                 # signaux du controle envoyes classes (pourcentage, volume)
                        save_state(st)
                st.setdefault("diag", {})["truefx"] = {"mode": data.TRUEFX["mode"], "paires_temps_reel": data.TRUEFX["paires"]}
                heat_record_all()                       # historique horaire de la heatmap crypto
                save_state(st)
                if q % 6 == 0:
                    mem_save()                          # toutes les 30 min : memoire pour la prochaine armee
            if st["open"] and int(now // 300) != done_trk and now % 300 >= 50:
                done_trk = int(now // 300)
                track(st, now)
                save_state(st)
        except Exception as ex:
            log.warning("boucle : %s", ex)
        data.poll_truefx()                          # prix temps reel forex toutes les ~5 s
        time.sleep(5)
    if not STOP[0]:
        with open(os.path.join(CFG["STATE_DIR"], "relance"), "w") as f:
            f.write("1")
    save_state(st)
    mem_save()


if __name__ == "__main__":
    main()
