"""L'armee en direct : toutes les 15 minutes, les 20 specialistes regardent chaque actif ; un signal ne part que si
au moins CONSENSUS specialistes ont une strategie VALIDEE par le labo (>= 80 % sur donnees jamais vues) active dans
le meme sens, sans majorite contraire. Briefing commun (« conseil des 20 ») a 8 h, 14 h et 20 h (Paris).
Signaux uniquement : aucun ordre n'est passe. Niveaux en prix ET en % pour s'appliquer chez n'importe quel courtier."""
import datetime
import json
import logging
import os
import signal
import time
from zoneinfo import ZoneInfo

import numpy as np

from . import data, ideas as IDEAS, indicators as IND, telegram as TG
from .config import CFG, UNIVERSE
from .miner import BENCH, align, macro_series

log = logging.getLogger("lab")
PARIS = ZoneInfo("Europe/Paris")
STATE_FILE = "lab_state.json"
STEP = {"15M": 900, "1H": 3600}
KEEP = {"15M": 3000, "1H": 2500}
CACHE = {}
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
    LAB_CACHE.update(ts=time.time(), v=v)
    return v


# ------------------------------------------------------------------ donnees en direct
def hist(sym, kind, tf):
    key = (sym, kind, tf)
    c = CACHE.get(key)
    now = time.time()
    if not c or now - c["full"] > 6 * 3600:
        a = data.bars(sym, kind, tf)
        CACHE[key] = {"a": a[-KEEP[tf]:], "full": now}
    else:
        new = data.bars(sym, kind, tf, rng="5d" if tf == "15M" else "1mo")
        rows = {int(r[0]): r for r in c["a"]}
        rows.update({int(r[0]): r for r in new})
        c["a"] = np.array([rows[k] for k in sorted(rows)])[-KEEP[tf]:]
    return CACHE[key]["a"]


MACRO = {"ts": 0, "v": {}}


def macro():
    if time.time() - MACRO["ts"] > 6 * 3600:
        MACRO["v"] = macro_series(data.daily)
        MACRO["ts"] = time.time()
    return MACRO["v"]


def features(sym, kind, tf, now, specs=None):
    a = hist(sym, kind, tf)
    a = a[a[:, 0] + STEP[tf] <= now]                # bougies cloturees uniquement
    if len(a) < 300:
        return None
    refs = None
    if sym != BENCH[kind]:
        try:
            b = hist(BENCH[kind], kind, tf)
            refs = {"bench": align(b[:, 0], b[:, 4], a[:, 0])}
        except Exception:
            refs = None
    mac = {k: align(v[0], v[1], a[:, 0]) for k, v in macro().items()}
    F, atr = IND.compute(a, kind, refs, mac)
    for nm, sp in (specs or {}).items():
        x = IDEAS.apply_spec(F, sp)
        if x is not None:
            F["inventeur"][nm] = x
    last = {(ag, nm): float(x[-1]) for ag, d in F.items() for nm, x in d.items()}
    return {"a": a, "last": last, "atr": float(atr[-1]), "ts": float(a[-1, 0]), "close": float(a[-1, 4])}


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
            if active(s, fx["last"]):
                best = votes[s["dir"]].get(s["agent"])
                if not best or s["wilson_test"] > best["wilson_test"]:
                    votes[s["dir"]][s["agent"]] = s
    return votes


def scan(st, lab, tf, now):
    sent = 0
    for kind, syms in UNIVERSE.items():
        if not any(cl["kind"] == kind and cl["tf"] == tf and cl["strategies"] for cl in lab["classes"]):
            continue
        for sym in syms:
            if STOP[0]:
                return sent
            try:
                fx = features(sym, kind, tf, now, lab.get("idees"))
            except Exception as ex:
                log.warning("%s %s : %s", sym, tf, ex)
                continue
            finally:
                time.sleep(CFG["PAUSE"])
            if not fx or now - fx["ts"] - STEP[tf] > 600:
                continue                            # bougie trop ancienne : jamais de signal en retard
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
                if not army or army.get("win", 0) < CFG["MIN_WIN"] or army.get("avg", 0) <= 0 or army.get("n", 0) < 20:
                    continue                        # le vote de l'armee lui-meme doit avoir prouve >= 80 % et un gain
                agents = votes[dr]
                strong = {a: s for a, s in agents.items() if a != "carnet"}
                if len(agents) < CFG["CONSENSUS"] or not strong or len(votes[-dr]) >= len(agents):
                    continue
                key = "%s|%s|%d|%d" % (sym, tf, dr, int(fx["ts"]))
                if key in st["alerts"]:
                    continue
                st["alerts"][key] = time.time()
                lead = max(strong.values(), key=lambda s: s["wilson_test"])
                send_signal(st, kind, tf, sym, dr, fx, lead, agents, votes[-dr], lab)
                sent += 1
    return sent


def send_signal(st, kind, tf, sym, dr, fx, lead, agents, against, lab):
    ex, px, atr = lead["exit"], fx["close"], fx["atr"]
    sl, tp = px - dr * ex["sl"] * atr, px + dr * ex["tp"] * atr
    t_exit = fx["ts"] + STEP[tf] * (ex["h"] + 1)
    t = lead["stats"]["test"]
    army = next((cl.get("armee", {}).get("achat" if dr > 0 else "vente") for cl in lab["classes"]
                 if cl["kind"] == kind and cl["tf"] == tf), None)
    L = [banner(kind, sym),
         "%s <b>%s %s</b>%s" % ("🟢" if dr > 0 else "🔴", "ACHETER" if dr > 0 else "VENDRE", TG.esc(data.label(sym, kind)),
                                "" if dr > 0 else "  <i>(tu gagnes si le prix baisse)</i>"),
         "🪖 Armée : <b>%d spécialistes d'accord</b> · bougies %s" % (len(agents), tf.lower()), SEP,
         "💵 Entrée au prix du marché : <code>%s</code>  · à faire avant %s" % (fmt(px), paris(fx["ts"] + STEP[tf] + 600).strftime("%H:%M")),
         "🛑 Sortie de sécurité : <code>%s</code>  (%+.2f %%)" % (fmt(sl), -dr * 100 * ex["sl"] * atr / px),
         "🎯 Prendre le gain : <code>%s</code>  (%+.2f %%)" % (fmt(tp), dr * 100 * ex["tp"] * atr / px),
         "⏰ Fermer de toute façon à %s (Paris)" % paris(t_exit).strftime("%H:%M"),
         "<i>Chez ton courtier le prix peut différer un peu : garde les mêmes % depuis ton prix d'entrée.</i>", SEP,
         "✅ Taux de réussite de la meilleure stratégie : <b>%.0f %%</b> sur %d trades jamais vus (gain moyen %+.2f × le risque, frais inclus)"
         % (100 * t["win"], t["n"], t["avg"])]
    if army:
        L.append("🪖 Vote de l'armée testé : %.0f %% de réussite sur %d trades jamais vus" % (100 * army["win"], army["n"]))
    if lead.get("mc"):
        L.append("🎲 %d simulations de 20 trades : positif dans %.0f %% des cas" % (CFG["SIMS"], 100 * lead["mc"]["p_gain"]))
    L.append("👥 D'accord : " + TG.esc(", ".join(IND.AGENTS.get(a, a) for a in agents)))
    if against:
        L.append("⚠️ Pas d'accord : " + TG.esc(", ".join(IND.AGENTS.get(a, a) for a in against)))
    L.append("<i>Signal seulement : c'est toi qui passes l'ordre. Pas un conseil.</i>")
    TG.send("\n".join(L))
    st["open"].append({"kind": kind, "tf": tf, "sym": sym, "dir": dr, "entry": px, "sl": sl, "tp": tp, "t_entry": fx["ts"] + STEP[tf],
                       "t_exit": t_exit, "ids": [s["id"] for s in agents.values() if s.get("id")], "agents": list(agents)})
    save_state(st)


# ------------------------------------------------------------------ suivi des resultats
def track(st, now):
    still = []
    for o in st["open"]:
        try:
            a = hist(o["sym"], o["kind"], o["tf"])
        except Exception:
            still.append(o)
            continue
        a = a[(a[:, 0] >= o["t_entry"]) & (a[:, 0] + STEP[o["tf"]] <= now)]
        dr, res = o["dir"], None
        for b in a:
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
        st["track"].append({"sym": o["sym"], "kind": o["kind"], "dir": dr, "win": win, "pct": round(pct, 3),
                            "day": paris(now).date().isoformat()})
        tot = st["track"]
        k = sum(1 for x in tot if x["win"])
        TG.send("%s\n%s <b>%s terminé : %s</b> (%+.2f %%)\n📒 Bilan de l'armée : %d trades · %d gagnants · <b>%.0f %% de réussite</b>"
                % (banner(o["kind"], o["sym"]), "✅" if win else "🛑", TG.esc(o["sym"]), res[0], pct, len(tot), k, 100 * k / len(tot)))
    st["open"] = still


# ------------------------------------------------------------------ le conseil des 20 (briefing)
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


def briefing(st, lab, now):
    L = ["🪖 <b>CONSEIL DES 20 SPÉCIALISTES</b> · %s" % paris(now).strftime("%d/%m %H:%M"), SEP]
    n_strat = sum(len(c["strategies"]) for c in lab["classes"])
    if n_strat:
        L.append("🧪 Labo : <b>%d stratégies validées</b> (≥ %.0f %% de réussite sur des données jamais vues)" % (n_strat, 100 * CFG["MIN_WIN"]))
    else:
        L.append("🧪 Labo : <b>aucune stratégie n'atteint encore %.0f %%</b> avec un gain positif → l'armée reste silencieuse (voulu)" % (100 * CFG["MIN_WIN"]))
    for kind, sym in MAJORS:
        try:
            fx = features(sym, kind, "1H", now)
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
    end = time.time() + CFG["LOOP_MINUTES"] * 60
    done15 = done_trk = 0
    while not STOP[0] and time.time() < end:
        now = time.time()
        try:
            if not TG.CHAT["id"] and TG.pair_check(st):
                TG.send("✅ <b>Labo institutionnel connecté</b> : l'armée des 20 spécialistes enverra ses signaux ici.")
                save_state(st)
            par = paris(now)
            if par.hour in (8, 14, 20) and par.minute < 30 and st["done"].get("brief") != "%s-%d" % (par.date(), par.hour):
                st["done"]["brief"] = "%s-%d" % (par.date(), par.hour)
                briefing(st, lab, now)
                save_state(st)
            q = int(now // 900)
            if q != done15 and now % 900 >= 40:
                done15 = q
                lab = load_strategies()
                scan(st, lab, "15M", now)
                if now % 3600 < 900:
                    scan(st, lab, "1H", now)
                save_state(st)
            if st["open"] and int(now // 300) != done_trk and now % 300 >= 50:
                done_trk = int(now // 300)
                track(st, now)
                save_state(st)
        except Exception as ex:
            log.warning("boucle : %s", ex)
        time.sleep(5)
    if not STOP[0]:
        with open(os.path.join(CFG["STATE_DIR"], "relance"), "w") as f:
            f.write("1")
    save_state(st)


if __name__ == "__main__":
    main()
