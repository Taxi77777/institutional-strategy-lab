"""Test AMD en direct sur la crypto (demande du trader, 15 nov.) : AUCUN ordre, AUCUN envoi Telegram.
Toutes les 15 minutes : dernieres bougies 15 min Coinbase de chaque crypto (>= 5 M$ / 24 h), setups AMD du jour
(lab.amd_test), chaque setup est note DES qu'il apparait avec l'heure ou il a ete vu (preuve qu'il n'est pas vu apres
coup), puis suivi jusqu'au stop, a l'objectif ou a 21 h UTC, avec la regle stricte du test historique.
Journal : donnees/amd_live/journal.json.   python -m lab.amd_live"""
import json
import logging
import os
import time

import numpy as np

from lab import amd_test
from lab.liq2liq import _atr
from lab.ob_retour import play

log = logging.getLogger("lab")
PATH = "donnees/amd_live/journal.json"
SUIVIES = [{"declencheur": dc, "serre": 0.0, "cible": t, "fin_manip": 10, "tp1": p}
           for dc in ("retour", "ouverture") for t in ("bord", "veille", "extension") for p in (0.5, 1.0)]


def vid(v):
    return "%s-%s-%s-%s" % (v["declencheur"], v["cible"], v["fin_manip"], v["tp1"])


def run():
    from lab import data
    from lab.config import CFG, UNIVERSE
    j = {"trades": {}}
    if os.path.exists(PATH):
        with open(PATH) as f:
            j = json.load(f)
    now = time.time()
    vol_min = CFG.get("VOLUME_MIN_CRYPTO", 5e6)
    vus = 0
    for sym in UNIVERSE["cr"]:
        try:
            a = data.coinbase(sym, 900, 4)
        except Exception as ex:
            log.warning("%s : %s", sym, ex)
            continue
        a = a[a[:, 0] + 900 <= now]                              # bougies fermees seulement
        if len(a) < 200:
            continue
        last = a[a[:, 0] >= now - 86400]
        vol = float((last[:, 5] * last[:, 4]).sum())
        atr = _atr(a)
        vus += 1
        for v in SUIVIES:
            for st in amd_test.setups(a, atr, v["declencheur"], 0.0, v["cible"], v["fin_manip"]):
                t_e = float(a[st["i"], 0])
                if t_e < now - 2 * 86400:
                    continue
                key = "%s|%s|%d|%d" % (sym, vid(v), int(t_e), st["dir"])
                tr = j["trades"].get(key)
                if tr is None:
                    if vol < vol_min:
                        continue
                    tr = j["trades"][key] = {"actif": sym, "variante": vid(v), "sens": "achat" if st["dir"] > 0 else "vente",
                                             "entree": st["entry"], "stop": st["stop"], "objectif": st["target"],
                                             "t_entree": t_e, "vu_a": now, "a_temps": bool(now - t_e <= 1800),
                                             "volume_24h_M": round(vol / 1e6, 1), "statut": "ouvert"}
                if tr["statut"] != "ouvert":
                    continue
                p = play(a, st, "cr", max_bars=max(1, st["fin"] - st["i"] + 1), tp1_r=v["tp1"])
                if p is None:
                    tr["statut"] = "annule"
                    continue
                r, win, end = p
                if end >= len(a) - 1 and st["fin"] > len(a) - 1:
                    continue                                      # encore en cours
                tr.update({"statut": "ferme", "r": round(float(r), 3), "gagne": bool(win), "t_sortie": float(a[end, 0])})
    res = {}
    for tr in j["trades"].values():
        if tr["statut"] != "ferme" or not tr.get("a_temps"):
            continue
        s = res.setdefault(tr["variante"], {"trades": 0, "gagnes": 0, "total_R": 0.0})
        s["trades"] += 1
        s["gagnes"] += int(tr["gagne"])
        s["total_R"] = round(s["total_R"] + tr["r"], 3)
    for s in res.values():
        s["reussite"] = round(s["gagnes"] / s["trades"], 3)
        s["gain_moyen_R"] = round(s["total_R"] / s["trades"], 3)
    j["bilan"] = res
    j["maj"] = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(now))
    j["cryptos_vues"] = vus
    j["ouverts"] = sum(1 for t in j["trades"].values() if t["statut"] == "ouvert")
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "w") as f:
        json.dump(j, f, indent=1)
    print("AMD DIRECT", j["maj"], "| cryptos vues", vus, "| ouverts", j["ouverts"], "| bilan", json.dumps(res), flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run()
