"""Kronos a-t-il raison ? (9 nov.) L'armee bloque tout signal contraire a l'avis de Kronos : on verifie sur l'historique
de ses previsions (donnees/kronos/<marche>_1D_<actif>.csv, prevision faite a la cloture de chaque bougie jour, jamais
de donnee future) si le sens prevu est le bon le lendemain et sur 5 jours.   python -m lab.kronos_test cr"""
import json
import os
import sys
import time

import numpy as np


def run(kind, tf=""):
    from lab import data, kronos_ia
    from lab.config import UNIVERSE
    rows = []
    for sym in UNIVERSE[kind]:
        try:
            k = kronos_ia.load(kind, "1D", sym)
            a = data.long_bars(sym, kind, "1D")
            if kind == "fx":
                a = a[((a[:, 0] // 86400 + 3) % 7) != 6]
            if len(k) < 30 or len(a) < 60:
                continue
            pos = {int(t): i for i, t in enumerate(a[:, 0])}
            atr = kronos_ia._atr(a)
            for t, p in k:
                i = pos.get(int(t))
                if i is None or i + 5 >= len(a) or not np.isfinite(atr[i]) or atr[i] <= 0 or p == 0:
                    continue
                rows.append((t, p, (a[i + 1, 4] - a[i, 4]) / atr[i], (a[i + 5, 4] - a[i, 4]) / atr[i]))
        except Exception as ex:
            print("ERREUR", sym, ex, flush=True)
    r = np.array(rows)
    out = {"marche": kind, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "previsions": int(len(r))}
    print("PREVISIONS", kind, len(r), "du", time.strftime("%d/%m/%Y", time.gmtime(r[:, 0].min())), "au",
          time.strftime("%d/%m/%Y", time.gmtime(r[:, 0].max())), flush=True)
    for nom, m in (("toutes", np.ones(len(r), bool)), ("fortes (>0,5 ATR)", abs(r[:, 1]) > 0.5),
                   ("tres fortes (>1 ATR)", abs(r[:, 1]) > 1.0), ("hausse prevue", r[:, 1] > 0),
                   ("baisse prevue", r[:, 1] < 0)):
        if m.sum() < 30:
            continue
        s = np.sign(r[m, 1])
        d = {"n": int(m.sum()), "bon_sens_1j": round(float(np.mean(s * r[m, 2] > 0)), 3),
             "bon_sens_5j": round(float(np.mean(s * r[m, 3] > 0)), 3),
             "gain_1j_atr": round(float(np.mean(s * r[m, 2])), 4), "gain_5j_atr": round(float(np.mean(s * r[m, 3])), 4)}
        out[nom] = d
        print("KRONOS", kind, nom, json.dumps(d), flush=True)
    base = {"hausse_1j": round(float(np.mean(r[:, 2] > 0)), 3), "hausse_5j": round(float(np.mean(r[:, 3] > 0)), 3)}
    out["marche_seul"] = base
    print("MARCHE", kind, json.dumps(base), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/kronos_test_%s.json" % kind, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    run(*sys.argv[1:])
