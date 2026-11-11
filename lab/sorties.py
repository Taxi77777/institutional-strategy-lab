"""Test des sorties (11 nov.) : en direct, une perte efface environ 5 gains (stop a 2 ATR, moitie encaissee a 0,75 ATR).
Pour les pistes ACTIVES d'une classe, on garde exactement les memes entrees et on rejoue d'autres sorties : stop plus
serre, premier objectif different, stop suiveur plus court, objectif fixe. Le resultat est compte en ATR par trade
(comparable d'une sortie a l'autre), frais deduits. Choix de la sortie sur apprentissage + validation (80 % anciens),
verdict sur le test (20 % recents). Attention : les pistes ont ete choisies avec leur sortie actuelle sur ce meme test,
ce qui avantage la sortie actuelle (comparaison prudente).   python -m lab.sorties cr 1H"""
import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("lab")
GRID = ([{"id": "s%g_%g_%g" % (sl, t1, tr), "tp": t1, "tp1": t1, "frac": 0.5, "sl": sl, "trail": tr, "h": 240}
         for sl in (1.0, 1.5, 2.0) for t1 in (0.5, 0.75, 1.0) for tr in (1.5, 3.0)] +
        [{"id": "f%g_%g" % (tp, sl), "tp": tp, "sl": sl, "h": 96} for tp in (0.75, 1.0, 1.5) for sl in (1.0, 1.5)])


def run(kind, tf="1H"):
    from lab import live, miner
    with open("strategies.json") as f:
        v = json.load(f)
    live.ACT_URL = "https://invalide.invalid/activation.json"      # regles locales (activation.json du depot)
    acts = [a for a in live.activated(v) if a["kind"] == kind and a["tf"] == tf]
    print("PISTES ACTIVES", kind, tf, len(acts), flush=True)
    if not acts:
        return
    cur = {}
    for a in acts:
        e = dict(a["exit"], id="cur_" + a["exit"]["id"])
        cur[e["id"]] = e
    exits = list(cur.values()) + GRID
    miner.EXITS[:] = exits                                         # build() simule toutes ces sorties
    specs = list((v.get("idees") or {}).values())
    ds = miner.build(kind, tf, idea_specs=specs or None)
    ts, aid, split, step = ds["ts"], ds["aid"], ds["split"], ds["step"]
    sym_id = {s: i for i, s in enumerate(ds["symbols"])}
    acc = {e["id"]: {0: [], 2: []} for e in exits}                 # 0 : apprentissage + validation ; 2 : test
    ref = {0: [], 2: []}
    used = 0
    for a in acts:
        try:
            m = miner.full_mask(ds, [tuple(c) for c in a["conds"]])
        except KeyError as ex:
            print("PISTE IGNOREE (indicateur absent)", a["id"], ex, flush=True)
            continue
        ok = np.isin(aid, [sym_id[s] for s in a["paires_ok"] if s in sym_id])
        cid = "cur_" + a["exit"]["id"]
        m = m & ok & ds["R"][(cid, a["dir"])][2] & (split >= 0)
        m = miner.thin(m, aid, ts, a["exit"]["h"] * step)           # memes entrees pour toutes les sorties
        used += 1
        for part, sel in ((0, m & (split <= 1)), (2, m & (split == 2))):
            ref[part].append(ds["R"][(cid, a["dir"])][0][sel] * a["exit"]["sl"])
            for e in exits:
                r, win, valid = ds["R"][(e["id"], a["dir"])]
                acc[e["id"]][part].append(np.where(valid[sel], r[sel] * e["sl"], 0.0))

    def st(x):
        g = np.concatenate(x) if x else np.array([])
        if not len(g):
            return {"n": 0}
        w, l = g[g > 0], g[g <= 0]
        return {"n": int(len(g)), "reussite": round(float((g > 0).mean()), 3), "atr_par_trade": round(float(g.mean()), 4),
                "gain_moyen": round(float(w.mean()), 3) if len(w) else 0, "perte_moyenne": round(float(l.mean()), 3) if len(l) else 0}
    res = {"marche": kind, "unite": tf, "genere": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "pistes": used,
           "actuelle": {"anciens": st(ref[0]), "test": st(ref[2])}, "sorties": []}
    print("ACTUELLE | ANCIENS", json.dumps(res["actuelle"]["anciens"]), "| TEST", json.dumps(res["actuelle"]["test"]), flush=True)
    for e in GRID:
        row = {"sortie": e, "anciens": st(acc[e["id"]][0]), "test": st(acc[e["id"]][2])}
        res["sorties"].append(row)
        print("SORTIE", e["id"], "| ANCIENS", json.dumps(row["anciens"]), "| TEST", json.dumps(row["test"]), flush=True)
    best = max(res["sorties"], key=lambda r: r["anciens"].get("atr_par_trade", -9))
    hi = [r for r in res["sorties"] if r["anciens"].get("reussite", 0) >= 0.8]
    best_hi = max(hi, key=lambda r: r["anciens"]["atr_par_trade"]) if hi else None
    res["choisie"] = best
    res["choisie_80"] = best_hi
    print("CHOISIE (sur les anciens)", best["sortie"]["id"], "-> TEST", json.dumps(best["test"]), flush=True)
    if best_hi:
        print("CHOISIE >= 80 % (sur les anciens)", best_hi["sortie"]["id"], "-> TEST", json.dumps(best_hi["test"]), flush=True)
    os.makedirs("resultats", exist_ok=True)
    with open("resultats/sorties_%s_%s.json" % (kind, tf), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run(sys.argv[1], sys.argv[2].upper() if len(sys.argv) > 2 else "1H")
