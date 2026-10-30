"""Verification independante d'une strategie publiee sur des donnees de courtier (Dukascopy), comparee a Yahoo.
   python -m lab.validate  -> validation.json"""
import datetime
import json
import logging
import os
import numpy as np
from lab import data, indicators as IND, sim
from lab.config import UNIVERSE, CFG
from lab.miner import thin

log = logging.getLogger("lab")


def strat_best():
    with open("strategies.json") as f:
        lab = json.load(f)
    cl = next(c for c in lab["classes"] if c["kind"] == "fx" and c["tf"] == "1H")
    return max(cl["strategies"], key=lambda s: (s["stats"]["test"]["win"], s["wilson_test"]))


def replay(a, s):
    F, atr = IND.compute(a, "fx")
    m = np.ones(len(a), dtype=bool)
    for ag, nm, lo, hi in s["conds"]:
        x = F.get(ag, {}).get(nm)
        if x is None:
            return None
        m &= (x > lo) & (x <= hi)
    r, win, valid = sim.outcomes(a, atr, s["exit"], None, CFG["COST_BPS"]["fx"])[s["dir"]]
    keep = thin(m & valid, np.zeros(len(a), dtype=int), a[:, 0], s["exit"]["h"] * 3600)
    return a[:, 0][keep], r[keep], win[keep]


def summary(ts, r, w):
    if len(r) == 0:
        return {"n": 0}
    return {"n": int(len(r)), "win": round(float(w.mean()), 3), "avg": round(float(r.mean()), 3),
            "total_R": round(float(r.sum()), 1)}


def main():
    s = strat_best()
    days = int(os.environ.get("VAL_DAYS", "730"))
    out = {"genere": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"), "strategie": s["id"],
           "conds": s["conds"], "exit": s["exit"], "labo_test": s["stats"]["test"], "jours": days, "sources": {}}
    for src in ("dukascopy", "yahoo"):
        allt, allr, allw, per = [], [], [], {}
        for sym in UNIVERSE["fx"]:
            try:
                a = data.dukascopy(sym, 3600, days) if src == "dukascopy" else data.bars(sym, "fx", "1H")
                res = replay(np.asarray(a, dtype=float), s)
            except Exception as ex:
                log.warning("%s %s : %s", src, sym, ex)
                continue
            if res is None:
                continue
            t, r, w = res
            per[sym] = summary(t, r, w)
            allt.append(t); allr.append(r); allw.append(w)
            log.info("%s %s : %s", src, sym, per[sym])
        if not allr:
            out["sources"][src] = {"erreur": "aucune donnee"}
            continue
        t, r, w = np.concatenate(allt), np.concatenate(allr), np.concatenate(allw)
        cut = np.quantile(t, 0.8)
        out["sources"][src] = {"total": summary(t, r, w), "derniers_20pct": summary(t[t >= cut], r[t >= cut], w[t >= cut]),
                               "par_paire": per}
    with open("validation.json", "w") as f:
        json.dump(out, f, indent=1)
    log.info("termine : %s", {k: v.get("total") for k, v in out["sources"].items()})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
